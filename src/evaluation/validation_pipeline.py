"""End-to-end held-out validation on the training split.

The development default uses 100k Source-1 rows to keep iterations practical,
while indexing every Source-2/3 training record. Set --max-source1 0 to use all
Source-1 rows. The deterministic every-fifth-row split separates model fitting
and threshold selection by entity, never by candidate pair.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import tempfile
import time
from pathlib import Path

import pandas as pd

from src.blocking.scalable_candidate_generation import block_keys, normalize_series
from src.evaluation.evaluate import evaluate_files
from src.model.predict import PAIR_COLUMNS, predict_file
from src.model.train import train_file
from src.preprocessing.normalization import FEATURE_COLUMNS, create_features

BASE_DIR = Path(__file__).resolve().parents[2]
TRAIN_DIR = BASE_DIR / "dataset" / "train"
CHUNK_SIZE = 50_000
BLOCK_CAP = 50
PAIR_HEADER = PAIR_COLUMNS + ["match"]


def _keys(name: str, address: str, country: str) -> set[str]:
    return block_keys(name, address, country)


def _index_target(conn: sqlite3.Connection, path: Path, chunk_size: int):
    print(f"Indexing {path.name} ...", flush=True)
    count = 0
    for frame in pd.read_csv(path, sep="\t", usecols=["entity_id", "business_name", "business_address", "country"],
                             dtype=str, keep_default_na=False, chunksize=chunk_size):
        names = normalize_series(frame.business_name)
        addresses = normalize_series(frame.business_address)
        countries = normalize_series(frame.country)
        records, blocks = [], []
        for eid, name, address, country in zip(frame.entity_id, names, addresses, countries):
            records.append((eid, name, address, country))
            blocks.extend((key, eid) for key in _keys(name, address, country))
        conn.executemany("INSERT INTO records VALUES (?, ?, ?, ?)", records)
        conn.executemany("INSERT INTO blocks VALUES (?, ?)", blocks)
        count += len(frame)
        if count % (chunk_size * 10) == 0:
            conn.commit()
            print(f"  indexed {count:,} rows", flush=True)
    conn.commit()
    print(f"  indexed {count:,} rows", flush=True)


def _write_labeled_pairs(conn, fit_df: pd.DataFrame, valid_df: pd.DataFrame,
                         train_dir: Path, temp: Path, block_cap: int, chunk_size: int):
    fit_ids = set(fit_df.entity_id)
    valid_ids = set(valid_df.entity_id)
    roles = {eid: "fit" for eid in fit_ids}
    roles.update({eid: "valid" for eid in valid_ids})
    pair_paths = {role: temp / f"{role}_pairs.tsv" for role in ("fit", "valid")}
    writers = {}
    streams = {}
    for role, path in pair_paths.items():
        streams[role] = path.open("w", encoding="utf-8", newline="")
        writers[role] = csv.writer(streams[role], delimiter="\t", lineterminator="\n")
        writers[role].writerow(PAIR_HEADER if role == "fit" else PAIR_COLUMNS)
    valid_truth_path = temp / "valid_ground_truth.tsv"
    valid_truth = valid_truth_path.open("w", encoding="utf-8", newline="")
    truth_writer = csv.writer(valid_truth, delimiter="\t", lineterminator="\n")
    truth_writer.writerow(["source1_entity_id", "matched_entity_ids"])
    valid_set = set(valid_df.entity_id)
    for row in pd.read_csv(train_dir / "train_ground_truth.tsv", sep="\t", dtype=str,
                           keep_default_na=False, chunksize=chunk_size):
        for s1, targets in zip(row.source1_entity_id, row.matched_entity_ids):
            if s1 in valid_set:
                truth_writer.writerow([s1, targets])
    valid_truth.close()

    conn.execute("CREATE TEMP TABLE query_keys (s1 TEXT, block_key TEXT, PRIMARY KEY(s1,block_key))")
    conn.execute("CREATE INDEX query_keys_lookup ON query_keys(block_key,s1)")
    s1_lookup = {}
    processed = 0
    all_s1 = pd.concat([fit_df, valid_df], ignore_index=True)
    for start in range(0, len(all_s1), chunk_size):
        frame = all_s1.iloc[start:start + chunk_size]
        names = normalize_series(frame.business_name)
        addresses = normalize_series(frame.business_address)
        countries = normalize_series(frame.country)
        key_rows = []
        s1_lookup.clear()
        for eid, raw_name, raw_address, raw_country, name, address, country in zip(
                frame.entity_id, frame.business_name, frame.business_address, frame.country,
                names, addresses, countries):
            s1_lookup[eid] = (roles[eid], raw_name, raw_address, raw_country)
            key_rows.extend((eid, key) for key in _keys(name, address, country))
        conn.execute("DELETE FROM query_keys")
        conn.executemany("INSERT OR IGNORE INTO query_keys VALUES (?, ?)", key_rows)
        cursor = conn.execute(
            "SELECT q.s1, r.entity_id, MAX(r.name), MAX(r.address), MAX(r.country) "
            "FROM query_keys q JOIN block_sizes z ON z.block_key=q.block_key AND z.n<=? "
            "JOIN blocks b ON b.block_key=q.block_key "
            "JOIN records r ON r.entity_id=b.entity_id "
            "GROUP BY q.s1, r.entity_id ORDER BY q.s1, r.entity_id", (block_cap,))
        current_s1 = None
        candidate_rows = []
        seen_targets = set()

        def emit(s1, pairs):
            role, s1_name, s1_address, s1_country = s1_lookup[s1]
            truth = {r[0] for r in conn.execute("SELECT target FROM truth WHERE s1=?", (s1,))} if role == "fit" else set()
            writer = writers[role]
            for target_id, target_name, target_address, target_country in pairs:
                row = [s1, target_id, s1_name, target_name, s1_address, target_address,
                       s1_country, target_country]
                if role == "fit":
                    writer.writerow(row + [int(target_id in truth)])
                else:
                    writer.writerow(row)

        for s1, target_id, target_name, target_address, target_country in cursor:
            if current_s1 is not None and s1 != current_s1:
                emit(current_s1, candidate_rows)
                candidate_rows, seen_targets = [], set()
            current_s1 = s1
            if target_id not in seen_targets:
                seen_targets.add(target_id)
                candidate_rows.append((target_id, target_name, target_address, target_country))
        if current_s1 is not None:
            emit(current_s1, candidate_rows)
        processed += len(frame)
        if processed % (chunk_size * 2) == 0:
            print(f"Generated candidates for {processed:,} Source 1 records", flush=True)
    for stream in streams.values():
        stream.close()
    return pair_paths, valid_truth_path


def _make_feature_file(pair_path: Path, feature_path: Path, chunk_size: int):
    first = True
    for frame in pd.read_csv(pair_path, sep="\t", usecols=PAIR_HEADER, dtype=str,
                             keep_default_na=False, chunksize=chunk_size):
        labels = pd.to_numeric(frame.pop("match"), errors="raise").astype("uint8")
        features = create_features(frame)
        features["match"] = labels.to_numpy()
        features.to_csv(feature_path, sep="\t", index=False, mode="w" if first else "a",
                        header=first)
        first = False
    if first:
        raise ValueError("No training candidate pairs were generated; cannot train the matcher")


def _exact_baseline(pair_path: Path, source1_path: Path, ground_truth_path: Path,
                    candidate_path: Path, chunk_size: int) -> float:
    """Score exact normalized name + country as a transparent validation baseline."""
    with tempfile.TemporaryDirectory(prefix="er_baseline_") as tmp:
        conn = sqlite3.connect(str(Path(tmp) / "baseline.sqlite"))
        conn.execute("CREATE TABLE pred (s1 TEXT, target TEXT, PRIMARY KEY(s1,target))")
        for frame in pd.read_csv(pair_path, sep="\t", usecols=PAIR_COLUMNS, dtype=str,
                                 keep_default_na=False, chunksize=chunk_size):
            left_name = normalize_series(frame.source1_business_name)
            right_name = normalize_series(frame.candidate_business_name)
            left_country = normalize_series(frame.source1_country)
            right_country = normalize_series(frame.candidate_country)
            mask = (left_name == right_name) & (left_name != "") & (left_country == right_country)
            conn.executemany("INSERT OR IGNORE INTO pred VALUES (?,?)",
                             zip(frame.loc[mask, "source1_entity_id"], frame.loc[mask, "candidate_entity_id"]))
        conn.commit()
        output = Path(tmp) / "baseline_matching.tsv"
        with output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(["source1_entity_id", "matched_entity_ids"])
            for chunk in pd.read_csv(source1_path, sep="\t", usecols=["entity_id"], dtype=str,
                                     keep_default_na=False, chunksize=chunk_size):
                for s1 in chunk.entity_id:
                    ids = [row[0] for row in conn.execute("SELECT target FROM pred WHERE s1=? ORDER BY target", (s1,))]
                    writer.writerow([s1, ",".join(ids)])
        metrics = evaluate_files(ground_truth_path, output, candidate_path)
        print(f"Exact-name baseline macro_F0.5={metrics['macro_f0_5']:.6f}")
        conn.close()
        return metrics["macro_f0_5"]


def _save_summary(output_dir: Path, threshold: float, model_metrics: dict, baseline_score: float):
    summary = {
        "threshold": threshold,
        "model_macro_f0_5": model_metrics["macro_f0_5"],
        "exact_name_baseline_macro_f0_5": baseline_score,
        "blocking_recall": model_metrics.get("blocking_recall"),
        "micro_precision": model_metrics["micro_precision"],
        "micro_recall": model_metrics["micro_recall"],
        "validation_entities": model_metrics["entities"],
        "candidate_pairs": model_metrics.get("candidate_pairs"),
    }
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def run_validation(train_dir: Path, output_dir: Path, model_path: Path, *,
                   max_source1: int = 100_000, block_cap: int = BLOCK_CAP,
                   chunk_size: int = CHUNK_SIZE, thresholds=None, threshold_only: bool = False):
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    pair_paths = {role: work_dir / f"{role}_pairs.tsv" for role in ("fit", "valid")}
    valid_truth = work_dir / "valid_ground_truth.tsv"
    source1_valid_path = work_dir / "valid_source1.tsv"
    thresholds = thresholds or [round(x / 100, 2) for x in range(20, 81, 5)]

    if threshold_only:
        for needed in (pair_paths["valid"], valid_truth, source1_valid_path, model_path):
            if not needed.is_file():
                raise FileNotFoundError(f"Threshold-only mode requires {needed}; run validation once first")
        results = []
        for threshold in thresholds:
            predict_file(pair_paths["valid"], model_path, output_dir,
                         source1_file=source1_valid_path, threshold=threshold,
                         chunk_size=chunk_size)
            metrics = evaluate_files(valid_truth, output_dir / "matching_results.tsv",
                                     output_dir / "candidate_pairs.tsv")
            results.append((metrics["macro_f0_5"], threshold, metrics))
            print(f"threshold={threshold:.2f} macro_F0.5={metrics['macro_f0_5']:.6f} "
                  f"blocking_recall={metrics.get('blocking_recall', 0):.6f}")
        best_score, best_threshold, best_metrics = max(results, key=lambda item: item[0])
        predict_file(pair_paths["valid"], model_path, output_dir,
                     source1_file=source1_valid_path, threshold=best_threshold,
                     chunk_size=chunk_size)
        baseline_score = _exact_baseline(pair_paths["valid"], source1_valid_path, valid_truth,
                                         output_dir / "candidate_pairs.tsv", chunk_size)
        _save_summary(output_dir, best_threshold, best_metrics, baseline_score)
        print(f"Best threshold: {best_threshold:.2f} (macro F0.5={best_score:.6f})")
        print(f"Exact-name baseline: macro F0.5={baseline_score:.6f}")
        print(f"Validation outputs: {output_dir}")
        return best_metrics

    source1_path = train_dir / "train_source1.tsv"
    source1 = pd.read_csv(source1_path, sep="\t", usecols=["entity_id", "business_name", "business_address", "country"],
                          dtype=str, keep_default_na=False, nrows=max_source1 or None)
    source1 = source1.reset_index(drop=True)
    is_valid = source1.index.to_series().mod(5).eq(0).to_numpy()
    valid_df = source1.loc[is_valid].copy()
    fit_df = source1.loc[~is_valid].copy()
    print(f"Source 1 development sample: {len(source1):,} ({len(fit_df):,} fit, {len(valid_df):,} validation)")

    with tempfile.TemporaryDirectory(prefix="er_validation_") as tmp_dir:
        temp = Path(tmp_dir)
        conn = sqlite3.connect(str(temp / "validation.sqlite"))
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=FILE")
        conn.execute("CREATE TABLE records (entity_id TEXT PRIMARY KEY, name TEXT, address TEXT, country TEXT)")
        conn.execute("CREATE TABLE blocks (block_key TEXT, entity_id TEXT)")
        conn.execute("CREATE TABLE truth (s1 TEXT, target TEXT, PRIMARY KEY(s1,target))")
        _index_target(conn, train_dir / "train_source2.tsv", chunk_size)
        _index_target(conn, train_dir / "train_source3.tsv", chunk_size)
        conn.execute("CREATE INDEX blocks_lookup ON blocks(block_key,entity_id)")
        conn.execute("CREATE INDEX records_lookup ON records(entity_id)")
        conn.execute("CREATE TABLE block_sizes AS SELECT block_key, COUNT(*) n FROM blocks GROUP BY block_key")
        conn.execute("CREATE UNIQUE INDEX block_sizes_lookup ON block_sizes(block_key)")
        selected = set(source1.entity_id)
        batch = []
        for frame in pd.read_csv(train_dir / "train_ground_truth.tsv", sep="\t", dtype=str,
                                 keep_default_na=False, chunksize=chunk_size):
            for s1, targets in zip(frame.source1_entity_id, frame.matched_entity_ids):
                if s1 in selected and targets:
                    batch.extend((s1, target.strip()) for target in targets.split(",") if target.strip())
            if len(batch) >= 100_000:
                conn.executemany("INSERT OR IGNORE INTO truth VALUES (?,?)", batch)
                batch.clear()
        if batch:
            conn.executemany("INSERT OR IGNORE INTO truth VALUES (?,?)", batch)
        conn.commit()
        pair_paths, valid_truth = _write_labeled_pairs(conn, fit_df, valid_df, train_dir, work_dir, block_cap, chunk_size)
        feature_path = work_dir / "fit_features.tsv"
        _make_feature_file(pair_paths["fit"], feature_path, chunk_size)
        train_file(feature_path, model_path, epochs=8, chunk_size=chunk_size)

        valid_df[["entity_id"]].to_csv(source1_valid_path, sep="\t", index=False)
        results = []
        for threshold in thresholds:
            predict_file(pair_paths["valid"], model_path, output_dir,
                         source1_file=source1_valid_path, threshold=threshold,
                         chunk_size=chunk_size)
            metrics = evaluate_files(valid_truth, output_dir / "matching_results.tsv",
                                     output_dir / "candidate_pairs.tsv")
            results.append((metrics["macro_f0_5"], threshold, metrics))
            print(f"threshold={threshold:.2f} macro_F0.5={metrics['macro_f0_5']:.6f} "
                  f"blocking_recall={metrics.get('blocking_recall', 0):.6f}")
        best_score, best_threshold, best_metrics = max(results, key=lambda item: item[0])
        # Re-emit final validation artifacts at the selected threshold.
        predict_file(pair_paths["valid"], model_path, output_dir,
                     source1_file=source1_valid_path, threshold=best_threshold,
                     chunk_size=chunk_size)
        baseline_score = _exact_baseline(pair_paths["valid"], source1_valid_path, valid_truth,
                                         output_dir / "candidate_pairs.tsv", chunk_size)
        _save_summary(output_dir, best_threshold, best_metrics, baseline_score)
        print(f"Best threshold: {best_threshold:.2f} (macro F0.5={best_score:.6f})")
        print(f"Exact-name baseline: macro F0.5={baseline_score:.6f}")
        print(f"Validation outputs: {output_dir}")
        conn.close()
    print(f"Elapsed: {time.time() - started:.1f}s")
    return best_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "validation")
    parser.add_argument("--model", type=Path, default=BASE_DIR / "models" / "entity_matcher.npz")
    parser.add_argument("--max-source1", type=int, default=100_000,
                        help="Development sample size; 0 uses all Source 1 rows")
    parser.add_argument("--block-cap", type=int, default=BLOCK_CAP)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--threshold-only", action="store_true",
                        help="Reuse saved validation candidate pairs/model to retune thresholds without rebuilding indexes")
    parser.add_argument("--thresholds", nargs="+", type=float,
                        help="Thresholds to evaluate (default: 0.20 through 0.80 by 0.05)")
    args = parser.parse_args()
    run_validation(args.train_dir, args.output_dir, args.model,
                   max_source1=args.max_source1, block_cap=args.block_cap,
                   chunk_size=args.chunk_size, threshold_only=args.threshold_only,
                   thresholds=args.thresholds)


if __name__ == "__main__":
    main()
