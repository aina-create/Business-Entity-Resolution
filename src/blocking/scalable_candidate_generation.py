"""Disk-backed candidate generation for the ML Challenge 2026 data.

Reads the README's dataset/{train,test} layout. SQLite holds blocking indexes so
all 10M+ target records need not be retained as pandas DataFrames. Candidate and
matching files are emitted in the required one-row-per-Source-1 TSV format.

Blocks above the configured frequency cap are omitted, including exact-name
blocks, to prevent common names from creating enormous and unreliable candidate
sets. The matching baseline accepts only same-country exact normalized names
that survive this cap.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import tempfile
import time
import unicodedata
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "output"
CHUNK_SIZE = 50_000
MAX_BLOCK_SIZE = 50


def normalize_series(values: pd.Series) -> pd.Series:
    values = values.fillna("").astype(str).str.lower().str.strip()
    values = values.map(lambda x: unicodedata.normalize("NFKC", x))
    return (values.str.replace(r"[^\w\s]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip())


def block_keys(name: str, address: str, country: str):
    """Multi-pass name/address blocks shared with held-out validation."""
    words = name.split()
    keys = set()
    if name and country:
        keys.add("e\x1f" + country + "\x1f" + name)
    if country and len(words) >= 2:
        keys.add("n2\x1f" + country + "\x1f" + " ".join(words[:2]))
        keys.add("n2s\x1f" + country + "\x1f" + " ".join(sorted(words[:2])))
    if country and words:
        keys.add("w1\x1f" + country + "\x1f" + words[0])
        keys.add("wl\x1f" + country + "\x1f" + words[-1])
        for word in set(words):
            if len(word) >= 4:
                keys.add("t4\x1f" + country + "\x1f" + word[:4])
    if country and len(address) >= 8:
        keys.add("a\x1f" + country + "\x1f" + address[:18])
    if country:
        for number in set(re.findall(r"\d{4,}", address)):
            keys.add("d\x1f" + country + "\x1f" + number)
    return keys


def add_target_file(conn, path: Path, source: str, chunk_size: int):
    print(f"Indexing {path.name} ...", flush=True)
    count = 0
    for frame in pd.read_csv(path, sep="\t", usecols=["entity_id", "business_name", "business_address", "country"], dtype=str, keep_default_na=False, chunksize=chunk_size):
        names = normalize_series(frame["business_name"])
        addresses = normalize_series(frame["business_address"])
        countries = normalize_series(frame["country"])
        rows = []
        for entity_id, raw_name, raw_address, raw_country, name, address, country in zip(
                frame.entity_id, frame.business_name, frame.business_address, frame.country,
                names, addresses, countries):
            for key in block_keys(name, address, country):
                rows.append((source, key, entity_id, int(key.startswith("e\x1f")),
                             raw_name, raw_address, raw_country))
        conn.executemany("INSERT INTO blocks(source, block_key, entity_id, is_exact, business_name, business_address, country) VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        count += len(frame)
        if count % (chunk_size * 10) == 0:
            conn.commit()
            print(f"  indexed {count:,} rows", flush=True)
    conn.commit()
    print(f"  indexed {count:,} rows", flush=True)


def build_submission(test_dir: Path, output_dir: Path, chunk_size: int, max_block_size: int,
                     model_path: Path | None = None, threshold: float | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    s1_path = test_dir / "test_source1.tsv"
    if not s1_path.is_file():
        raise FileNotFoundError(s1_path)
    started = time.time()
    # Keep the temporary database on disk; a private temp directory is cleaned
    # automatically even if generation fails.
    with tempfile.TemporaryDirectory(prefix="entity_resolution_") as temp_dir:
        conn = sqlite3.connect(str(Path(temp_dir) / "blocks.sqlite"))
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=FILE")
        conn.execute("CREATE TABLE blocks (source TEXT, block_key TEXT, entity_id TEXT, is_exact INTEGER, business_name TEXT, business_address TEXT, country TEXT)")
        add_target_file(conn, test_dir / "test_source2.tsv", "S2", chunk_size)
        add_target_file(conn, test_dir / "test_source3.tsv", "S3", chunk_size)
        conn.execute("CREATE INDEX block_lookup ON blocks(block_key, source)")
        conn.execute("CREATE INDEX exact_lookup ON blocks(block_key, is_exact)")
        conn.execute("CREATE TABLE block_sizes AS SELECT block_key, COUNT(*) AS n FROM blocks GROUP BY block_key")
        conn.execute("CREATE UNIQUE INDEX block_sizes_lookup ON block_sizes(block_key)")
        conn.execute("CREATE TEMP TABLE query_keys (s1 TEXT, block_key TEXT, PRIMARY KEY(s1, block_key))")
        conn.execute("CREATE INDEX query_keys_lookup ON query_keys(block_key, s1)")
        conn.commit()

        candidate_path = output_dir / "candidate_pairs.tsv"
        matching_path = output_dir / "matching_results.tsv"
        pairwise_path = Path(temp_dir) / "pairwise_candidates.tsv"
        use_model = model_path is not None and model_path.is_file()
        n_source1 = 0
        with candidate_path.open("w", encoding="utf-8", newline="") as candidate_out, matching_path.open("w", encoding="utf-8", newline="") as matching_out:
            candidate_out.write("source1_entity_id\tcandidate_entity_ids\n")
            matching_out.write("source1_entity_id\tmatched_entity_ids\n")
            pairwise_out = pairwise_path.open("w", encoding="utf-8", newline="") if use_model else None
            pairwise_writer = csv.writer(pairwise_out, delimiter="\t", lineterminator="\n") if pairwise_out else None
            if pairwise_writer:
                pairwise_writer.writerow(["source1_entity_id", "candidate_entity_id", "source1_business_name",
                                          "candidate_business_name", "source1_business_address", "candidate_business_address",
                                          "source1_country", "candidate_country"])
            for frame in pd.read_csv(s1_path, sep="\t", usecols=["entity_id", "business_name", "business_address", "country"], dtype=str, keep_default_na=False, chunksize=chunk_size):
                names = normalize_series(frame["business_name"])
                addresses = normalize_series(frame["business_address"])
                countries = normalize_series(frame["country"])
                key_rows = []
                s1_values = {}
                for entity_id, raw_name, raw_address, raw_country, name, address, country in zip(
                        frame.entity_id, frame.business_name, frame.business_address, frame.country,
                        names, addresses, countries):
                    s1_values[entity_id] = (raw_name, raw_address, raw_country)
                    keys = block_keys(name, address, country)
                    key_rows.extend((entity_id, key) for key in keys)
                    if not keys:
                        key_rows.append((entity_id, "\x1fNO_BLOCK_KEY\x1f"))
                conn.execute("DELETE FROM query_keys")
                conn.executemany("INSERT OR IGNORE INTO query_keys VALUES (?, ?)", key_rows)
                conn.commit()

                # One indexed join per input chunk instead of several SQL
                # round-trips for every Source 1 record. All oversized blocks,
                # including frequent exact names, are excluded before joining.
                cursor = conn.execute(
                    "SELECT q.s1, b.entity_id, MAX(b.is_exact), MAX(b.business_name), "
                    "MAX(b.business_address), MAX(b.country) "
                    "FROM query_keys q LEFT JOIN block_sizes z ON z.block_key=q.block_key "
                    "LEFT JOIN blocks b ON b.block_key=q.block_key "
                    "AND z.n<=? "
                    "GROUP BY q.s1, b.entity_id ORDER BY q.s1, b.entity_id",
                    (max_block_size,),
                )
                current_id = None
                candidates, matches, pair_records = set(), set(), {}

                def emit(s1_id, candidate_ids, match_ids, records):
                    candidate_out.write(f"{s1_id}\t{','.join(sorted(candidate_ids))}\n")
                    matching_out.write(f"{s1_id}\t{','.join(sorted(match_ids))}\n")
                    if pairwise_writer:
                        s1_name, s1_address, s1_country = s1_values[s1_id]
                        for target_id in sorted(records):
                            target_name, target_address, target_country = records[target_id]
                            pairwise_writer.writerow([s1_id, target_id, s1_name, target_name,
                                                      s1_address, target_address, s1_country, target_country])

                for s1_id, target_id, is_exact, target_name, target_address, target_country in cursor:
                    if current_id is not None and s1_id != current_id:
                        emit(current_id, candidates, matches, pair_records)
                        n_source1 += 1
                        candidates, matches, pair_records = set(), set(), {}
                    current_id = s1_id
                    if target_id:
                        candidates.add(target_id)
                        pair_records[target_id] = (target_name, target_address, target_country)
                        if is_exact:
                            matches.add(target_id)
                if current_id is not None:
                    emit(current_id, candidates, matches, pair_records)
                    n_source1 += 1
                if n_source1 and n_source1 % (chunk_size * 2) == 0:
                    print(f"Generated rows for {n_source1:,} Source 1 entities", flush=True)
            if pairwise_out:
                pairwise_out.close()
        conn.close()
        if use_model:
            from src.model.predict import predict_file
            if threshold is None:
                threshold_path = BASE_DIR / "validation" / "validation_metrics.json"
                threshold = json.loads(threshold_path.read_text(encoding="utf-8"))["threshold"] if threshold_path.exists() else 0.5
            print(f"Scoring candidates with {model_path} at threshold {threshold:.2f}", flush=True)
            predict_file(pairwise_path, model_path, output_dir, source1_file=s1_path,
                         threshold=threshold, chunk_size=min(chunk_size, 20_000))
    print(f"Wrote {n_source1:,} rows to {candidate_path} and {matching_path}")
    print(f"Elapsed: {time.time() - started:.1f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=DATA_DIR / "test")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--max-block-size", type=int, default=MAX_BLOCK_SIZE)
    parser.add_argument("--model", type=Path, default=BASE_DIR / "models" / "entity_matcher.npz" if (BASE_DIR / "models" / "entity_matcher.npz").exists() else None,
                        help="Optional trained matcher; when present, scores the generated candidates")
    parser.add_argument("--threshold", type=float, help="Override the threshold saved by validation")
    args = parser.parse_args()
    build_submission(args.test_dir, args.output_dir, args.chunk_size, args.max_block_size,
                     model_path=args.model, threshold=args.threshold)


if __name__ == "__main__":
    main()
