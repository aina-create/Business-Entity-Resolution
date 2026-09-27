"""Disk-backed evaluator for README-format entity match TSV files."""

from __future__ import annotations

import argparse
import csv
import sqlite3
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]


def _load_ground_truth(conn, path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        expected = {"source1_entity_id", "matched_entity_ids"}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError(f"Ground truth must contain {sorted(expected)}")
        batch = []
        entities = []
        for row in reader:
            s1 = row["source1_entity_id"].strip()
            entities.append((s1,))
            ids = (x.strip() for x in (row["matched_entity_ids"] or "").split(","))
            batch.extend((s1, target) for target in ids if target)
            if len(entities) >= 100_000:
                if batch:
                    conn.executemany("INSERT INTO truth VALUES (?, ?)", batch)
                conn.executemany("INSERT OR IGNORE INTO entities VALUES (?)", entities)
                batch.clear()
                entities.clear()
        if batch:
            conn.executemany("INSERT INTO truth VALUES (?, ?)", batch)
        if entities:
            conn.executemany("INSERT OR IGNORE INTO entities VALUES (?)", entities)


def _load_predictions(conn, path: Path, table: str, id_column: str):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        expected = {"source1_entity_id", id_column}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain {sorted(expected)}")
        batch = []
        for row in reader:
            s1 = row["source1_entity_id"].strip()
            ids = (x.strip() for x in (row[id_column] or "").split(","))
            batch.extend((s1, target) for target in ids if target)
            if len(batch) >= 100_000:
                conn.executemany(f"INSERT OR IGNORE INTO {table} VALUES (?, ?)", batch)
                batch.clear()
        if batch:
            conn.executemany(f"INSERT OR IGNORE INTO {table} VALUES (?, ?)", batch)


def evaluate_files(ground_truth: Path, predictions: Path, candidate_file: Path | None = None) -> dict:
    """Return macro F0.5 and micro summaries using temporary on-disk SQLite."""
    with tempfile.TemporaryDirectory(prefix="er_evaluation_") as temp_dir:
        conn = sqlite3.connect(str(Path(temp_dir) / "evaluation.sqlite"))
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("CREATE TABLE truth (s1 TEXT, target TEXT, PRIMARY KEY(s1,target))")
        conn.execute("CREATE TABLE pred (s1 TEXT, target TEXT, PRIMARY KEY(s1,target))")
        conn.execute("CREATE TABLE candidates (s1 TEXT, target TEXT, PRIMARY KEY(s1,target))")
        conn.execute("CREATE TABLE entities (s1 TEXT PRIMARY KEY)")
        _load_ground_truth(conn, ground_truth)
        _load_predictions(conn, predictions, "pred", "matched_entity_ids")
        if candidate_file:
            _load_predictions(conn, candidate_file, "candidates", "candidate_entity_ids")
        conn.commit()
        conn.execute("INSERT OR IGNORE INTO entities SELECT DISTINCT s1 FROM pred")
        conn.execute("CREATE TEMP TABLE gt_counts AS SELECT s1, COUNT(*) n FROM truth GROUP BY s1")
        conn.execute("CREATE TEMP TABLE pred_counts AS SELECT s1, COUNT(*) n FROM pred GROUP BY s1")
        conn.execute("CREATE TEMP TABLE tp_counts AS SELECT p.s1, COUNT(*) n FROM pred p JOIN truth t USING(s1,target) GROUP BY p.s1")
        f_sum = 0.0
        for actual, predicted, true_positive in conn.execute(
            "SELECT COALESCE(g.n,0), COALESCE(p.n,0), COALESCE(t.n,0) FROM entities e "
            "LEFT JOIN gt_counts g USING(s1) LEFT JOIN pred_counts p USING(s1) LEFT JOIN tp_counts t USING(s1)"
        ):
            if actual == 0 and predicted == 0:
                score = 1.0
            elif predicted == 0 or true_positive == 0:
                score = 0.0
            else:
                precision = true_positive / predicted
                recall = true_positive / actual
                score = 1.25 * precision * recall / (0.25 * precision + recall)
            f_sum += score
        total_gt = conn.execute("SELECT COALESCE(SUM(n),0) FROM gt_counts").fetchone()[0]
        total_pred = conn.execute("SELECT COALESCE(SUM(n),0) FROM pred_counts").fetchone()[0]
        total_tp = conn.execute("SELECT COALESCE(SUM(n),0) FROM tp_counts").fetchone()[0]
        entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        result = {
            "entities": entity_count,
            "macro_f0_5": f_sum / entity_count if entity_count else 0.0,
            "micro_precision": total_tp / total_pred if total_pred else 0.0,
            "micro_recall": total_tp / total_gt if total_gt else 0.0,
            "true_matches": total_tp,
            "predicted_matches": total_pred,
            "ground_truth_matches": total_gt,
        }
        if candidate_file:
            candidate_count = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
            recalled = conn.execute("SELECT COUNT(*) FROM truth t JOIN candidates c USING(s1,target)").fetchone()[0]
            result["candidate_pairs"] = candidate_count
            result["blocking_recall"] = recalled / total_gt if total_gt else 0.0
        conn.close()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, default=BASE_DIR / "dataset/train/train_ground_truth.tsv")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--candidates", type=Path)
    args = parser.parse_args()
    result = evaluate_files(args.ground_truth, args.predictions, args.candidates)
    for key, value in result.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value:,}")


if __name__ == "__main__":
    main()
