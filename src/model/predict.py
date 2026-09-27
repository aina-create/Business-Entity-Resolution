"""Score pairwise candidate rows and write README-format submission TSVs.

Candidate input columns: ``source1_entity_id``, ``candidate_entity_id``,
``source1_business_name``, ``candidate_business_name``, corresponding address
and country columns. Every row is a pair actually passed to the model.
"""

from __future__ import annotations

import argparse
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.normalization import FEATURE_COLUMNS, create_features
from src.model.train import _sigmoid

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = BASE_DIR / "models" / "entity_matcher.npz"
PAIR_COLUMNS = ["source1_entity_id", "candidate_entity_id", "source1_business_name",
                "candidate_business_name", "source1_business_address", "candidate_business_address",
                "source1_country", "candidate_country"]


def load_model(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def predict_pairs(frame: pd.DataFrame, model: dict) -> np.ndarray:
    features = create_features(frame).loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    mean = np.asarray(model.get("mean", np.zeros(len(FEATURE_COLUMNS))))
    scale = np.asarray(model.get("scale", np.ones(len(FEATURE_COLUMNS))))
    return _sigmoid(((features - mean) / scale) @ model["weights"] + float(model["bias"]))


def predict_file(candidate_file: Path, model_file: Path, output_dir: Path, *,
                 source1_file: Path | None = None, threshold: float = 0.5,
                 chunk_size: int = 20_000):
    """Stream pairwise candidates through the model, then group via disk SQLite."""
    model = load_model(model_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"
    with tempfile.TemporaryDirectory(prefix="entity_prediction_") as tmp:
        conn = sqlite3.connect(str(Path(tmp) / "scores.sqlite"))
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("CREATE TABLE pairs (s1 TEXT, s2 TEXT, score REAL, PRIMARY KEY(s1,s2))")
        conn.execute("CREATE TABLE source_ids (ord INTEGER PRIMARY KEY, s1 TEXT UNIQUE)")
        count = 0
        for frame in pd.read_csv(candidate_file, sep="\t", usecols=PAIR_COLUMNS, dtype=str,
                                 keep_default_na=False, chunksize=chunk_size):
            scores = predict_pairs(frame, model)
            conn.executemany("INSERT OR REPLACE INTO pairs VALUES (?, ?, ?)",
                             ((s1, s2, float(score)) for s1, s2, score in zip(
                                 frame.source1_entity_id, frame.candidate_entity_id, scores)))
            count += len(frame)
            if count % (chunk_size * 10) == 0:
                conn.commit()
                print(f"Scored {count:,} pairs", flush=True)
        conn.commit()
        if source1_file:
            ordinal = 0
            for ids_chunk in pd.read_csv(source1_file, sep="\t", usecols=["entity_id"], dtype=str,
                                         keep_default_na=False, chunksize=chunk_size):
                conn.executemany("INSERT INTO source_ids VALUES (?,?)",
                                 ((ordinal + i, s1) for i, s1 in enumerate(ids_chunk.entity_id)))
                ordinal += len(ids_chunk)
            conn.commit()
        else:
            conn.execute("INSERT INTO source_ids SELECT ROW_NUMBER() OVER (ORDER BY s1), s1 FROM (SELECT DISTINCT s1 FROM pairs)")
            conn.commit()
        with matching_path.open("w", encoding="utf-8", newline="") as mout, candidate_path.open("w", encoding="utf-8", newline="") as cout:
            mout.write("source1_entity_id\tmatched_entity_ids\n")
            cout.write("source1_entity_id\tcandidate_entity_ids\n")
            current_s1 = None
            candidates, matches = [], []

            def emit(s1):
                cout.write(f"{s1}\t{','.join(candidates)}\n")
                mout.write(f"{s1}\t{','.join(matches)}\n")

            cursor = conn.execute(
                "SELECT i.s1, p.s2, p.score FROM source_ids i LEFT JOIN pairs p ON p.s1=i.s1 "
                "ORDER BY i.ord, p.s2")
            for s1, s2, score in cursor:
                if current_s1 is not None and s1 != current_s1:
                    emit(current_s1)
                    candidates, matches = [], []
                current_s1 = s1
                if s2 is not None:
                    candidates.append(s2)
                    if score >= threshold:
                        matches.append(s2)
            if current_s1 is not None:
                emit(current_s1)
        conn.close()
    print(f"Scored {count:,} pairs; wrote {matching_path} and {candidate_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path, help="Pairwise candidate TSV with raw business fields")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "output")
    parser.add_argument("--source1", type=Path, help="Source-1 TSV; include to emit empty rows for entities with no candidates")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=20_000)
    args = parser.parse_args()
    predict_file(args.candidates, args.model, args.output_dir, source1_file=args.source1,
                 threshold=args.threshold, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
