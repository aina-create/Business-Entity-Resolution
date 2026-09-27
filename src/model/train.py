"""Train a compact logistic matcher from pairwise feature TSV data.

Input must contain the feature columns produced by normalization.create_features
and a binary ``match`` label. Keeping candidate generation separate avoids
materializing the Cartesian product of the multi-million-row source files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.normalization import FEATURE_COLUMNS

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = BASE_DIR / "models" / "entity_matcher.npz"


def _sigmoid(z):
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def train_model(features: pd.DataFrame, labels, *, epochs: int = 8, learning_rate: float = 0.08,
                l2: float = 1e-4) -> dict:
    """Fit weighted logistic regression on already materialized pair features."""
    missing = [c for c in FEATURE_COLUMNS if c not in features]
    if missing:
        raise ValueError(f"Training data is missing feature columns: {missing}")
    x = features[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if len(x) != len(y) or not len(y):
        raise ValueError("Features and labels must have equal, non-zero row counts")
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) < 2:
        raise ValueError("Training labels must contain both 0 and 1")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x = (x - mean) / scale
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    positive_weight = (len(y) - y.sum()) / max(y.sum(), 1.0)
    sample_weight = np.where(y == 1, positive_weight, 1.0)
    # Full batch gradient descent is deterministic and stable for this small
    # feature set. Chunked file loading is exposed separately for large tables.
    for _ in range(epochs):
        error = (_sigmoid(x @ weights + bias) - y) * sample_weight
        weights -= learning_rate * ((x.T @ error) / len(y) + l2 * weights)
        bias -= learning_rate * error.mean()
    return {"weights": weights, "bias": np.asarray(bias), "mean": mean, "scale": scale,
            "feature_columns": np.asarray(FEATURE_COLUMNS), "positive_weight": np.asarray(positive_weight)}


def train_file(input_path: Path, model_path: Path, *, epochs: int = 8, chunk_size: int = 50_000):
    """Train using repeated sequential chunk passes, keeping only one chunk in RAM."""
    columns = [*FEATURE_COLUMNS, "match"]
    positives = negatives = 0
    for frame in pd.read_csv(input_path, sep="\t", usecols=["match"], chunksize=chunk_size):
        positives += int((frame.match == 1).sum())
        negatives += int((frame.match == 0).sum())
    total = positives + negatives
    if not positives or not negatives:
        raise ValueError("Training labels must contain both 0 and 1")
    weights = np.zeros(len(FEATURE_COLUMNS), dtype=np.float64)
    bias = 0.0
    positive_weight = negatives / positives
    step = 0
    # Features are bounded [0,1], so no global standardization pass is needed.
    # Use a decaying rate because chunks arrive in file order, not shuffled.
    for epoch in range(epochs):
        for frame in pd.read_csv(input_path, sep="\t", usecols=columns, chunksize=chunk_size):
            x = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
            y = frame["match"].to_numpy(dtype=np.float64)
            sw = np.where(y == 1, positive_weight, 1.0)
            error = (_sigmoid(x @ weights + bias) - y) * sw
            rate = 0.03 / np.sqrt(1.0 + step / 100.0)
            weights -= rate * ((x.T @ error) / max(1, len(y)) + 1e-4 * weights)
            bias -= rate * error.mean()
            step += 1
        print(f"Epoch {epoch + 1}/{epochs}", flush=True)
    model = {"weights": weights, "bias": np.asarray(bias),
             "mean": np.zeros(len(FEATURE_COLUMNS)), "scale": np.ones(len(FEATURE_COLUMNS)),
             "feature_columns": np.asarray(FEATURE_COLUMNS), "positive_weight": np.asarray(positive_weight)}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(model_path, **model)
    print(f"Saved model to {model_path} ({total:,} pairs; {positives:,} positive)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path, help="Pairwise labeled feature TSV")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    args = parser.parse_args()
    train_file(args.features, args.model, epochs=args.epochs, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
