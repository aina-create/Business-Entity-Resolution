"""Shared Unicode normalization and bounded pairwise similarity features."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

FEATURE_COLUMNS = [
    "name_token_jaccard", "name_char3_jaccard", "name_length_ratio",
    "address_token_jaccard", "address_char3_jaccard", "address_length_ratio",
    "country_match", "name_exact",
]


def normalize_text(value) -> str:
    """NFKC, lowercase, punctuation-to-space normalization; retains Unicode letters."""
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKC", str(value)).lower().strip()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def normalize_business_name(value) -> str:
    return normalize_text(value)


def normalize_address(value) -> str:
    return normalize_text(value)


def normalize_country(value) -> str:
    return normalize_text(value)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized columns; safe to apply chunk by chunk."""
    result = df.copy()
    for col, fn in (("business_name", normalize_business_name),
                    ("business_address", normalize_address), ("country", normalize_country)):
        if col in result:
            result[f"{col}_normalized"] = result[col].map(fn)
    return result


def _tokens(value: str) -> set[str]:
    return set(value.split()) if value else set()


def _chargrams(value: str, n: int = 3) -> set[str]:
    compact = value.replace(" ", "")
    if not compact:
        return set()
    if len(compact) < n:
        return {compact}
    return {compact[i:i + n] for i in range(len(compact) - n + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return float(not left and not right)
    return len(left & right) / len(left | right)


def _length_ratio(left: str, right: str) -> float:
    longest = max(len(left), len(right))
    return min(len(left), len(right)) / longest if longest else 1.0


def pair_features(row: dict) -> dict[str, float]:
    """Features for a pairwise record with source1/candidate field suffixes."""
    n1 = normalize_text(row.get("source1_business_name", ""))
    n2 = normalize_text(row.get("candidate_business_name", ""))
    a1 = normalize_text(row.get("source1_business_address", ""))
    a2 = normalize_text(row.get("candidate_business_address", ""))
    c1 = normalize_text(row.get("source1_country", ""))
    c2 = normalize_text(row.get("candidate_country", ""))
    ng1, ng2 = _chargrams(n1), _chargrams(n2)
    ag1, ag2 = _chargrams(a1), _chargrams(a2)
    return {
        "name_token_jaccard": _jaccard(_tokens(n1), _tokens(n2)),
        "name_char3_jaccard": _jaccard(ng1, ng2),
        "name_length_ratio": _length_ratio(n1, n2),
        "address_token_jaccard": _jaccard(_tokens(a1), _tokens(a2)),
        "address_char3_jaccard": _jaccard(ag1, ag2),
        "address_length_ratio": _length_ratio(a1, a2),
        "country_match": float(bool(c1) and c1 == c2),
        "name_exact": float(bool(n1) and n1 == n2),
    }


def create_features(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """Create model features for pairwise candidate rows."""
    if candidate_df.empty:
        return pd.DataFrame(index=candidate_df.index, columns=FEATURE_COLUMNS, dtype=float)
    missing = [c for c in ("source1_business_name", "candidate_business_name",
                           "source1_business_address", "candidate_business_address",
                           "source1_country", "candidate_country") if c not in candidate_df]
    if missing:
        raise ValueError(f"Candidate rows are missing feature fields: {missing}")
    return pd.DataFrame.from_records(
        (pair_features(row) for row in candidate_df.to_dict(orient="records")),
        index=candidate_df.index, columns=FEATURE_COLUMNS,
    ).astype("float32")
