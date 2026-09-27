"""Low-cost, Unicode-safe cleaning for business record tables."""

from __future__ import annotations

import re
import pandas as pd

REQUIRED_COLUMNS = ("entity_id", "business_name", "business_address", "country")


def clean_text(value) -> str:
    """Convert nulls to empty text and collapse whitespace, preserving scripts."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def clean_dataframe(df: pd.DataFrame, *, validate: bool = True) -> pd.DataFrame:
    """Return a cleaned copy; does not discard duplicate records or IDs."""
    if validate:
        missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"Missing required business columns: {missing}")
    result = df.copy()
    for column in ("entity_id", "business_name", "business_address", "country"):
        if column in result.columns:
            result[column] = result[column].map(clean_text)
    return result


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows while keeping the first occurrence."""
    return df.drop_duplicates().reset_index(drop=True)


def clean_business_data(df: pd.DataFrame, *, drop_duplicates: bool = False) -> pd.DataFrame:
    """Clean input without unexpectedly removing distinct entity IDs."""
    result = clean_dataframe(df)
    return remove_duplicate_rows(result) if drop_duplicates else result
