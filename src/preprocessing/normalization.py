import pandas as pd
import re
import unicodedata


def normalize_text(value):
    """
    Normalize text while preserving Unicode characters.
    """

    if pd.isna(value):
        return ""

    value = str(value).strip().lower()

    # Unicode normalization
    value = unicodedata.normalize(
        "NFKC",
        value
    )

    # Replace punctuation with spaces
    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE
    )

    # Remove extra spaces
    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


def normalize_business_name(value):
    """
    Normalize a business name.
    """

    return normalize_text(value)


def normalize_address(value):
    """
    Normalize business address.
    """

    return normalize_text(value)


def normalize_country(value):
    """
    Normalize country.
    """

    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def normalize_dataframe(df):
    """
    Normalize the business dataset.
    """

    df = df.copy()

    if "business_name" in df.columns:

        df["business_name_normalized"] = (
            df["business_name"]
            .apply(normalize_business_name)
        )

    if "business_address" in df.columns:

        df["business_address_normalized"] = (
            df["business_address"]
            .apply(normalize_address)
        )

    if "country" in df.columns:

        df["country_normalized"] = (
            df["country"]
            .apply(normalize_country)
        )

    return df


if __name__ == "__main__":

    print(
        "Normalization module loaded successfully."
    )