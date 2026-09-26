import pandas as pd
import re


def clean_text(value):
    """
    Clean text while preserving Unicode characters.

    This is important because the dataset contains
    Hindi, Kannada and other non-English business names.
    """

    if pd.isna(value):
        return ""

    value = str(value)

    # Remove leading/trailing spaces
    value = value.strip()

    # Replace multiple whitespace characters with one space
    value = re.sub(r"\s+", " ", value)

    return value


def clean_dataframe(df):
    """
    Clean business dataset.
    """

    df = df.copy()

    for column in df.columns:

        if df[column].dtype == "object":

            df[column] = df[column].apply(
                clean_text
            )

    return df


def remove_duplicate_rows(df):
    """
    Remove completely duplicated rows.
    """

    return (
        df
        .drop_duplicates()
        .reset_index(drop=True)
    )


def clean_business_data(df):
    """
    Complete cleaning pipeline.
    """

    df = clean_dataframe(df)

    df = remove_duplicate_rows(df)

    return df


if __name__ == "__main__":

    print("Cleaning module loaded successfully.")