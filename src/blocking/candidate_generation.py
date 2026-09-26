import pandas as pd
from pathlib import Path
import re
from difflib import SequenceMatcher


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "train"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_FILE = OUTPUT_DIR / "candidate_pairs.tsv"


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).strip().lower()

    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


# =========================================================
# FUZZY NAME SIMILARITY
# =========================================================

def fuzzy_name_similarity(text1, text2):

    text1 = normalize_text(text1)
    text2 = normalize_text(text2)

    if not text1 or not text2:
        return 0.0

    return SequenceMatcher(
        None,
        text1,
        text2
    ).ratio()


# =========================================================
# CREATE BLOCKING COLUMNS
# =========================================================

def create_blocking_columns(df):

    df = df.copy()

    # Normalize business name
    df["name_normalized"] = (
        df["business_name"]
        .apply(normalize_text)
    )

    # Normalize address
    df["address_normalized"] = (
        df["business_address"]
        .apply(normalize_text)
    )

    # Normalize country
    df["country_normalized"] = (
        df["country"]
        .apply(normalize_text)
    )

    # Split business name into words
    df["name_words"] = (
        df["name_normalized"]
        .str.split()
    )

    # First word
    df["name_word1"] = (
        df["name_words"]
        .str[0]
        .fillna("")
    )

    # First two words
    df["name_key2"] = (
        df["name_words"]
        .str[:2]
        .str.join(" ")
    )

    # First three words
    df["name_key3"] = (
        df["name_words"]
        .str[:3]
        .str.join(" ")
    )

    # Exact normalized name
    df["exact_name_key"] = (
        df["name_normalized"]
    )

    # First 25 characters of address
    df["address_key"] = (
        df["address_normalized"]
        .str[:25]
    )

    return df


# =========================================================
# CREATE CANDIDATES FOR TWO SOURCES
# =========================================================

def generate_candidates(source1, source_other):

    source1 = create_blocking_columns(source1)

    source_other = create_blocking_columns(
        source_other
    )

    candidates = []

    # =====================================================
    # BLOCK 1
    # Exact normalized business name
    # =====================================================

    left = source1[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
            "name_normalized",
            "address_normalized",
            "country_normalized",
            "exact_name_key"
        ]
    ]

    right = source_other[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
            "name_normalized",
            "address_normalized",
            "country_normalized",
            "exact_name_key"
        ]
    ]

    merged = left.merge(
        right,
        on="exact_name_key",
        suffixes=(
            "_source1",
            "_candidate"
        )
    )

    if len(merged) > 0:

        merged["blocking_method"] = (
            "exact_name"
        )

        candidates.append(merged)

    # =====================================================
    # BLOCK 2
    # Country + first word
    # =====================================================

    left = source1[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
            "name_normalized",
            "address_normalized",
            "country_normalized",
            "name_word1"
        ]
    ]

    right = source_other[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
            "name_normalized",
            "address_normalized",
            "country_normalized",
            "name_word1"
        ]
    ]

    merged = left.merge(
        right,
        on=[
            "country_normalized",
            "name_word1"
        ],
        suffixes=(
            "_source1",
            "_candidate"
        )
    )

    if len(merged) > 0:

        merged["blocking_method"] = (
            "country_first_word"
        )

        candidates.append(merged)

    # =====================================================
    # BLOCK 3
    # Country + first two words
    # =====================================================

    left = source1[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
            "name_normalized",
            "address_normalized",
            "country_normalized",
            "name_key2"
        ]
    ]

    right = source_other[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
            "name_normalized",
            "address_normalized",
            "country_normalized",
            "name_key2"
        ]
    ]

    merged = left.merge(
        right,
        on=[
            "country_normalized",
            "name_key2"
        ],
        suffixes=(
            "_source1",
            "_candidate"
        )
    )

    if len(merged) > 0:

        merged["blocking_method"] = (
            "country_first_two_words"
        )

        candidates.append(merged)

    # =====================================================
    # BLOCK 4
    # Country + first three words
    # =====================================================

    left = source1[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
            "name_normalized",
            "address_normalized",
            "country_normalized",
            "name_key3"
        ]
    ]

    right = source_other[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
            "name_normalized",
            "address_normalized",
            "country_normalized",
            "name_key3"
        ]
    ]

    merged = left.merge(
        right,
        on=[
            "country_normalized",
            "name_key3"
        ],
        suffixes=(
            "_source1",
            "_candidate"
        )
    )

    if len(merged) > 0:

        merged["blocking_method"] = (
            "country_first_three_words"
        )

        candidates.append(merged)

    # =====================================================
    # BLOCK 5
    # Country + address beginning
    # =====================================================

    left = source1[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
            "name_normalized",
            "address_normalized",
            "country_normalized",
            "address_key"
        ]
    ]

    right = source_other[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
            "name_normalized",
            "address_normalized",
            "country_normalized",
            "address_key"
        ]
    ]

    merged = left.merge(
        right,
        on=[
            "country_normalized",
            "address_key"
        ],
        suffixes=(
            "_source1",
            "_candidate"
        )
    )

    if len(merged) > 0:

        merged["blocking_method"] = (
            "country_address"
        )

        candidates.append(merged)

    # =====================================================
    # NO CANDIDATES
    # =====================================================

    if not candidates:

        return pd.DataFrame()

    # =====================================================
    # COMBINE ALL BLOCKING RESULTS
    # =====================================================

    result = pd.concat(
        candidates,
        ignore_index=True
    )

    # =====================================================
    # RENAME IMPORTANT COLUMNS
    # =====================================================

    result = result.rename(
        columns={
            "entity_id_source1":
                "source1_entity_id",

            "entity_id_candidate":
                "candidate_entity_id",

            "business_name_source1":
                "source1_business_name",

            "business_name_candidate":
                "candidate_business_name",

            "business_address_source1":
                "source1_business_address",

            "business_address_candidate":
                "candidate_business_address",

            "country_source1":
                "source1_country",

            "country_candidate":
                "candidate_country"
        }
    )

    # =====================================================
    # ADD CANDIDATE SOURCE
    # =====================================================

    result["candidate_source"] = (
        result["candidate_entity_id"]
        .str.split("-")
        .str[0]
    )

    # =====================================================
    # REMOVE DUPLICATES
    # =====================================================

    result = result.drop_duplicates(
        subset=[
            "source1_entity_id",
            "candidate_entity_id"
        ]
    )

    return result.reset_index(
        drop=True
    )


# =========================================================
# TEST
# =========================================================

def main():

    print("=" * 60)
    print("BUSINESS ENTITY RESOLUTION")
    print("CANDIDATE GENERATION TEST")
    print("=" * 60)

    # -----------------------------------------------------
    # Load Source 1 sample
    # -----------------------------------------------------

    print("\nLoading Source 1 sample...")

    source1 = pd.read_csv(
        DATA_DIR / "train_source1.tsv",
        sep="\t",
        nrows=10000
    )

    print(
        f"Source 1 rows: {len(source1):,}"
    )

    # -----------------------------------------------------
    # Load Source 2 sample
    # -----------------------------------------------------

    print("\nLoading Source 2 sample...")

    source2 = pd.read_csv(
        DATA_DIR / "train_source2.tsv",
        sep="\t",
        nrows=10000
    )

    print(
        f"Source 2 rows: {len(source2):,}"
    )

    # -----------------------------------------------------
    # Generate S1 -> S2 candidates
    # -----------------------------------------------------

    print(
        "\nGenerating Source 1 -> Source 2 candidates..."
    )

    candidates_12 = generate_candidates(
        source1,
        source2
    )

    print(
        f"Candidates generated: "
        f"{len(candidates_12):,}"
    )

    # -----------------------------------------------------
    # Load Source 3 sample
    # -----------------------------------------------------

    print("\nLoading Source 3 sample...")

    source3 = pd.read_csv(
        DATA_DIR / "train_source3.tsv",
        sep="\t",
        nrows=10000
    )

    print(
        f"Source 3 rows: {len(source3):,}"
    )

    # -----------------------------------------------------
    # Generate S1 -> S3 candidates
    # -----------------------------------------------------

    print(
        "\nGenerating Source 1 -> Source 3 candidates..."
    )

    candidates_13 = generate_candidates(
        source1,
        source3
    )

    print(
        f"Candidates generated: "
        f"{len(candidates_13):,}"
    )

    # -----------------------------------------------------
    # Combine
    # -----------------------------------------------------

    all_candidates = pd.concat(
        [
            candidates_12,
            candidates_13
        ],
        ignore_index=True
    )

    all_candidates = all_candidates.drop_duplicates(
        subset=[
            "source1_entity_id",
            "candidate_entity_id"
        ]
    )

    print(
        f"\nTotal candidate pairs: "
        f"{len(all_candidates):,}"
    )

    # -----------------------------------------------------
    # Show sample
    # -----------------------------------------------------

    if len(all_candidates) > 0:

        print("\nSample candidates:\n")

        print(
            all_candidates[
                [
                    "source1_entity_id",
                    "source1_business_name",
                    "candidate_entity_id",
                    "candidate_business_name",
                    "blocking_method"
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    print("\nTest completed successfully.")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    main()