import pandas as pd
import re
import unicodedata


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(value):
    """
    Convert text into a normalized form.

    Example:
        "ABC Pvt. Ltd."
        ->
        "abc pvt ltd"
    """

    if pd.isna(value):
        return ""

    value = str(value).strip().lower()

    value = unicodedata.normalize(
        "NFKC",
        value
    )

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
# JACCARD TOKEN SIMILARITY
# =========================================================

def token_similarity(text1, text2):
    """
    Calculate similarity based on common words.

    Example:

        "ABC Health Center"
        "ABC Health Clinic"

    Common words:
        ABC
        Health

    Returns a value between 0 and 1.
    """

    text1 = normalize_text(text1)
    text2 = normalize_text(text2)

    if not text1 or not text2:
        return 0.0

    tokens1 = set(text1.split())
    tokens2 = set(text2.split())

    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    if not union:
        return 0.0

    return len(intersection) / len(union)


# =========================================================
# CHARACTER SIMILARITY
# =========================================================

def character_similarity(text1, text2):
    """
    Calculate simple character-based similarity.

    Uses SequenceMatcher.

    Returns a value between 0 and 1.
    """

    from difflib import SequenceMatcher

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
# LENGTH DIFFERENCE
# =========================================================

def length_difference(text1, text2):
    """
    Calculate normalized length difference.

    Returns a value between 0 and 1.

    0 = same length
    1 = very different length
    """

    text1 = normalize_text(text1)
    text2 = normalize_text(text2)

    len1 = len(text1)
    len2 = len(text2)

    if len1 == 0 and len2 == 0:
        return 0.0

    maximum = max(len1, len2)

    if maximum == 0:
        return 0.0

    return abs(len1 - len2) / maximum


# =========================================================
# COUNTRY MATCH
# =========================================================

def country_match(country1, country2):
    """
    Check whether both businesses belong to
    the same country.

    Returns:

        1 = same country
        0 = different country
    """

    country1 = normalize_text(country1)
    country2 = normalize_text(country2)

    if not country1 or not country2:
        return 0

    return int(country1 == country2)


# =========================================================
# SINGLE PAIR FEATURES
# =========================================================

def calculate_features(row):
    """
    Calculate all similarity features for one
    candidate pair.
    """

    name1 = row["source1_business_name"]
    name2 = row["candidate_business_name"]

    address1 = row["source1_business_address"]
    address2 = row["candidate_business_address"]

    country1 = row["source1_country"]
    country2 = row["candidate_country"]

    features = {

        # -------------------------------------------------
        # BUSINESS NAME
        # -------------------------------------------------

        "name_character_similarity":
            character_similarity(
                name1,
                name2
            ),

        "name_token_similarity":
            token_similarity(
                name1,
                name2
            ),

        "name_length_difference":
            length_difference(
                name1,
                name2
            ),

        # -------------------------------------------------
        # ADDRESS
        # -------------------------------------------------

        "address_character_similarity":
            character_similarity(
                address1,
                address2
            ),

        "address_token_similarity":
            token_similarity(
                address1,
                address2
            ),

        "address_length_difference":
            length_difference(
                address1,
                address2
            ),

        # -------------------------------------------------
        # COUNTRY
        # -------------------------------------------------

        "country_match":
            country_match(
                country1,
                country2
            )
    }

    return features


# =========================================================
# CREATE FEATURE DATAFRAME
# =========================================================

def create_features(candidate_df):
    """
    Convert candidate pairs into a feature dataframe.
    """

    if candidate_df.empty:

        return pd.DataFrame()

    feature_rows = []

    for _, row in candidate_df.iterrows():

        features = calculate_features(row)

        # Keep IDs for later training/evaluation
        features["source1_entity_id"] = (
            row["source1_entity_id"]
        )

        features["candidate_entity_id"] = (
            row["candidate_entity_id"]
        )

        features["candidate_source"] = (
            row["candidate_source"]
        )

        feature_rows.append(features)

    return pd.DataFrame(feature_rows)


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SIMILARITY FEATURE MODULE")
    print("=" * 60)

    print("\nTesting business name similarity...")

    name1 = "Juniper"
    name2 = "JUNIPER"

    similarity = character_similarity(
        name1,
        name2
    )

    print(
        f"Name similarity: {similarity:.4f}"
    )

    print("\nTesting token similarity...")

    similarity = token_similarity(
        "ABC Health Center",
        "ABC Health Clinic"
    )

    print(
        f"Token similarity: {similarity:.4f}"
    )

    print("\nTesting country match...")

    result = country_match(
        "US",
        "US"
    )

    print(
        f"Country match: {result}"
    )

    print("\nSimilarity module working correctly.")