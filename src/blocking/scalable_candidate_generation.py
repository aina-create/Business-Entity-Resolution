import pandas as pd
from pathlib import Path
import re
import time

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "train"

SOURCE1_FILE = DATA_DIR / "train_source1.tsv"
SOURCE2_FILE = DATA_DIR / "train_source2.tsv"
GROUND_TRUTH_FILE = DATA_DIR / "train_ground_truth.tsv"


def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).strip().lower()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def prepare_dataframe(df):

    df = df.copy()

    df["name_normalized"] = (
        df["business_name"]
        .apply(normalize_text)
    )

    df["address_normalized"] = (
        df["business_address"]
        .apply(normalize_text)
    )

    df["country_normalized"] = (
        df["country"]
        .apply(normalize_text)
    )

    words = df["name_normalized"].str.split()

    df["name_word1"] = (
        words.str[0]
        .fillna("")
    )

    df["name_key2"] = (
        words.str[:2]
        .str.join(" ")
    )

    df["name_key3"] = (
        words.str[:3]
        .str.join(" ")
    )

    df["address_key"] = (
        df["address_normalized"]
        .str[:25]
    )

    return df


def generate_candidates(source1, source2):

    candidates = []

    # -----------------------------
    # 1. Exact name
    # -----------------------------

    result = source1.merge(
        source2,
        left_on="name_normalized",
        right_on="name_normalized",
        suffixes=("_source1", "_source2")
    )

    if not result.empty:

        result["blocking_method"] = "exact_name"

        candidates.append(
            result[
                [
                    "entity_id_source1",
                    "entity_id_source2",
                    "blocking_method"
                ]
            ]
        )

    # -----------------------------
    # 2. Country + first word
    # -----------------------------

    result = source1.merge(
        source2,
        on=["country_normalized", "name_word1"],
        suffixes=("_source1", "_source2")
    )

    if not result.empty:

        result["blocking_method"] = (
            "country_first_word"
        )

        candidates.append(
            result[
                [
                    "entity_id_source1",
                    "entity_id_source2",
                    "blocking_method"
                ]
            ]
        )

    # -----------------------------
    # 3. Country + first 2 words
    # -----------------------------

    result = source1.merge(
        source2,
        on=["country_normalized", "name_key2"],
        suffixes=("_source1", "_source2")
    )

    if not result.empty:

        result["blocking_method"] = (
            "country_first_two_words"
        )

        candidates.append(
            result[
                [
                    "entity_id_source1",
                    "entity_id_source2",
                    "blocking_method"
                ]
            ]
        )

    # -----------------------------
    # 4. Country + first 3 words
    # -----------------------------

    result = source1.merge(
        source2,
        on=["country_normalized", "name_key3"],
        suffixes=("_source1", "_source2")
    )

    if not result.empty:

        result["blocking_method"] = (
            "country_first_three_words"
        )

        candidates.append(
            result[
                [
                    "entity_id_source1",
                    "entity_id_source2",
                    "blocking_method"
                ]
            ]
        )

    # -----------------------------
    # 5. Country + address
    # -----------------------------

    result = source1.merge(
        source2,
        on=["country_normalized", "address_key"],
        suffixes=("_source1", "_source2")
    )

    if not result.empty:

        result["blocking_method"] = (
            "country_address"
        )

        candidates.append(
            result[
                [
                    "entity_id_source1",
                    "entity_id_source2",
                    "blocking_method"
                ]
            ]
        )

    if not candidates:
        return pd.DataFrame(
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
                "blocking_method"
            ]
        )

    result = pd.concat(
        candidates,
        ignore_index=True
    )

    result = result.rename(
        columns={
            "entity_id_source1":
                "source1_entity_id",

            "entity_id_source2":
                "candidate_entity_id"
        }
    )

    result = result.drop_duplicates(
        subset=[
            "source1_entity_id",
            "candidate_entity_id"
        ]
    )

    return result


def load_ground_truth():

    print("\nLoading ground truth...")

    gt = pd.read_csv(
        GROUND_TRUTH_FILE,
        sep="\t"
    )

    match_dict = {}

    for row in gt.itertuples(index=False):

        if pd.isna(row.matched_entity_ids):
            match_dict[row.source1_entity_id] = set()

        else:

            match_dict[row.source1_entity_id] = {
                x.strip()
                for x in str(
                    row.matched_entity_ids
                ).split(",")
                if x.strip()
            }

    return match_dict


def evaluate_candidates(candidates, match_dict):

    if candidates.empty:

        print("\nNo candidates generated.")

        return

    candidates["actual_match"] = [
        candidate_id in match_dict.get(
            source1_id,
            set()
        )
        for source1_id, candidate_id
        in zip(
            candidates["source1_entity_id"],
            candidates["candidate_entity_id"]
        )
    ]

    total_matches = (
        candidates["actual_match"]
        .sum()
    )

    total_candidates = len(candidates)

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    print(
        f"Total candidates: "
        f"{total_candidates:,}"
    )

    print(
        f"Actual matches found: "
        f"{total_matches:,}"
    )

    print(
        f"Non-matches: "
        f"{total_candidates - total_matches:,}"
    )

    if total_matches > 0:

        print("\nRecovered matches:")

        print(
            candidates[
                candidates["actual_match"]
            ].head(20).to_string(
                index=False
            )
        )

    print("=" * 60)


def main():

    print("=" * 60)
    print("10,000 ROW CANDIDATE VALIDATION")
    print("=" * 60)

    start_time = time.time()

    # --------------------------------
    # Load only 10,000 Source 1 rows
    # --------------------------------

    print("\nLoading 10,000 Source 1 rows...")

    source1 = pd.read_csv(
        SOURCE1_FILE,
        sep="\t",
        nrows=10_000
    )

    print(
        f"Source 1 rows: "
        f"{len(source1):,}"
    )

    # --------------------------------
    # Load Source 2
    # --------------------------------

    print("\nLoading Source 2...")

    source2 = pd.read_csv(
        SOURCE2_FILE,
        sep="\t"
    )

    print(
        f"Source 2 rows: "
        f"{len(source2):,}"
    )

    # --------------------------------
    # Prepare data
    # --------------------------------

    print("\nPreparing Source 1...")

    source1 = prepare_dataframe(
        source1
    )

    print("Preparing Source 2...")

    source2 = prepare_dataframe(
        source2
    )

    # --------------------------------
    # Generate candidates
    # --------------------------------

    print("\nGenerating candidates...")

    candidates = generate_candidates(
        source1,
        source2
    )

    print(
        f"Candidates generated: "
        f"{len(candidates):,}"
    )

    # --------------------------------
    # Ground truth
    # --------------------------------

    match_dict = load_ground_truth()

    # --------------------------------
    # Evaluate
    # --------------------------------

    evaluate_candidates(
        candidates,
        match_dict
    )

    elapsed = time.time() - start_time

    print(
        f"\nTotal time: "
        f"{elapsed:.2f} seconds"
    )

    print("\nTest completed.")


if __name__ == "__main__":
    main()