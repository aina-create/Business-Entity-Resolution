import pandas as pd
from pathlib import Path


from src.blocking.candidate_generation import (
    generate_candidates
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "train"


# =========================================================
# LOAD GROUND TRUTH
# =========================================================

def load_ground_truth():

    ground_truth_path = (
        DATA_DIR / "train_ground_truth.tsv"
    )

    return pd.read_csv(
        ground_truth_path,
        sep="\t"
    )


# =========================================================
# CREATE MATCH DICTIONARY
# =========================================================

def create_match_set(ground_truth):

    match_dict = {}

    for source1_id, matched_ids in zip(
        ground_truth["source1_entity_id"],
        ground_truth["matched_entity_ids"]
    ):

        if pd.isna(matched_ids):

            match_dict[source1_id] = set()

            continue

        match_dict[source1_id] = {
            entity_id.strip()
            for entity_id in str(
                matched_ids
            ).split(",")
            if entity_id.strip()
        }

    return match_dict


# =========================================================
# LABEL CANDIDATES
# =========================================================

def label_candidates(
    candidates,
    match_dict
):

    candidates = candidates.copy()

    def get_label(row):

        source1_id = row[
            "source1_entity_id"
        ]

        candidate_id = row[
            "candidate_entity_id"
        ]

        actual_matches = match_dict.get(
            source1_id,
            set()
        )

        return int(
            candidate_id in actual_matches
        )

    candidates["match"] = (
        candidates.apply(
            get_label,
            axis=1
        )
    )

    return candidates


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CANDIDATE LABELING TEST")
    print("=" * 60)

    # -----------------------------------------------------
    # Load small samples
    # -----------------------------------------------------

    print("\nLoading Source 1...")

    source1 = pd.read_csv(
        DATA_DIR / "train_source1.tsv",
        sep="\t",
        nrows=1000
    )

    print(
        f"Source 1 rows: {len(source1):,}"
    )

    print("\nLoading Source 2...")

    source2 = pd.read_csv(
        DATA_DIR / "train_source2.tsv",
        sep="\t",
        nrows=1000
    )

    print(
        f"Source 2 rows: {len(source2):,}"
    )

    # -----------------------------------------------------
    # Generate candidates
    # -----------------------------------------------------

    print(
        "\nGenerating candidate pairs..."
    )

    candidates = generate_candidates(
        source1,
        source2
    )

    print(
        f"Candidate pairs: {len(candidates):,}"
    )

    # -----------------------------------------------------
    # Load ground truth
    # -----------------------------------------------------

    print(
        "\nLoading ground truth..."
    )

    ground_truth = load_ground_truth()

    print(
        f"Ground truth rows: "
        f"{len(ground_truth):,}"
    )

    # -----------------------------------------------------
    # Create match dictionary
    # -----------------------------------------------------

    print(
        "\nCreating match dictionary..."
    )

    match_dict = create_match_set(
        ground_truth
    )

    # -----------------------------------------------------
    # Label candidates
    # -----------------------------------------------------

    print(
        "\nLabeling candidates..."
    )

    labeled_candidates = label_candidates(
        candidates,
        match_dict
    )

    # -----------------------------------------------------
    # Count labels
    # -----------------------------------------------------

    matches = (
        labeled_candidates["match"] == 1
    ).sum()

    non_matches = (
        labeled_candidates["match"] == 0
    ).sum()

    print("\nRESULT")
    print("-" * 40)

    print(
        f"Total candidates : "
        f"{len(labeled_candidates):,}"
    )

    print(
        f"Actual matches   : "
        f"{matches:,}"
    )

    print(
        f"Non-matches      : "
        f"{non_matches:,}"
    )

    # -----------------------------------------------------
    # Display actual matches
    # -----------------------------------------------------

    if matches > 0:

        print(
            "\nSample actual matches:"
        )

        print(
            labeled_candidates[
                labeled_candidates["match"] == 1
            ][
                [
                    "source1_entity_id",
                    "source1_business_name",
                    "candidate_entity_id",
                    "candidate_business_name",
                    "match"
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    print(
        "\nCandidate labeling test completed."
    )

    print("=" * 60)