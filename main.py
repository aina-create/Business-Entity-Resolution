from pathlib import Path
import pandas as pd


# Project folder
BASE_DIR = Path(__file__).resolve().parent

# Dataset folder
DATA_DIR = BASE_DIR / "data" / "train"


def inspect_dataset(file_path):
    print("\n" + "=" * 60)
    print(f"FILE: {file_path.name}")
    print("=" * 60)

    try:
        # Read only 5 rows
        df = pd.read_csv(
            file_path,
            sep="\t",
            nrows=5
        )

        print("\nColumns:")

        for column in df.columns:
            print(f"  - {column}")

        print("\nSample data:")
        print(df.to_string(index=False))

    except Exception as error:
        print(f"\nERROR reading {file_path.name}:")
        print(error)


def main():

    print("\nBUSINESS ENTITY RESOLUTION")
    print("=" * 60)
    print("Dataset inspection started...")

    files = [
        "train_source1.tsv",
        "train_source2.tsv",
        "train_source3.tsv",
        "train_ground_truth.tsv"
    ]

    for file_name in files:

        file_path = DATA_DIR / file_name

        if file_path.exists():
            inspect_dataset(file_path)
        else:
            print(f"\nWARNING: {file_name} not found.")

    print("\n" + "=" * 60)
    print("Dataset inspection completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()