"""
data_loader.py
Loads the UCI Heart Disease dataset and performs initial cleaning.
"""

from pathlib import Path

import pandas as pd


def load_data(path=None):
    """Load heart disease dataset from CSV."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "data" / "raw" / "heart.csv"
    df = pd.read_csv(path)
    return df


def clean_data(df):
    """Remove duplicate rows and reset index."""
    original_size = len(df)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    removed = original_size - len(df_clean)
    print(f"Removed {removed} duplicate rows ({original_size} -> {len(df_clean)})")
    return df_clean


if __name__ == "__main__":
    df = load_data()
    print("Dataset loaded successfully.")
    print("Shape (raw):", df.shape)
    print("Duplicates found:", df.duplicated().sum())

    df_clean = clean_data(df)
    print("Shape (clean):", df_clean.shape)
    print(df_clean.head())
