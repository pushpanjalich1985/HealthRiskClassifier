# ─────────────────────────────────────────────
#  preprocess.py
#  Loads health_data.csv, checks data quality,
#  encodes labels, and splits into train/test
# ─────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



def load_data(path="data/health_data.csv"):
    """
    Load the CSV file into a Pandas DataFrame.
    Also runs basic quality checks and prints a summary.
    """

    # ── Step 1: Load the CSV ─────────────────────────────────────────────────
    df = pd.read_csv(path)

    print("=" * 50)
    print("DATASET LOADED")
    print("=" * 50)

    # Shape tells us (rows, columns)
    print(f"Shape        : {df.shape[0]} rows × {df.shape[1]} columns")

    # dtypes shows us the data type of every column
    # int64 = integer, float64 = decimal, object = text/string
    print(f"\nColumn types :\n{df.dtypes}")

    # ── Step 2: Check for missing values ─────────────────────────────────────
    # isnull() returns True/False for every cell
    # .sum() counts the Trues (missing values) per column
    missing = df.isnull().sum()

    print(f"\nMissing values per column:\n{missing}")

    # If any column has missing values, warn the user
    if missing.sum() > 0:
        print("\nWARNING: Missing values found! Consider handling them.")
    else:
        print("\nNo missing values found.")

    # ── Step 3: Check for duplicates ─────────────────────────────────────────
    # duplicated() marks rows that are exact copies of a previous row
    n_duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {n_duplicates}")

    if n_duplicates > 0:
        # drop_duplicates() removes all duplicate rows
        # keep="first" means keep the first occurrence, remove the rest
        df = df.drop_duplicates(keep="first")
        print(f"Duplicates removed. New shape: {df.shape}")

    # ── Step 4: Basic statistics ─────────────────────────────────────────────
    # describe() gives count, mean, std, min, 25%, 50%, 75%, max
    # for every numeric column — very useful for spotting weird values
    print(f"\nBasic Statistics:\n{df.describe().round(2)}")

    # ── Step 5: Label distribution ───────────────────────────────────────────
    # value_counts() counts how many rows have each unique value
    # normalize=True gives percentages instead of raw counts
    print(f"\nRisk Label Distribution:")
    counts = df["RiskLabel"].value_counts()
    percentages = df["RiskLabel"].value_counts(normalize=True) * 100
    for label in counts.index:
        print(f"  {label:8s}: {counts[label]:4d} rows ({percentages[label]:.1f}%)")

    return df


def encode_labels(df):
    """
    Convert the text RiskLabel column into numbers.

    Low    → 0
    Medium → 1
    High   → 2
    """

    # Manual mapping — we control the order, not the alphabet
    risk_map = {"Low": 0, "Medium": 1, "High": 2}
    df["RiskLabel"] = df["RiskLabel"].map(risk_map)

    print("\n" + "=" * 50)
    print("LABEL ENCODING")
    print("=" * 50)
    print("Mapping:")
    for label, number in risk_map.items():
        print(f"  {label:8s} → {number}")

    # We return risk_map 
    # so we can decode predictions back to text later
    return df, risk_map


def split_data(df):
    """
    Separate features from target, then split into train and test sets.

    Features (X) = the 8 input columns the model learns from
    Target   (y) = the RiskLabel column we want to predict

    Returns:
        X_train, X_test   : Feature sets
        y_train, y_test   : Label sets
    """

    # ── Step 1: Separate X and y ─────────────────────────────────────────────
    # X = everything EXCEPT the label (these are our inputs)
    # We drop "RiskLabel" because that's what we're trying to predict
    X = df.drop(columns=["RiskLabel"])

    # y = ONLY the label column (this is what we're predicting)
    y = df["RiskLabel"]

    print("\n" + "=" * 50)
    print("TRAIN / TEST SPLIT")
    print("=" * 50)
    print(f"Features (X) shape : {X.shape}")
    print(f"Target   (y) shape : {y.shape}")
    print(f"Feature columns    : {list(X.columns)}")

    # ── Step 2: Split ────────────────────────────────────────────────────────
    # test_size=0.2  → 20% goes to test, 80% goes to training
    # random_state=42 → seed for reproducibility (same split every run)
    # stratify=y  → ensure class proportions are preserved in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining set : {X_train.shape[0]} rows (80%)")
    print(f"Test set     : {X_test.shape[0]} rows (20%)")

    # Verify stratification worked — proportions should match original
    print(f"\nClass distribution in training set:")
    train_dist = y_train.value_counts(normalize=True) * 100
    test_dist  = y_test.value_counts(normalize=True) * 100

    for label in sorted(y_train.unique()):
        print(f"  Class {label}: Train={train_dist[label]:.1f}%  "
              f"Test={test_dist[label]:.1f}%")

    return X_train, X_test, y_train, y_test


# ── Run directly for testing ──────────────────────────────────────────────────
if __name__ == "__main__":
    df            = load_data()
    df, risk_map   = encode_labels(df)
    X_train, X_test, y_train, y_test = split_data(df)

    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETE")
    print(f"X_train : {X_train.shape}")
    print(f"X_test  : {X_test.shape}")
    print(f"y_train : {y_train.shape}")
    print(f"y_test  : {y_test.shape}")
    print("=" * 50)