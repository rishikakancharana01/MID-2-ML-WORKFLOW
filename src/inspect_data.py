"""
Inspection and Validation Script for Diabetes Healthcare Dataset.
==================================================================
Purpose:
    Inspects data/diabetes.csv, verifying dimensions, schema, data types,
    data hygiene (nulls and duplicates), descriptive statistics,
    and target class balance.
"""

import os
import sys

try:
    import pandas as pd
except ImportError:
    print("\n" + "=" * 70)
    print("[ERROR] The 'pandas' library is not yet installed in your Python environment.")
    print("To install the necessary project dependencies, please run:")
    print("    pip install -r requirements.txt")
    print("or:")
    print("    pip install pandas")
    print("=" * 70 + "\n")
    sys.exit(1)


def load_and_inspect_dataset(csv_path: str):
    """
    Load dataset from csv_path and print comprehensive summary diagnostics.
    """
    print("=" * 70)
    print("         HEALTHCARE DATASET INSPECTION & VALIDATION REPORT        ")
    print("=" * 70)

    # 1. Check file existence
    if not os.path.exists(csv_path):
        print(f"[ERROR] Dataset file not found at: {csv_path}")
        print("Please ensure data/diabetes.csv is generated or downloaded first.")
        sys.exit(1)

    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"[OK] Successfully loaded: {csv_path}\n")

    # 2. Dataset Dimensions
    n_rows, n_cols = df.shape
    print("-" * 70)
    print(f"1. DATASET DIMENSIONS: {n_rows} rows x {n_cols} columns")
    print("-" * 70)

    # 3. Column Names and Data Types
    print("\n" + "-" * 70)
    print("2. COLUMN NAMES & DATA TYPES:")
    print("-" * 70)
    col_info = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Non-Null Count": df.notnull().sum(),
        "Unique Values": df.nunique()
    }).reset_index(drop=True)
    print(col_info.to_string(index=False))

    # 4. First 5 Rows
    print("\n" + "-" * 70)
    print("3. FIRST 5 ROWS OF DATASET:")
    print("-" * 70)
    print(df.head().to_string())

    # 5. Missing Values Analysis
    print("\n" + "-" * 70)
    print("4. MISSING VALUES CHECK (NaN / Null):")
    print("-" * 70)
    missing = df.isnull().sum()
    missing_pct = (missing / n_rows) * 100
    missing_df = pd.DataFrame({
        "Missing Count": missing,
        "Percentage (%)": missing_pct.round(2)
    })
    print(missing_df.to_string())
    if missing.sum() == 0:
        print("\n-> Zero explicit missing (NaN) values detected.")
    else:
        print(f"\n-> Warning: {missing.sum()} missing values detected.")

    # 6. Duplicate Rows Check
    print("\n" + "-" * 70)
    print("5. DUPLICATE ROWS CHECK:")
    print("-" * 70)
    n_duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {n_duplicates}")
    if n_duplicates == 0:
        print("-> Clean: No duplicate records found.")
    else:
        print(f"-> Attention: {n_duplicates} duplicate records found.")

    # 7. Summary Descriptive Statistics
    print("\n" + "-" * 70)
    print("6. SUMMARY DESCRIPTIVE STATISTICS (Numeric Features):")
    print("-" * 70)
    stats = df.describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    print(stats.round(2).to_string())

    # 8. Target Class Distribution
    print("\n" + "-" * 70)
    print("7. TARGET CLASS ('Outcome') DISTRIBUTION:")
    print("-" * 70)
    if "Outcome" in df.columns:
        counts = df["Outcome"].value_counts().sort_index()
        proportions = (df["Outcome"].value_counts(normalize=True) * 100).sort_index()
        target_df = pd.DataFrame({
            "Class Label": [
                "0 (Negative / Non-Diabetic)",
                "1 (Positive / Diabetic)"
            ],
            "Count": counts.values,
            "Percentage (%)": proportions.round(2).values
        })
        print(target_df.to_string(index=False))
        
        ratio = counts.get(0, 1) / max(counts.get(1, 1), 1)
        print(f"\nImbalance Ratio (Class 0 : Class 1): ~{ratio:.2f} : 1")
    else:
        print("[WARNING] 'Outcome' column not found in dataset!")

    print("\n" + "=" * 70)
    print("                 INSPECTION COMPLETE                      ")
    print("=" * 70)


if __name__ == "__main__":
    # Path relative to script location
    default_path = os.path.join(os.path.dirname(__file__), "..", "data", "diabetes.csv")
    load_and_inspect_dataset(os.path.abspath(default_path))
