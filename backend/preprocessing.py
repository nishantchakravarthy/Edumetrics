"""
preprocessing.py
Cleans the raw dataset and engineers features for analysis.
All numeric columns are normalized before computing engagement scores.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
RAW_PATH    = os.path.join(BASE_DIR, "data", "courses.csv")
CLEAN_PATH  = os.path.join(BASE_DIR, "data", "courses_cleaned.csv")


def load_data() -> pd.DataFrame:
    """Load the raw CSV dataset."""
    df = pd.read_csv(RAW_PATH)
    print(f"📂 Loaded {len(df)} rows from {RAW_PATH}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"🔄 Removed {before - after} duplicate rows ({after} remaining)")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values — drop rows with NaN."""
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"🧹 Dropped {before - after} rows with missing values ({after} remaining)")
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize key numeric columns using MinMaxScaler (0–1 range).
    Creates new 'norm_*' columns while preserving originals.
    """
    scaler = MinMaxScaler()
    cols_to_normalize = [
        "rating",
        "num_subscribers",
        "num_reviews",
        "content_duration",
        "num_lectures",
        "completion_rate",
        "watch_time",
        "price",
    ]

    for col in cols_to_normalize:
        norm_col = f"norm_{col}"
        df[norm_col] = scaler.fit_transform(df[[col]])

    print(f"📊 Normalized {len(cols_to_normalize)} columns (norm_* prefix)")
    return df


def compute_engagement_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute engagement score from NORMALIZED values only.
    Formula: 0.4 * norm_rating + 0.3 * norm_reviews + 0.3 * norm_completion
    """
    df["engagement_score"] = (
        df["norm_rating"]          * 0.4 +
        df["norm_num_reviews"]     * 0.3 +
        df["norm_completion_rate"] * 0.3
    )
    df["engagement_score"] = df["engagement_score"].round(4)
    print(f"⚡ Engagement score computed (mean: {df['engagement_score'].mean():.4f})")
    return df


def categorize_retention(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize retention level based on raw completion_rate.
    High ≥ 80  |  Medium ≥ 50  |  Low < 50
    """
    def retention_label(x: float) -> str:
        if x >= 80:
            return "High"
        elif x >= 50:
            return "Medium"
        else:
            return "Low"

    df["retention_level"] = df["completion_rate"].apply(retention_label)

    counts = df["retention_level"].value_counts()
    print(f"🏷️  Retention levels: {dict(counts)}")
    return df


def preprocess():
    """Run the full preprocessing pipeline."""
    print("=" * 60)
    print("  E-Learning Data Preprocessing Pipeline")
    print("=" * 60)

    df = load_data()
    df = remove_duplicates(df)
    df = handle_missing(df)
    df = normalize_columns(df)
    df = compute_engagement_score(df)
    df = categorize_retention(df)

    # Save cleaned data
    os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)

    print(f"\n✅ Cleaned dataset saved to: {CLEAN_PATH}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n📈 Summary Statistics:")
    print(df[["rating", "num_subscribers", "completion_rate",
              "watch_time", "engagement_score"]].describe().round(2))

    return df


if __name__ == "__main__":
    preprocess()
