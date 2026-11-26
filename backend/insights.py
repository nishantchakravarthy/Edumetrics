"""
insights.py
Generates textual insights, performs dropout prediction using Logistic Regression,
and exports everything as JSON for the frontend dashboard.

Features:  rating, num_reviews, watch_time, num_lectures, content_duration, level (encoded)
Target:    retention_level (binary: High=1 vs Low/Medium=0)
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH  = os.path.join(BASE_DIR, "data", "courses_cleaned.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows")
    return df


def generate_textual_insights(df):
    """Generate key analytical insights from the data."""
    insights = {}

    # Top category by engagement
    cat_eng = df.groupby("category")["engagement_score"].mean().sort_values(ascending=False)
    insights["top_engagement_category"] = {
        "category": cat_eng.index[0],
        "score": round(float(cat_eng.iloc[0]), 4),
    }
    insights["bottom_engagement_category"] = {
        "category": cat_eng.index[-1],
        "score": round(float(cat_eng.iloc[-1]), 4),
    }

    # Top category by completion
    cat_comp = df.groupby("category")["completion_rate"].mean().sort_values(ascending=False)
    insights["top_completion_category"] = {
        "category": cat_comp.index[0],
        "rate": round(float(cat_comp.iloc[0]), 2),
    }
    insights["lowest_completion_category"] = {
        "category": cat_comp.index[-1],
        "rate": round(float(cat_comp.iloc[-1]), 2),
    }

    # Level with highest dropout (lowest completion)
    level_comp = df.groupby("level")["completion_rate"].mean().sort_values()
    insights["highest_dropout_level"] = {
        "level": level_comp.index[0],
        "avg_completion": round(float(level_comp.iloc[0]), 2),
    }

    # Correlation insights
    corr = df.corr(numeric_only=True)
    comp_corr = corr["completion_rate"].drop(["completion_rate"]).sort_values(ascending=False)
    insights["strongest_completion_correlations"] = {
        col: round(float(val), 3) for col, val in comp_corr.head(3).items()
    }

    # Duration vs completion
    short = df[df["content_duration"] <= 10]["completion_rate"].mean()
    medium = df[(df["content_duration"] > 10) & (df["content_duration"] <= 30)]["completion_rate"].mean()
    long_c = df[df["content_duration"] > 30]["completion_rate"].mean()
    insights["duration_vs_completion"] = {
        "short_courses_under_10h": round(float(short), 2),
        "medium_courses_10_30h": round(float(medium), 2),
        "long_courses_over_30h": round(float(long_c), 2),
    }

    # Rating vs completion
    low_rated = df[df["rating"] < 3.5]["completion_rate"].mean()
    mid_rated = df[(df["rating"] >= 3.5) & (df["rating"] < 4.5)]["completion_rate"].mean()
    high_rated = df[df["rating"] >= 4.5]["completion_rate"].mean()
    insights["rating_vs_completion"] = {
        "low_rated_below_3_5": round(float(low_rated), 2),
        "mid_rated_3_5_to_4_5": round(float(mid_rated), 2),
        "high_rated_above_4_5": round(float(high_rated), 2),
    }

    # Retention distribution
    ret_counts = df["retention_level"].value_counts().to_dict()
    insights["retention_distribution"] = {k: int(v) for k, v in ret_counts.items()}

    # Overall stats
    insights["overall_stats"] = {
        "total_courses": int(len(df)),
        "total_students": int(df["num_subscribers"].sum()),
        "avg_engagement_score": round(float(df["engagement_score"].mean()), 4),
        "avg_completion_rate": round(float(df["completion_rate"].mean()), 2),
        "avg_watch_time": round(float(df["watch_time"].mean()), 2),
        "avg_rating": round(float(df["rating"].mean()), 2),
    }

    return insights


def dropout_prediction(df):
    """
    Logistic Regression: Predict retention (High vs Low/Medium).
    Features: rating, num_reviews, watch_time, num_lectures, content_duration, level
    Target:   retention_binary (1 = High, 0 = Low/Medium)
    """
    print("\n--- Dropout Prediction (Logistic Regression) ---")

    # Encode level
    le = LabelEncoder()
    df_ml = df.copy()
    df_ml["level_encoded"] = le.fit_transform(df_ml["level"])

    # Binary target
    df_ml["retention_binary"] = (df_ml["retention_level"] == "High").astype(int)

    # Features & target
    feature_cols = ["rating", "num_reviews", "watch_time", "num_lectures", "content_duration", "level_encoded"]
    X = df_ml[feature_cols]
    y = df_ml["retention_binary"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  Confusion Matrix: {cm}")
    print(classification_report(y_test, y_pred, target_names=["Low/Medium", "High"]))

    # Feature importance
    importances = dict(zip(feature_cols, [round(float(c), 4) for c in model.coef_[0]]))

    ml_results = {
        "model": "Logistic Regression",
        "features": feature_cols,
        "target": "retention_binary (High=1, Low/Medium=0)",
        "test_size": 0.25,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "confusion_matrix": cm,
        "feature_coefficients": importances,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
    }

    return ml_results


def generate_recommendations(insights):
    """Generate actionable recommendations based on insights."""
    recs = []

    # Duration recommendation
    dur = insights["duration_vs_completion"]
    if dur["short_courses_under_10h"] > dur["long_courses_over_30h"]:
        recs.append({
            "title": "Optimize Course Length",
            "description": f"Short courses (<10h) have {dur['short_courses_under_10h']}% completion vs {dur['long_courses_over_30h']}% for long courses (>30h). Consider breaking long courses into shorter modules.",
            "impact": "High",
        })

    # Rating recommendation
    rat = insights["rating_vs_completion"]
    recs.append({
        "title": "Improve Course Quality",
        "description": f"High-rated courses (>4.5) achieve {rat['high_rated_above_4_5']}% completion vs {rat['low_rated_below_3_5']}% for low-rated (<3.5). Investing in content quality directly improves retention.",
        "impact": "High",
    })

    # Level recommendation
    drop = insights["highest_dropout_level"]
    recs.append({
        "title": f"Address {drop['level']} Level Dropout",
        "description": f"{drop['level']} courses have the lowest avg completion at {drop['avg_completion']}%. Add prerequisites, mentorship, or scaffolded learning paths.",
        "impact": "Medium",
    })

    # Category recommendation
    top = insights["top_engagement_category"]
    bottom = insights["bottom_engagement_category"]
    recs.append({
        "title": "Learn from Top Performers",
        "description": f"{top['category']} leads engagement ({top['score']:.3f}) while {bottom['category']} trails ({bottom['score']:.3f}). Analyze what makes top categories succeed and replicate those patterns.",
        "impact": "Medium",
    })

    return recs


def run_insights():
    print("=" * 60)
    print("  E-Learning Insights & ML Prediction")
    print("=" * 60)

    df = load_data()
    insights = generate_textual_insights(df)
    ml_results = dropout_prediction(df)
    recommendations = generate_recommendations(insights)

    output = {
        "insights": insights,
        "ml_prediction": ml_results,
        "recommendations": recommendations,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "insights.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nInsights exported to: {output_path}")
    return output


if __name__ == "__main__":
    run_insights()
