"""
analysis.py
Generates static (Matplotlib/Seaborn) and interactive (Plotly) visualizations.
Produces 6 PNG charts and 4 interactive HTML charts.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
warnings.filterwarnings("ignore")

BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH  = os.path.join(BASE_DIR, "data", "courses_cleaned.csv")
CHARTS_DIR = os.path.join(BASE_DIR, "outputs", "charts")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

sns.set_theme(style="darkgrid", palette="viridis")
plt.rcParams.update({
    "figure.facecolor": "#0a0a1a",
    "axes.facecolor": "#1a1a2e",
    "axes.edgecolor": "#333355",
    "axes.labelcolor": "#e0e0e0",
    "text.color": "#e0e0e0",
    "xtick.color": "#b0b0c0",
    "ytick.color": "#b0b0c0",
    "grid.color": "#2a2a4a",
    "figure.figsize": (12, 7),
    "font.size": 12,
})

PLOTLY_TEMPLATE = "plotly_dark"


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows")
    return df


def static_charts(df):
    """Generate 6 static PNG charts."""
    print("\nGenerating static charts...")

    # 1 — Bar: Engagement by Category
    fig, ax = plt.subplots()
    order = df.groupby("category")["engagement_score"].mean().sort_values(ascending=False).index
    sns.barplot(x="category", y="engagement_score", data=df, order=order, palette="viridis", ax=ax, ci=None)
    ax.set_title("Average Engagement Score by Category", fontweight="bold")
    ax.set_xlabel("Category")
    ax.set_ylabel("Engagement Score")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "01_engagement_by_category.png"), dpi=150)
    plt.close(fig)
    print("  Chart 1 done")

    # 2 — Histogram: Rating Distribution
    fig, ax = plt.subplots()
    sns.histplot(df["rating"], bins=30, kde=True, color="#6c63ff", ax=ax, alpha=0.8)
    ax.axvline(df["rating"].mean(), color="#00d2ff", linestyle="--", linewidth=2, label=f'Mean: {df["rating"].mean():.2f}')
    ax.legend()
    ax.set_title("Distribution of Course Ratings", fontweight="bold")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "02_rating_distribution.png"), dpi=150)
    plt.close(fig)
    print("  Chart 2 done")

    # 3 — Heatmap: Correlation Matrix
    cols = ["rating", "num_subscribers", "num_reviews", "content_duration", "num_lectures", "completion_rate", "watch_time", "engagement_score"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, linewidths=0.5, square=True)
    ax.set_title("Feature Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "03_correlation_heatmap.png"), dpi=150)
    plt.close(fig)
    print("  Chart 3 done")

    # 4 — Scatter: Watch Time vs Completion Rate
    fig, ax = plt.subplots()
    cats = df["category"].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(cats)))
    for cat, c in zip(cats, colors):
        m = df["category"] == cat
        ax.scatter(df.loc[m, "watch_time"], df.loc[m, "completion_rate"], alpha=0.5, s=25, color=c, label=cat)
    ax.set_title("Watch Time vs Completion Rate", fontweight="bold")
    ax.set_xlabel("Watch Time (hours)")
    ax.set_ylabel("Completion Rate (%)")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "04_watchtime_vs_completion.png"), dpi=150)
    plt.close(fig)
    print("  Chart 4 done")

    # 5 — Box Plot: Rating by Category
    fig, ax = plt.subplots()
    order = df.groupby("category")["rating"].median().sort_values(ascending=False).index
    sns.boxplot(x="category", y="rating", data=df, order=order, palette="viridis", ax=ax)
    ax.set_title("Rating Distribution by Category", fontweight="bold")
    ax.set_xlabel("Category")
    ax.set_ylabel("Rating")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "05_rating_boxplot.png"), dpi=150)
    plt.close(fig)
    print("  Chart 5 done")

    # 6 — Line: Retention Trends by Year
    yearly = df.groupby("published_year").agg(avg_completion=("completion_rate", "mean"), count=("course_id", "count")).reset_index()
    fig, ax1 = plt.subplots()
    ax1.plot(yearly["published_year"], yearly["avg_completion"], marker="o", color="#6c63ff", linewidth=2.5, markersize=8, label="Avg Completion %")
    ax1.fill_between(yearly["published_year"], yearly["avg_completion"], alpha=0.15, color="#6c63ff")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Avg Completion Rate (%)", color="#6c63ff")
    ax2 = ax1.twinx()
    ax2.bar(yearly["published_year"], yearly["count"], alpha=0.25, color="#00d2ff", label="Course Count", width=0.6)
    ax2.set_ylabel("Number of Courses", color="#00d2ff")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("Retention Trends Over Time (2010-2025)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, "06_retention_trends.png"), dpi=150)
    plt.close(fig)
    print("  Chart 6 done")


def interactive_charts(df):
    """Generate 4 interactive Plotly HTML charts."""
    print("\nGenerating interactive charts...")
    layout_common = dict(font=dict(family="Inter, sans-serif"), paper_bgcolor="#0a0a1a", plot_bgcolor="#1a1a2e", margin=dict(t=60, b=40, l=40, r=40))

    # 1 — Bar
    agg = df.groupby("category").agg(avg_eng=("engagement_score", "mean"), avg_comp=("completion_rate", "mean"), count=("course_id", "count")).reset_index().sort_values("avg_eng", ascending=False)
    fig = px.bar(agg, x="category", y="avg_eng", color="avg_eng", color_continuous_scale="Viridis", hover_data=["avg_comp", "count"], template=PLOTLY_TEMPLATE, title="Average Engagement Score by Category")
    fig.update_layout(**layout_common)
    fig.write_html(os.path.join(OUTPUT_DIR, "plotly_engagement_bar.html"), include_plotlyjs="cdn")
    print("  Plotly 1 done")

    # 2 — Scatter
    fig = px.scatter(df, x="watch_time", y="completion_rate", color="category", size="num_subscribers", size_max=20, hover_data=["course_name", "rating", "level"], template=PLOTLY_TEMPLATE, title="Watch Time vs Completion Rate", opacity=0.7)
    fig.update_layout(**layout_common)
    fig.write_html(os.path.join(OUTPUT_DIR, "plotly_scatter.html"), include_plotlyjs="cdn")
    print("  Plotly 2 done")

    # 3 — Heatmap
    cols = ["rating", "num_subscribers", "num_reviews", "content_duration", "num_lectures", "completion_rate", "watch_time", "engagement_score"]
    corr = df[cols].corr().round(3)
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu_r", zmid=0, text=corr.values.round(2), texttemplate="%{text}", textfont=dict(size=11)))
    fig.update_layout(title="Feature Correlation Matrix", template=PLOTLY_TEMPLATE, width=700, height=650, **layout_common)
    fig.write_html(os.path.join(OUTPUT_DIR, "plotly_heatmap.html"), include_plotlyjs="cdn")
    print("  Plotly 3 done")

    # 4 — Sunburst
    agg2 = df.groupby(["category", "level", "retention_level"]).agg(count=("course_id", "count"), avg_eng=("engagement_score", "mean")).reset_index()
    fig = px.sunburst(agg2, path=["category", "level", "retention_level"], values="count", color="avg_eng", color_continuous_scale="Viridis", template=PLOTLY_TEMPLATE, title="Course Distribution: Category > Level > Retention")
    fig.update_layout(font=dict(family="Inter, sans-serif"), paper_bgcolor="#0a0a1a", margin=dict(t=60, b=20, l=20, r=20))
    fig.write_html(os.path.join(OUTPUT_DIR, "plotly_sunburst.html"), include_plotlyjs="cdn")
    print("  Plotly 4 done")


def run_analysis():
    print("=" * 60)
    print("  E-Learning Data Analysis & Visualization")
    print("=" * 60)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    df = load_data()

    print("\nStatistical Summary:")
    print(df.describe().round(2))
    corr = df.corr(numeric_only=True)
    print("\nTop correlations with completion_rate:")
    print(corr["completion_rate"].drop("completion_rate").sort_values(ascending=False).head(5))

    static_charts(df)
    interactive_charts(df)
    print("\nAll charts generated successfully!")


if __name__ == "__main__":
    run_analysis()
