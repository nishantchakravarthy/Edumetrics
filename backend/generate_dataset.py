"""
generate_dataset.py
Generates a realistic synthetic e-learning dataset (2,500 courses)
modeled after Udemy/Coursera platform distributions.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

# ── Configuration ──────────────────────────────────────────────
NUM_COURSES = 2500
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

CATEGORIES = [
    "Web Development",
    "Data Science",
    "Business",
    "Design",
    "Marketing",
    "IT & Software",
    "Photography",
    "Music",
]

LEVELS = ["Beginner", "Intermediate", "Advanced", "All Levels"]

# Probability weights for realistic distributions
CATEGORY_WEIGHTS = [0.22, 0.20, 0.15, 0.12, 0.10, 0.10, 0.06, 0.05]
LEVEL_WEIGHTS    = [0.30, 0.30, 0.15, 0.25]

# Course name templates per category
COURSE_TEMPLATES = {
    "Web Development": [
        "Complete {adj} Web Developer Bootcamp",
        "{adj} HTML, CSS & JavaScript Masterclass",
        "React & Node.js — {adj} Full-Stack Course",
        "Modern {adj} Frontend Development",
        "Build {adj} Responsive Websites from Scratch",
        "Vue.js {adj} Framework Deep Dive",
        "Angular {adj} Complete Guide",
        "Python Django {adj} Web Applications",
        "TypeScript {adj} for Professionals",
        "WordPress {adj} Theme Development",
    ],
    "Data Science": [
        "{adj} Python for Data Science & ML",
        "Machine Learning A–Z: {adj} Guide",
        "Data Analysis with {adj} Pandas",
        "Deep Learning {adj} with TensorFlow",
        "Statistics & Probability {adj} Bootcamp",
        "SQL for {adj} Data Analytics",
        "Natural Language Processing {adj}",
        "Data Visualization {adj} Masterclass",
        "R Programming {adj} for Analytics",
        "Big Data {adj} with Apache Spark",
    ],
    "Business": [
        "{adj} Financial Analysis Bootcamp",
        "MBA in a Box: {adj} Business Strategy",
        "Project Management {adj} Professional",
        "Agile & Scrum {adj} Masterclass",
        "{adj} Business Analytics with Excel",
        "Entrepreneurship {adj} Fundamentals",
        "Leadership & Management {adj}",
        "Digital Transformation {adj} Strategy",
        "Supply Chain {adj} Management",
        "Business Communication {adj} Skills",
    ],
    "Design": [
        "{adj} UI/UX Design Bootcamp",
        "Adobe Photoshop {adj} Masterclass",
        "Graphic Design {adj} for Beginners",
        "Figma {adj} Complete Guide",
        "Motion Graphics {adj} with After Effects",
        "Logo Design {adj} from Scratch",
        "Illustration {adj} Fundamentals",
        "3D Modeling {adj} with Blender",
        "Color Theory {adj} for Designers",
        "Web Design {adj} Principles",
    ],
    "Marketing": [
        "{adj} Digital Marketing Masterclass",
        "SEO {adj} Complete Guide",
        "Social Media Marketing {adj}",
        "Google Ads {adj} Certification Prep",
        "Content Marketing {adj} Strategy",
        "Email Marketing {adj} Automation",
        "Brand Building {adj} Fundamentals",
        "Copywriting {adj} for Conversions",
        "Influencer Marketing {adj} Guide",
        "Marketing Analytics {adj} with Python",
    ],
    "IT & Software": [
        "{adj} Cybersecurity Bootcamp",
        "AWS Solutions Architect {adj}",
        "Linux Administration {adj}",
        "Docker & Kubernetes {adj} Guide",
        "Ethical Hacking {adj} Complete Course",
        "CompTIA A+ {adj} Certification",
        "Networking {adj} Fundamentals",
        "Cloud Computing {adj} Essentials",
        "DevOps {adj} Engineering",
        "System Administration {adj} Pro",
    ],
    "Photography": [
        "{adj} Photography Masterclass",
        "Lightroom {adj} Complete Guide",
        "Portrait Photography {adj}",
        "Landscape Photography {adj} Secrets",
        "iPhone Photography {adj}",
        "Night Photography {adj} Techniques",
        "Food Photography {adj} Pro",
        "Street Photography {adj}",
        "Photo Editing {adj} Workflow",
        "Drone Photography {adj} Guide",
    ],
    "Music": [
        "{adj} Music Production in Ableton",
        "Guitar {adj} for Beginners",
        "Piano {adj} Complete Course",
        "Music Theory {adj} Fundamentals",
        "Mixing & Mastering {adj}",
        "Songwriting {adj} Bootcamp",
        "FL Studio {adj} Production",
        "Vocal Training {adj}",
        "DJ {adj} Mixing Techniques",
        "Sound Design {adj} Essentials",
    ],
}

ADJECTIVES = [
    "Complete", "Ultimate", "Essential", "Advanced", "Professional",
    "Comprehensive", "Practical", "Modern", "Expert", "Intensive",
    "Certified", "Premium", "Pro-Level", "Hands-On", "Mastery",
]


def generate_course_name(category: str, idx: int) -> str:
    """Generate a realistic course title."""
    templates = COURSE_TEMPLATES[category]
    template = templates[idx % len(templates)]
    adj = ADJECTIVES[idx % len(ADJECTIVES)]
    # Add edition number for uniqueness
    edition = (idx // len(templates)) + 1
    name = template.format(adj=adj)
    if edition > 1:
        name += f" (Edition {edition})"
    return name


def generate_dataset() -> pd.DataFrame:
    """Generate the full synthetic dataset."""
    data = []

    for i in range(NUM_COURSES):
        # ── Category & Level ──
        category = np.random.choice(CATEGORIES, p=CATEGORY_WEIGHTS)
        level = np.random.choice(LEVELS, p=LEVEL_WEIGHTS)

        # ── Course name ──
        cat_index = sum(1 for d in data if d["category"] == category)
        course_name = generate_course_name(category, cat_index)

        # ── Published year (2010–2025, skewed toward recent) ──
        published_year = int(np.random.triangular(2010, 2021, 2025))

        # ── Content duration (hours): 0.5–80, skewed by level ──
        level_duration_map = {
            "Beginner": (1, 15),
            "Intermediate": (5, 40),
            "Advanced": (10, 80),
            "All Levels": (2, 50),
        }
        dur_low, dur_high = level_duration_map[level]
        content_duration = round(np.random.uniform(dur_low, dur_high), 1)

        # ── Num lectures: correlated with duration ──
        lectures_per_hour = np.random.uniform(3, 8)
        num_lectures = max(5, int(content_duration * lectures_per_hour))

        # ── Price ──
        price = float(np.random.choice(
            [0, 19.99, 24.99, 29.99, 49.99, 84.99, 99.99, 129.99, 199.99],
            p=[0.15, 0.10, 0.15, 0.20, 0.15, 0.10, 0.08, 0.05, 0.02],
        ))

        # ── Rating (1.0–5.0): normally distributed around 4.2 ──
        rating = round(np.clip(np.random.normal(4.2, 0.6), 1.0, 5.0), 1)

        # ── Num subscribers: log-normal, skewed ──
        base_subs = np.random.lognormal(mean=8, sigma=1.5)
        # Boost for popular categories
        if category in ("Web Development", "Data Science"):
            base_subs *= 1.5
        # Boost for free courses
        if price == 0:
            base_subs *= 2.0
        num_subscribers = max(50, int(base_subs))

        # ── Num reviews: fraction of subscribers ──
        review_rate = np.random.uniform(0.02, 0.15)
        num_reviews = max(1, int(num_subscribers * review_rate))

        # ── Completion rate: influenced by rating, duration, level ──
        base_completion = 55  # percent
        # Higher rating → higher completion (stronger effect)
        base_completion += (rating - 3.0) * 12
        # Longer courses → lower completion (milder penalty)
        duration_penalty = min(content_duration * 0.2, 12)
        base_completion -= duration_penalty
        # Level effect (stronger)
        level_bonus = {
            "Beginner": 12,
            "Intermediate": 2,
            "Advanced": -10,
            "All Levels": 7,
        }
        base_completion += level_bonus[level]
        # Wider random noise for more spread
        base_completion += np.random.normal(0, 14)
        completion_rate = round(np.clip(base_completion, 5, 98), 1)

        # ── Watch time (hours): fraction of content_duration * completion ──
        watch_fraction = (completion_rate / 100) * np.random.uniform(0.7, 1.1)
        watch_time = round(content_duration * watch_fraction, 1)

        data.append({
            "course_id": f"CRS{i+1:05d}",
            "course_name": course_name,
            "category": category,
            "rating": rating,
            "num_subscribers": num_subscribers,
            "num_reviews": num_reviews,
            "content_duration": content_duration,
            "num_lectures": num_lectures,
            "level": level,
            "completion_rate": completion_rate,
            "watch_time": watch_time,
            "price": price,
            "published_year": published_year,
        })

    return pd.DataFrame(data)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = generate_dataset()
    output_path = os.path.join(OUTPUT_DIR, "courses.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Dataset generated: {output_path}")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Year range: {df['published_year'].min()}–{df['published_year'].max()}")
    print(f"   Categories: {df['category'].nunique()}")
    print(f"\n{df.head()}")
