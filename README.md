# E-Learning Engagement & Student Retention

<p>
  <a href="https://edumetrics-brown.vercel.app"><img src="https://img.shields.io/badge/live-demo-000000?logo=vercel&logoColor=white" alt="Live demo"></a>
  <img src="https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Three.js-WebGL-000000?logo=threedotjs&logoColor=white" alt="Three.js">
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Plotly-charts-3F4F75?logo=plotly&logoColor=white" alt="Plotly">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License: MIT"></a>
</p>

A Python-based data analytics and interactive dashboard capstone project.

This project analyzes the key drivers of student retention, engagement, and dropout rates in e-learning environments (analyzing 2,500 courses spanning from 2010 to 2025). It uses a custom-generated realistic dataset modeled after patterns found on major learning platforms.

## Architecture

1. **Python Data Pipeline** (`backend/`)
   - `generate_dataset.py`: Generates synthetic data.
   - `preprocessing.py`: Cleans data, normalizes features, and calculates engagement scores.
   - `analysis.py`: Generates static PNG charts (Seaborn) and interactive HTML charts (Plotly).
   - `insights.py`: Extracts text insights and runs Logistic Regression for dropout prediction.

2. **Frontend** (`frontend/`)
   - A single-page, scroll-driven 3D showcase built with Three.js and plain HTML/CSS/JS.
   - 2,500 particles, one per course, reshape into each finding: the course cloud, length × rating columns, completion rings by level, model results and category pillars.
   - Live: https://edumetrics-brown.vercel.app

## Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Data Pipeline
Run the scripts in order from the project root:
```bash
python backend/generate_dataset.py
python backend/preprocessing.py
python backend/analysis.py
python backend/insights.py
```

### 3. Open the Frontend
The frontend is self-contained, so you can open `frontend/index.html` directly in a browser, or serve it locally:

```bash
python -m http.server 8000
```

Then open http://localhost:8000/frontend/ (or use the live link above).

## Highlights for Viva / Demonstration
- **Scroll story**: One scroll moves to the next chapter; the particles fly from one chart into the next.
- **ML Integration**: The model chapter shows the logistic regression's 81.9% accuracy, precision, recall and its four outcome groups.
- **Drivers**: The 3D columns show completion falling from 83.6% (short, top-rated) to 50.6% (long, poorly rated).
- **Automated Insights**: Every figure comes from the Python pipeline's output.
