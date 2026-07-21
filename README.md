# Binary Classification ML Pipeline

A full-stack **binary classification** web application with an end-to-end data science pipeline. Built entirely with **free, open-source tools** and deployable locally via **Docker**.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41-red)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

## Features

| Stage | Capabilities |
|-------|-------------|
| **Data Profiling** | ydata-profiling reports, correlation matrix, distributions |
| **Data Quality** | Multi-dimensional quality scoring (completeness, uniqueness, validity, consistency) |
| **Missing Values** | Auto-recommendations + mean/median/mode/KNN imputation |
| **Outlier Detection** | IQR & Z-score detection with cap/remove treatment |
| **Duplicate Detection** | Row-level duplicate analysis and removal |
| **Encoding** | Label, one-hot, target, binary, hashing encoders |
| **Scaling** | Standard, min-max, robust, max-abs scalers |
| **Feature Engineering** | Interactions, polynomials, log transforms, binning |
| **Variable Reduction** | Variance threshold + PCA |
| **Feature Selection** | SelectKBest, RFE, importance-based |
| **Model Selection** | 8 classifiers compared via cross-validated ROC-AUC |
| **Hyperparameter Tuning** | Optuna Bayesian optimization |
| **Model Explanation** | SHAP values, feature importance, ROC/confusion matrix |
| **Business Insights** | Actionable recommendations for stakeholders |
| **Executive Report** | Auto-generated PDF summary |

## Tech Stack (All Free)

- **UI**: [Streamlit](https://streamlit.io/)
- **ML**: scikit-learn, imbalanced-learn, category-encoders
- **Tuning**: [Optuna](https://optuna.org/)
- **Explainability**: [SHAP](https://shap.readthedocs.io/)
- **Profiling**: [ydata-profiling](https://github.com/ydataai/ydata-profiling)
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Reports**: fpdf2

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Build and run
docker compose up --build

# Open in browser
# http://localhost:8501
```

### Option 2: Local Python

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Generate sample data (optional)
python generate_sample_data.py

# Run the app
streamlit run app/main.py
```

## Usage

1. **Upload** a CSV file with a binary target column
2. **Select** the target column in the sidebar
3. **Walk through tabs** left to right — each step updates the processed dataset
4. **Compare models** and train the best one
5. **Generate** business insights and download the executive PDF report

### Sample Dataset

Run `python generate_sample_data.py` to create `data/sample_loan_default.csv` — a loan default prediction dataset with missing values, outliers, and duplicates built in for testing.

## Project Structure

```
├── app/
│   ├── main.py                  # Streamlit UI
│   ├── modules/
│   │   ├── data_profiling.py
│   │   ├── missing_values.py
│   │   ├── outliers.py
│   │   ├── data_quality.py
│   │   ├── duplicates.py
│   │   ├── encoding.py
│   │   ├── scaling.py
│   │   ├── feature_engineering.py
│   │   ├── variable_reduction.py
│   │   ├── feature_selection.py
│   │   ├── model_selection.py
│   │   ├── hyperparameter_tuning.py
│   │   ├── model_explanation.py
│   │   ├── business_insights.py
│   │   └── report_generator.py
│   └── utils/
│       └── helpers.py
├── data/                        # Upload / sample data
├── reports/                     # Generated PDF reports
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── generate_sample_data.py
```

## Deploy on Render

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Vekri/Binary-Classification/tree/master)

**One-click deploy:** click the button above → sign in with GitHub → approve the blueprint.

Render reads `render.yaml` automatically:
- **Build:** `pip install -r requirements.txt`
- **Start:** `streamlit run app/main.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`

Live URL: `https://binary-classification-ml.onrender.com` (after deploy)

Free tier may sleep after idle; first load can take ~30–60s.

Hub icon on [Singareddy AI](https://singareddy-ai.vercel.app) links to that Render URL.
