# lab0-titanic-pipeline
### Titanic Survival Prediction MLOps Pipeline Lab0

# 🚢 Titanic Survival Prediction — MLOps Pipeline

A production-ready MLOps training pipeline that predicts Titanic survival
using Random Forest and Gradient Boosting classifiers.

## 🗂️ Project Structure

```
ITI-MLOps/
├── conf/               # Configuration files
│   └── config.yaml     # Paths, features, model hyperparameters
├── data/
│   ├── raw/            # Original unmodified CSV files
│   └── processed/      # Cleaned & engineered data
├── models/             # Saved trained models (.pkl)
├── notebooks/          # EDA notebook
├── src/
│   └── training/       # Python pipeline scripts
│       ├── data_loader.py
│       ├── feature_engineering.py
│       ├── preprocessor.py
│       └── train.py
├── main.py             # Pipeline entry point
└── pyproject.toml      # Project config & dependencies
```

## ⚙️ Pipeline Flow

Raw CSV → Feature Engineering → Preprocessing → Train Models → Save Models

## 🚀 Quickstart

```bash
# 1. Clone this repo
git clone https://github.com/YOUR_USERNAME/lab0-titanic-pipeline.git
cd lab0-titanic-pipeline

# 2. Set up environment with uv
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync

# 3. Add Titanic data
# Download train.csv from Kaggle → put in data/raw/

# 4. Run the pipeline
uv run python main.py
```

## 🤖 Models

| Model | CV Accuracy |
|-------|-------------|
| Random Forest | ~83% |
| Gradient Boosting | ~82% |

## 🛠️ Code Quality

```bash
make format  # runs isort + black + ruff
```
