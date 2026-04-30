# Titanic Survival Prediction — MLOps Pipeline

A production-ready MLOps training pipeline that predicts Titanic survival using Random Forest and Gradient Boosting classifiers, built progressively across 4 labs.

![Python](https://img.shields.io/badge/python-3.11-blue)
![DVC](https://img.shields.io/badge/dvc-tracked-green)
![DagsHub](https://img.shields.io/badge/dagshub-remote%20storage-orange)
![Hydra](https://img.shields.io/badge/hydra-configurable-purple)
![MLflow](https://img.shields.io/badge/mlflow-tracking-blue)

---

## 🗂️ Project Structure

```
lab0-titanic-pipeline/
├── conf/                        # Hydra configuration files (Lab 1)
│   ├── config.yaml              # Root config — ties all groups together
│   ├── data/
│   │   └── titanic.yaml         # Data paths
│   ├── features/
│   │   └── titanic.yaml         # Feature lists
│   └── model/
│       ├── random_forest.yaml   # RF hyperparameters
│       └── gradient_boosting.yaml
├── data/
│   ├── raw/                     # Original CSV (tracked by DVC)
│   └── processed/               # Cleaned & engineered data
├── models/                      # Trained models (.pkl, tracked by DVC)
├── notebooks/                   # EDA notebook
├── scripts/
│   └── predict.py               # Load Production model & predict (Lab 4)
├── src/
│   └── training/
│       ├── data_loader.py
│       ├── feature_engineering.py
│       ├── preprocessor.py
│       ├── train.py             # Training + MLflow logging (Lab 4)
│       └── mlflow_config.py     # DagsHub MLflow connection (Lab 4)
├── dvc.yaml                     # DVC pipeline definition (Lab 2)
├── dvc.lock                     # Locked pipeline state (Lab 2)
├── main.py                      # Pipeline entry point (Hydra)
├── run_all.py                   # Trains all models in one command (Lab 2)
└── pyproject.toml               # Project config & dependencies
```

---

## ⚙️ Pipeline Flow

```
Raw CSV → Feature Engineering → Preprocessing → Train Models → Save Models → MLflow Registry
  (DVC)        (Hydra config)     (Hydra config)   (DVC tracked)   (DagsHub)
```

---

## 🚀 Quickstart

```bash
# 1. Clone this repo
git clone https://github.com/Omar-Shamakh/lab0-titanic-pipeline.git
cd lab0-titanic-pipeline

# 2. Set up environment with uv
uv sync

# 3. Pull data and models from DagsHub remote storage
dvc pull

# 4. Run the full pipeline
uv run dvc repro
```

> **Note:** `dvc pull` requires DagsHub credentials. Set them once with:
> ```bash
> uv run dvc remote modify --local origin access_key_id YOUR_DAGSHUB_USERNAME
> uv run dvc remote modify --local origin secret_access_key YOUR_DAGSHUB_TOKEN
> ```

---

## 🤖 Models

| Model | CV Accuracy |
|-------|-------------|
| Random Forest | ~83% |
| Gradient Boosting | ~81% |

---

## 🔧 Configuration (Hydra)

Switch models or override hyperparameters from the CLI — no code changes needed:

```bash
# Run with default model (Random Forest)
uv run python main.py

# Switch to Gradient Boosting
uv run python main.py model=gradient_boosting

# Override hyperparameters
uv run python main.py model.n_estimators=200 model.max_depth=8

# Print the full resolved config
uv run python main.py --cfg job
```

---

## 📦 DVC Pipeline

```bash
# Run the full pipeline (smart — skips unchanged stages)
uv run dvc repro

# Push data and models to DagsHub remote storage
uv run dvc push

# Pull data and models from DagsHub remote storage
uv run dvc pull
```

---

## 📊 MLflow Experiment Tracking

Every training run is automatically logged to DagsHub's MLflow tracking server:

```bash
# Run the pipeline — experiments are logged automatically
uv run python run_all.py
```

View experiments at:
```
https://dagshub.com/omar.sameh.shamakh/lab0-titanic-pipeline/experiments
```

What gets logged per run:
- **Parameters** — model type, n_estimators, max_depth, learning_rate, cv_folds
- **Metrics** — cv_accuracy_mean, cv_accuracy_std
- **Model artifact** — registered in MLflow Model Registry

---

## 🔮 Load Production Model & Predict

The best model (Random Forest) is registered in the MLflow Model Registry with a `production` alias. Load and predict with:

```bash
# Set your DagsHub token first
$env:DAGSHUB_TOKEN = "your_token_here"   # PowerShell
export DAGSHUB_TOKEN="your_token_here"   # Linux/Mac

# Run predictions
uv run python scripts/predict.py
```

Output:
```
📦 Loading model: models:/titanic-random_forest@production
✅ Model loaded successfully!

🚢 Titanic Survival Predictions:
---------------------------------------------
Passenger 1: ✅ SURVIVED (93.1% confidence)
Passenger 2: ❌ DID NOT SURVIVE (90.6% confidence)
Passenger 3: ✅ SURVIVED (81.1% confidence)
```

---

## 🛠️ Code Quality

```bash
make format   # runs isort + black + ruff
```

---

## 🧪 Labs Progress

| Lab | Topic | Status |
|-----|-------|--------|
| Lab 0 | Basic pipeline — data loading, feature engineering, training | ✅ Done |
| Lab 1 | Hydra configuration — YAML configs, CLI overrides, config groups | ✅ Done |
| Lab 2 | DVC + DagsHub — remote storage, versioned pipeline, reproducibility | ✅ Done |
| Lab 4 | MLflow tracking — experiment logging, model registry, production deployment | ✅ Done |
