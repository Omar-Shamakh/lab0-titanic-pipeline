# Titanic Survival Prediction — MLOps Pipeline

A production-ready MLOps training pipeline that predicts Titanic survival using Random Forest and Gradient Boosting classifiers, built progressively across 3 labs.

![Python](https://img.shields.io/badge/python-3.11-blue)
![DVC](https://img.shields.io/badge/dvc-tracked-green)
![DagsHub](https://img.shields.io/badge/dagshub-remote%20storage-orange)
![Hydra](https://img.shields.io/badge/hydra-configurable-purple)

---

## Project Structure

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
├── src/
│   └── training/
│       ├── data_loader.py
│       ├── feature_engineering.py
│       ├── preprocessor.py
│       └── train.py
├── dvc.yaml                     # DVC pipeline definition (Lab 2)
├── dvc.lock                     # Locked pipeline state (Lab 2)
├── main.py                      # Pipeline entry point (Hydra)
├── run_all.py                   # Trains all models in one command (Lab 2)
└── pyproject.toml               # Project config & dependencies
```

---

## Pipeline Flow

```
Raw CSV → Feature Engineering → Preprocessing → Train Models → Save Models
  (DVC)        (Hydra config)     (Hydra config)   (DVC tracked)
```

---

## Quickstart

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

## Models

| Model | CV Accuracy |
|-------|-------------|
| Random Forest | ~83% |
| Gradient Boosting | ~82% |

---

## Configuration (Hydra)

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

## DVC Pipeline

```bash
# Run the full pipeline (smart — skips unchanged stages)
uv run dvc repro

# Push data and models to DagsHub remote storage
uv run dvc push

# Pull data and models from DagsHub remote storage
uv run dvc pull
```

---

## Code Quality

```bash
make format   # runs isort + black + ruff
```

---

## Labs Progress

| Lab | Topic | Status |
|-----|-------|--------|
| Lab 0 | Basic pipeline — data loading, feature engineering, training | ✅ Done |
| Lab 1 | Hydra configuration — YAML configs, CLI overrides, config groups | ✅ Done |
| Lab 2 | DVC + DagsHub — remote storage, versioned pipeline, reproducibility | ✅ Done |
