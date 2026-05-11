# Titanic Survival Prediction — MLOps Pipeline

A production-ready MLOps pipeline that predicts Titanic survival using Random Forest and Gradient Boosting classifiers, built progressively across 6 labs covering the full ML lifecycle: data versioning, experiment tracking, model registry, online serving, and batch inference.

![Python](https://img.shields.io/badge/python-3.11-blue)
![DVC](https://img.shields.io/badge/dvc-tracked-green)
![DagsHub](https://img.shields.io/badge/dagshub-remote%20storage-orange)
![Hydra](https://img.shields.io/badge/hydra-configurable-purple)
![MLflow](https://img.shields.io/badge/mlflow-tracking-blue)
![FastAPI](https://img.shields.io/badge/fastapi-serving-teal)
![Docker](https://img.shields.io/badge/docker-containerized-blue)
![Prefect](https://img.shields.io/badge/prefect-orchestration-violet)
![MotherDuck](https://img.shields.io/badge/motherduck-cloud%20database-yellow)

---

## 🗂️ Project Structure

```
lab0-titanic-pipeline/
├── app/                         # FastAPI serving application (Lab 6)
│   ├── __init__.py
│   ├── main.py                  # API endpoints + model loading on startup
│   └── schemas.py               # Pydantic request/response validation models
├── bruno/                       # Bruno API test collection (Lab 6)
│   └── Titanic API/
│       ├── Health Check.yml
│       ├── Predict Single.yml
│       └── Predict Multiple.yml
├── conf/                        # Hydra configuration files (Lab 1)
│   ├── config.yaml              # Root config — ties all groups together
│   ├── data/
│   │   └── titanic.yaml         # Data paths and target column
│   ├── features/
│   │   └── titanic.yaml         # Numeric and categorical feature lists
│   └── model/
│       ├── random_forest.yaml   # RF hyperparameters
│       └── gradient_boosting.yaml
├── data/
│   ├── raw/                     # Original CSVs (gitignored, tracked by DVC)
│   └── processed/               # Cleaned & engineered data
├── models/                      # Trained models (.pkl, tracked by DVC)
├── notebooks/                   # EDA notebook
├── pipelines/                   # Prefect batch inference pipeline (Lab 7)
│   ├── __init__.py
│   └── batch_inference.py       # 4-task Prefect flow: extract→transform→predict→save
├── scripts/
│   ├── predict.py               # Load Production model & run predictions (Lab 4)
│   └── load_to_motherduck.py    # One-time upload of test.csv to MotherDuck (Lab 7)
├── src/
│   └── training/
│       ├── __init__.py
│       ├── data_loader.py       # Load and validate raw CSV
│       ├── feature_engineering.py  # FamilySize, Title, Age bands, Fare categories
│       ├── preprocessor.py      # Build sklearn ColumnTransformer
│       ├── train.py             # Train model + log everything to MLflow (Lab 4)
│       └── mlflow_config.py     # DagsHub MLflow tracking connection (Lab 4)
├── .dockerignore                # Files excluded from Docker image (Lab 6)
├── .dvc/                        # DVC internal config (Lab 2)
├── .gitignore
├── data/raw/train.csv.dvc       # DVC pointer for training data (Lab 2)
├── dvc.yaml                     # DVC pipeline stage definitions (Lab 2)
├── dvc.lock                     # Locked pipeline state for reproducibility (Lab 2)
├── Dockerfile                   # Container definition for API serving (Lab 6)
├── main.py                      # Hydra pipeline entry point
├── run_all.py                   # Trains all models in one command (Lab 2)
├── Makefile                     # Code quality shortcuts
└── pyproject.toml               # Project config & all dependencies
```

---

## ⚙️ Full Pipeline Flow

```
Raw CSV → Feature Engineering → Preprocessing → Train Models → MLflow Registry → FastAPI → Docker
  (DVC)      (Hydra config)      (sklearn)       (DVC + MLflow)   (DagsHub)      (REST API) (Container)

MotherDuck (test data) → Prefect Pipeline → Batch Predictions → MotherDuck (results)
```

---

## 🚀 Quickstart

### Run the Training Pipeline

```bash
# 1. Clone this repo
git clone https://github.com/Omar-Shamakh/lab0-titanic-pipeline.git
cd lab0-titanic-pipeline

# 2. Set up environment
uv sync

# 3. Set your tokens
$env:DAGSHUB_TOKEN = "your_dagshub_token"   # PowerShell
export DAGSHUB_TOKEN="your_dagshub_token"   # Linux/Mac

# 4. Pull data and models from DagsHub remote storage
uv run dvc pull

# 5. Run the full training pipeline
uv run dvc repro
```

> **DVC remote credentials** — set once with:
> ```bash
> uv run dvc remote modify --local origin access_key_id YOUR_DAGSHUB_USERNAME
> uv run dvc remote modify --local origin secret_access_key YOUR_DAGSHUB_TOKEN
> ```

### Run the Online API Locally

```bash
$env:DAGSHUB_TOKEN = "your_token"
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

### Run the Online API with Docker

```bash
docker build -t titanic-api:latest .
docker run -p 8000:8000 \
  -e DAGSHUB_TOKEN="your_token" \
  --name titanic-api \
  titanic-api:latest
```

### Run the Batch Inference Pipeline

```bash
# Set both tokens
$env:DAGSHUB_TOKEN    = "your_dagshub_token"
$env:MOTHERDUCK_TOKEN = "your_motherduck_token"

# Start Prefect server (keep this terminal open)
uv run prefect server start

# In a new terminal — run the batch pipeline
uv run prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
uv run python -m pipelines.batch_inference
```

Monitor the run at `http://127.0.0.1:4200`.

---

## 🤖 Models

| Model | CV Accuracy | CV Std | Status |
|-------|-------------|--------|--------|
| Random Forest | ~83% | ±0.016 | ✅ Production |
| Gradient Boosting | ~81% | ±0.037 | Staging |

---

## 🔧 Configuration (Hydra)

Switch models or override any hyperparameter from the CLI — no code changes needed:

```bash
uv run python main.py                                      # default (Random Forest)
uv run python main.py model=gradient_boosting              # switch model
uv run python main.py model.n_estimators=200 model.max_depth=8  # override params
uv run python run_all.py                                   # train all models
uv run python main.py --cfg job                            # print resolved config
```

---

## 📦 DVC Pipeline

```bash
uv run dvc repro    # re-run only changed stages
uv run dvc push     # push data and models to DagsHub
uv run dvc pull     # pull data and models from DagsHub
uv run dvc status   # check pipeline status
```

---

## 📊 MLflow Experiment Tracking

Every training run is automatically logged to DagsHub's MLflow tracking server.

```bash
uv run python run_all.py
```

**View experiments:** `https://dagshub.com/omar.sameh.shamakh/lab0-titanic-pipeline/experiments`

| Type | Details |
|------|---------|
| Parameters | model_type, n_estimators, max_depth, learning_rate, cv_folds |
| Metrics | cv_accuracy_mean, cv_accuracy_std |
| Artifacts | Trained model registered in MLflow Model Registry |

---

## 🔮 Model Registry & Production

The best model (Random Forest) is registered in the MLflow Model Registry on DagsHub with a `@production` alias.

**View registry:** `https://dagshub.com/omar.sameh.shamakh/lab0-titanic-pipeline/models`

```bash
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

## 🌐 REST API (FastAPI)

The trained model is served as a REST API with full input validation and batch support.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check + model status |
| POST | `/predict` | Predict survival (single or batch) |

**Example request:**
```json
POST http://localhost:8000/predict
{
  "passengers": [
    {
      "Pclass": 1, "Sex": "female", "Age": 29.0,
      "SibSp": 0, "Parch": 0, "Fare": 211.3,
      "Embarked": "S", "Family_size": 1, "Alone": 1,
      "Initial": "Mrs", "Fare_cat": 3, "Age_band": 2
    }
  ]
}
```

**Example response:**
```json
{
  "model_name": "titanic-random_forest@production",
  "total_passengers": 1,
  "predictions": [
    {
      "passenger_index": 0,
      "survived": true,
      "survival_probability": 0.931,
      "prediction_label": "SURVIVED"
    }
  ]
}
```

---

## ⚙️ Batch Inference Pipeline (Prefect + MotherDuck)

A scheduled batch pipeline that processes all 418 test passengers at once and stores predictions in MotherDuck.

**Pipeline steps:**

| Step | Task | Description |
|------|------|-------------|
| 1 | `extract-from-motherduck` | Pulls raw test data from MotherDuck via SQL |
| 2 | `transform-features` | Applies the same feature engineering as training |
| 3 | `load-production-model` | Fetches the Production model from DagsHub MLflow registry |
| 4 | `predict-and-save` | Runs batch predictions and saves results to MotherDuck |

**MotherDuck tables:**

| Table | Description |
|-------|-------------|
| `titanic_ml.raw_test` | 418 raw passenger rows from test.csv |
| `titanic_ml.predictions` | Predictions: PassengerId, Survived, SurvivalProbability, PredictedAt, ModelName |

**Query predictions in MotherDuck:**
```sql
SELECT Pclass,
       COUNT(*) as total,
       SUM(p.Survived) as predicted_survived,
       ROUND(AVG(p.SurvivalProbability) * 100, 1) as avg_survival_pct
FROM predictions p
JOIN raw_test r ON p.PassengerId = r.PassengerId
GROUP BY Pclass ORDER BY Pclass;
```

---

## 🛠️ Code Quality

```bash
make format   # runs isort + black + ruff
```

---

## 🧪 Labs Progress

| Lab | Topic | Key Tools | Status |
|-----|-------|-----------|--------|
| Lab 0 | Basic pipeline — data loading, feature engineering, training | Python, sklearn | Done |
| Lab 1 | Configurable pipeline — YAML configs, CLI overrides, config groups | Hydra | Done |
| Lab 2 | Data versioning — remote storage, reproducible pipeline | DVC, DagsHub | Done |
| Lab 4 | Experiment tracking — logging, model registry, production promotion | MLflow, DagsHub | Done |
| Lab 6 | Online serving — REST API, batch inference, containerization | FastAPI, Docker, Bruno | Done |
| Lab 7 | Batch inference — scheduled pipeline, cloud database, orchestration | Prefect, MotherDuck | Done |
