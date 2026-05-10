"""
Batch inference pipeline using Prefect + MotherDuck + MLflow.

Flow:
  1. Extract raw test data from MotherDuck
  2. Transform: apply feature engineering
  3. Load Production model from MLflow registry
  4. Predict and save results back to MotherDuck
"""

import os
import duckdb
import mlflow
import mlflow.sklearn
import pandas as pd
from datetime import datetime
from prefect import flow, task, get_run_logger
from src.training.feature_engineering import (
    extract_initial, fill_missing_age, create_age_band,
    create_family_size, create_fare_category
)

MODEL_NAME = "titanic-random_forest"
MODEL_ALIAS = "production"
DB_NAME = "titanic_ml"


def get_conn():
    token = os.environ["MOTHERDUCK_TOKEN"]
    return duckdb.connect(f"md:{DB_NAME}?motherduck_token={token}")


@task(name="extract-from-motherduck", retries=2)
def extract_data() -> pd.DataFrame:
    """Extract raw test data from MotherDuck."""
    logger = get_run_logger()
    logger.info("📥 Extracting data from MotherDuck...")
    conn = get_conn()
    df = conn.execute("SELECT * FROM raw_test").df()
    conn.close()
    logger.info(f"✅ Extracted {len(df)} rows")
    return df


@task(name="transform-features")
def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering to raw test data."""
    logger = get_run_logger()
    logger.info("⚙️ Applying feature engineering...")

    passenger_ids = df["PassengerId"].copy()

    # Fill missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna("S")

    # Apply the same feature engineering steps as training
    df = extract_initial(df)
    df = fill_missing_age(df)
    df = create_age_band(df)
    df = create_family_size(df)
    df = create_fare_category(df)

    # Select only the model features
    feature_cols = [
        "Pclass", "Sex", "Age", "SibSp", "Parch",
        "Fare", "Embarked", "Family_size", "Alone",
        "Initial", "Fare_cat", "Age_band"
    ]
    df_features = df[feature_cols].copy()
    df_features.insert(0, "PassengerId", passenger_ids.values)
    logger.info(f"✅ Feature engineering complete: {df_features.shape}")
    return df_features


@task(name="load-production-model", retries=2)
def load_model():
    """Load the Production model from MLflow registry on DagsHub."""
    logger = get_run_logger()
    os.environ["MLFLOW_TRACKING_USERNAME"] = "omar.sameh.shamakh"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ.get("DAGSHUB_TOKEN", "")
    mlflow.set_tracking_uri(
        "https://dagshub.com/omar.sameh.shamakh/lab0-titanic-pipeline.mlflow"
    )
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    logger.info(f"📦 Loading model: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    logger.info("✅ Model loaded successfully")
    return model


@task(name="predict-and-save")
def predict_and_save(df: pd.DataFrame, model) -> pd.DataFrame:
    """Run predictions and save results to MotherDuck."""
    logger = get_run_logger()
    logger.info(f"🔮 Running predictions for {len(df)} passengers...")

    passenger_ids = df["PassengerId"]
    feature_cols = [c for c in df.columns if c != "PassengerId"]
    X = df[feature_cols]

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    results = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": predictions.astype(int),
        "SurvivalProbability": probabilities.round(4),
        "PredictedAt": datetime.now().isoformat(),
        "ModelName": f"{MODEL_NAME}@{MODEL_ALIAS}",
    })

    conn = get_conn()
    conn.execute("CREATE OR REPLACE TABLE predictions AS SELECT * FROM results")
    conn.close()

    survived = results["Survived"].sum()
    logger.info(f"✅ Saved {len(results)} predictions to MotherDuck")
    logger.info(f"📊 Predicted survivors: {survived}/{len(results)} ({survived/len(results)*100:.1f}%)")
    return results


@flow(name="titanic-batch-inference", log_prints=True)
def batch_inference_flow():
    """Extract → Transform → Load Model → Predict → Save"""
    print("🚀 Starting Titanic Batch Inference Pipeline...")
    raw_df = extract_data()
    transformed_df = transform_data(raw_df)
    model = load_model()
    results = predict_and_save(transformed_df, model)
    print(f"🎉 Pipeline complete! {len(results)} predictions saved to MotherDuck.")
    return results


if __name__ == "__main__":
    batch_inference_flow()