"""Model training script for the Titanic pipeline."""

import os

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


def build_rf_pipeline(params: dict) -> Pipeline:
    """Build a scikit-learn Pipeline: OrdinalEncoder → RandomForest."""
    return Pipeline(
        steps=[
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=params["n_estimators"],
                    random_state=params["random_state"],
                ),
            )
        ]
    )


def build_gbm_pipeline(params: dict) -> Pipeline:
    """Build a scikit-learn Pipeline: GradientBoostingClassifier."""
    return Pipeline(
        steps=[
            (
                "gbm",
                GradientBoostingClassifier(
                    n_estimators=params["n_estimators"],
                    learning_rate=params["learning_rate"],
                    random_state=params["random_state"],
                ),
            )
        ]
    )


def train_and_evaluate(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, model_name: str
) -> Pipeline:
    """Train a pipeline and print cross-validation accuracy."""
    scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"📊 {model_name} CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    pipeline.fit(X, y)
    print(f"✅ {model_name} trained successfully")
    return pipeline


def save_model(pipeline: Pipeline, model_dir: str, filename: str) -> None:
    """Save a trained pipeline to disk using joblib."""
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, filename)
    joblib.dump(pipeline, path)
    print(f"💾 Model saved → {path}")


def run_training(df: pd.DataFrame, config: dict) -> None:
    """Run full training: split features/target, train models, save."""
    target = config["features"]["target"]
    X = df.drop(columns=[target])
    y = df[target]

    rf_pipeline = build_rf_pipeline(config["models"]["random_forest"])
    rf_pipeline = train_and_evaluate(rf_pipeline, X, y, "RandomForest")
    save_model(rf_pipeline, config["paths"]["model_dir"], "random_forest.pkl")

    gbm_pipeline = build_gbm_pipeline(config["models"]["gradient_boosting"])
    gbm_pipeline = train_and_evaluate(gbm_pipeline, X, y, "GradientBoosting")
    save_model(gbm_pipeline, config["paths"]["model_dir"], "gradient_boosting.pkl")
