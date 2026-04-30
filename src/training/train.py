"""Model training with MLflow experiment tracking."""

import mlflow
import mlflow.sklearn
import pandas as pd
from omegaconf import DictConfig
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

from src.training.mlflow_config import setup_mlflow


def get_model(cfg: DictConfig):
    """Instantiate the model from Hydra config."""
    if cfg.model.name == "random_forest":
        return RandomForestClassifier(
            n_estimators=cfg.model.n_estimators,
            max_depth=cfg.model.max_depth,
            random_state=cfg.model.random_state,
        )
    elif cfg.model.name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=cfg.model.n_estimators,
            max_depth=cfg.model.max_depth,
            learning_rate=cfg.model.learning_rate,
            random_state=cfg.model.random_state,
        )
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")


def train_model(df: pd.DataFrame, preprocessor, cfg: DictConfig) -> None:
    """Train, evaluate, log with MLflow, and save the model."""

    # Setup MLflow → DagsHub
    setup_mlflow(cfg)

    # Prepare features and target
    feature_cols = list(cfg.features.numeric) + list(cfg.features.categorical)
    X = df[feature_cols]
    y = df[cfg.data.target_column]

    # Build sklearn pipeline
    model = get_model(cfg)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])

    # Start MLflow run
    with mlflow.start_run(run_name=cfg.model.name):

        # ① Log all hyperparameters from Hydra config
        mlflow.log_param("model_type", cfg.model.name)
        mlflow.log_param("n_estimators", cfg.model.n_estimators)
        mlflow.log_param("max_depth", cfg.model.max_depth)
        mlflow.log_param("cv_folds", cfg.pipeline.cv_folds)

        # Log learning_rate only for gradient boosting
        if hasattr(cfg.model, "learning_rate"):
            mlflow.log_param("learning_rate", cfg.model.learning_rate)

        # ② Cross-validate and log metrics
        cv_scores = cross_val_score(
            pipeline, X, y,
            cv=cfg.pipeline.cv_folds,
            scoring="accuracy"
        )
        mean_acc = cv_scores.mean()
        std_acc = cv_scores.std()

        mlflow.log_metric("cv_accuracy_mean", mean_acc)
        mlflow.log_metric("cv_accuracy_std", std_acc)

        print(f"📊 {cfg.model.name} CV Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

        # ③ Fit on full data
        pipeline.fit(X, y)

        # ④ Log the model artifact to MLflow
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=f"titanic-{cfg.model.name}",
        )

        # ⑤ Also save .pkl locally (for DVC tracking)
        model_path = Path(cfg.paths.models_dir) / f"{cfg.model.name}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_path)

        print(f"💾 Model saved → {model_path}")
        print(f"📡 Logged to MLflow run: {mlflow.active_run().info.run_id}")