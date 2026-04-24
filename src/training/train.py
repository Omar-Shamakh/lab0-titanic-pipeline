"""Model training script for the Titanic pipeline."""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from omegaconf import DictConfig


def get_model(cfg: DictConfig):
    """Instantiate the correct model from config."""
    model_name = cfg.model.name

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=cfg.model.n_estimators,
            max_depth=cfg.model.max_depth,
            min_samples_split=cfg.model.min_samples_split,
            min_samples_leaf=cfg.model.min_samples_leaf,
            random_state=cfg.pipeline.random_state,
        )
    elif model_name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=cfg.model.n_estimators,
            max_depth=cfg.model.max_depth,
            learning_rate=cfg.model.learning_rate,
            subsample=cfg.model.subsample,
            random_state=cfg.pipeline.random_state,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_model(
    df: pd.DataFrame,
    preprocessor,
    cfg: DictConfig,
) -> Pipeline:
    """Train model defined in config and save it."""

    target = cfg.data.target_column
    feature_cols = list(cfg.features.numeric) + list(cfg.features.categorical)

    X = df[feature_cols]
    y = df[target]

    # Build the full sklearn pipeline
    model = get_model(cfg)
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])

    # Cross-validation
    cv_scores = cross_val_score(
        full_pipeline, X, y,
        cv=cfg.pipeline.cv_folds,
        scoring="accuracy"
    )
    model_name = cfg.model.name
    print(f"📊 {model_name} CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Final fit on all data
    full_pipeline.fit(X, y)
    print(f"✅ {model_name} trained successfully")

    # Save the model
    save_dir = cfg.pipeline.save_dir
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}.pkl")
    joblib.dump(full_pipeline, save_path)
    print(f"💾 Model saved → {save_path}")

    return full_pipeline