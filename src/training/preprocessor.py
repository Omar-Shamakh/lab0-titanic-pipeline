"""Data preprocessing for the Titanic pipeline."""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from omegaconf import DictConfig


def build_preprocessor(cfg: DictConfig) -> ColumnTransformer:
    """Build sklearn ColumnTransformer from config feature lists."""

    # Read feature lists from config
    numeric_features = list(cfg.features.numeric)
    categorical_features = list(cfg.features.categorical)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    print("✅ Preprocessor built from config")
    return preprocessor