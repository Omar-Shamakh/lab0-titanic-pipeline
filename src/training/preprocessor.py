"""Data preprocessing for the Titanic pipeline."""

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def fill_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing Embarked values with most frequent port (S)."""
    df = df.copy()
    df["Embarked"].fillna("S", inplace=True)
    return df


def encode_categoricals(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Encode categorical columns with OrdinalEncoder.

    OrdinalEncoder converts text categories to integers:
    female -> 0, male -> 1, etc.
    Unlike get_dummies, it works inside sklearn Pipelines.
    """
    df = df.copy()
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df[cat_cols] = enc.fit_transform(df[cat_cols].astype(str))
    return df, enc


def drop_unused_columns(df: pd.DataFrame, drop_cols: list[str]) -> pd.DataFrame:
    """Drop columns not useful for modelling."""
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    return df.drop(columns=cols_to_drop)


def run_preprocessing(
    df: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, OrdinalEncoder]:
    """Full preprocessing pipeline: fill nulls, encode, drop columns.

    Returns:
        Preprocessed DataFrame and the fitted encoder.
    """
    df = fill_embarked(df)
    df = drop_unused_columns(df, config["features"]["drop_cols"])
    df, encoder = encode_categoricals(df, config["features"]["categorical_cols"])
    print("✅ Preprocessing complete")
    return df, encoder
