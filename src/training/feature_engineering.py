"""Feature engineering for the Titanic dataset."""

import pandas as pd
from omegaconf import DictConfig


def extract_initial(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Initial"] = df["Name"].str.extract(r"([A-Za-z]+)\.")
    df["Initial"] = df["Initial"].replace(
        ["Mlle","Mme","Ms","Dr","Major","Lady","Countess","Jonkheer","Col","Rev","Capt","Sir","Don"],
        ["Miss","Miss","Miss","Mr","Mr","Mrs","Mrs","Other","Other","Other","Mr","Mr","Mr"],
    )
    return df


def fill_missing_age(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    age_map = {"Mr": 33, "Mrs": 36, "Master": 5, "Miss": 22, "Other": 46}
    for initial, age in age_map.items():
        df.loc[(df["Age"].isnull()) & (df["Initial"] == initial), "Age"] = age
    return df


def create_age_band(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Age_band"] = pd.cut(
        df["Age"], bins=[0, 16, 32, 48, 64, 80], labels=[0, 1, 2, 3, 4]
    ).astype(int)
    return df


def create_family_size(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Family_size"] = df["SibSp"] + df["Parch"]
    df["Alone"] = (df["Family_size"] == 0).astype(int)
    return df


def create_fare_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Fare_cat"] = pd.cut(
        df["Fare"], bins=[-1, 7.91, 14.454, 31.0, 600], labels=[0, 1, 2, 3]
    ).astype(int)
    return df


def engineer_features(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Run all feature engineering steps, then drop columns from config."""
    df = extract_initial(df)
    df = fill_missing_age(df)
    df = create_age_band(df)
    df = create_family_size(df)
    df = create_fare_category(df)
    drop_cols = list(cfg.features.drop_columns)
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    print("? Feature engineering complete")
    return df
