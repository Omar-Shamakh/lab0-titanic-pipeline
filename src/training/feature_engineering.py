"""Feature engineering for the Titanic dataset."""

import pandas as pd


def extract_initial(df: pd.DataFrame) -> pd.DataFrame:
    """Extract title (Initial) from passenger Name column.

    Maps rare titles to common groups: Miss, Mr, Mrs, Master, Other.
    """
    df = df.copy()
    df["Initial"] = df["Name"].str.extract(r"([A-Za-z]+)\.")
    df["Initial"].replace(
        [
            "Mlle",
            "Mme",
            "Ms",
            "Dr",
            "Major",
            "Lady",
            "Countess",
            "Jonkheer",
            "Col",
            "Rev",
            "Capt",
            "Sir",
            "Don",
        ],
        [
            "Miss",
            "Miss",
            "Miss",
            "Mr",
            "Mr",
            "Mrs",
            "Mrs",
            "Other",
            "Other",
            "Other",
            "Mr",
            "Mr",
            "Mr",
        ],
        inplace=True,
    )
    return df


def fill_missing_age(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing Age values using group mean by Initial (title).

    Based on EDA: Master ≈ 5, Miss ≈ 22, Mr ≈ 33, Mrs ≈ 36, Other ≈ 46.
    """
    df = df.copy()
    age_map = {"Mr": 33, "Mrs": 36, "Master": 5, "Miss": 22, "Other": 46}
    for initial, age in age_map.items():
        df.loc[(df["Age"].isnull()) & (df["Initial"] == initial), "Age"] = age
    return df


def create_age_band(df: pd.DataFrame) -> pd.DataFrame:
    """Bin Age into 5 categories (0-4) for easier learning."""
    df = df.copy()
    df["Age_band"] = pd.cut(
        df["Age"], bins=[0, 16, 32, 48, 64, 80], labels=[0, 1, 2, 3, 4]
    ).astype(int)
    return df


def create_family_size(df: pd.DataFrame) -> pd.DataFrame:
    """Create Family_size and Alone features from SibSp and Parch."""
    df = df.copy()
    df["Family_size"] = df["SibSp"] + df["Parch"]
    df["Alone"] = (df["Family_size"] == 0).astype(int)
    return df


def create_fare_category(df: pd.DataFrame) -> pd.DataFrame:
    """Bin Fare into 4 categories based on quartiles."""
    df = df.copy()
    df["Fare_cat"] = pd.cut(
        df["Fare"], bins=[-1, 7.91, 14.454, 31.0, 600], labels=[0, 1, 2, 3]
    ).astype(int)
    return df


def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature engineering steps in sequence."""
    df = extract_initial(df)
    df = fill_missing_age(df)
    df = create_age_band(df)
    df = create_family_size(df)
    df = create_fare_category(df)
    print("✅ Feature engineering complete")
    return df
