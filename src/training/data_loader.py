"""Data loading utilities for the Titanic pipeline."""

import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw Titanic CSV data from the given path.

    Args:
        path: Path to the raw CSV file.

    Returns:
        DataFrame with the raw data.
    """
    df = pd.read_csv(path)
    print(f"✅ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
