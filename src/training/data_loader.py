"""Data loading utilities for the Titanic pipeline."""

import pandas as pd
from omegaconf import DictConfig


def load_data(cfg: DictConfig) -> pd.DataFrame:
    """Load raw data from the path specified in config."""
    path = cfg.data.raw_path
    df = pd.read_csv(path)
    print(f"✅ Loaded data: {len(df)} rows, {len(df.columns)} columns")
    return df