"""Main entry point for the Titanic MLOps training pipeline.

Run with:
    uv run python main.py
"""

import yaml

from src.training.data_loader import load_raw_data
from src.training.feature_engineering import run_feature_engineering
from src.training.preprocessor import run_preprocessing
from src.training.train import run_training


def load_config(path: str = "conf/config.yaml") -> dict:
    """Load YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    print("🚢 Starting Titanic MLOps Pipeline...")
    config = load_config()

    # Step 1: Load data
    df = load_raw_data(config["paths"]["raw_data"])

    # Step 2: Feature engineering (must run before preprocessing)
    df = run_feature_engineering(df)

    # Step 3: Preprocessing (encode, drop, fill)
    df, _ = run_preprocessing(df, config)

    # Step 4: Train and save models
    run_training(df, config)

    print("🎉 Pipeline complete! Models saved in models/")


if __name__ == "__main__":
    main()
