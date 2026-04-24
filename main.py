"""Main entry point for the Titanic MLOps training pipeline.

Run with:
    uv run python main.py
"""

import hydra
from omegaconf import DictConfig

from src.training.data_loader import load_data
from src.training.feature_engineering import engineer_features
from src.training.preprocessor import build_preprocessor
from src.training.train import train_model


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    Main pipeline entry point.
    cfg is automatically populated by Hydra from conf/config.yaml
    """
    print("🚀 Starting Titanic MLOps Pipeline...")
    print(f"   Model: {cfg.model.name}")
    print(f"   CV folds: {cfg.pipeline.cv_folds}")

    # Step 1: Load raw data
    df = load_data(cfg)

    # Step 2: Engineer features
    df = engineer_features(df, cfg)

    # Step 3: Build preprocessor
    preprocessor = build_preprocessor(cfg)

    # Step 4: Train and save model
    train_model(df, preprocessor, cfg)

    print("🎉 Pipeline complete!")


if __name__ == "__main__":
    main()
