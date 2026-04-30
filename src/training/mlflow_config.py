"""MLflow configuration for DagsHub tracking."""

import os
import mlflow


def setup_mlflow(cfg) -> None:
    """Configure MLflow to log to DagsHub."""

    # DagsHub MLflow tracking URI
    tracking_uri = (
        f"https://dagshub.com/{cfg.mlflow.dagshub_username}"
        f"/{cfg.mlflow.repo_name}.mlflow"
    )

    # Set credentials from environment variables
    os.environ["MLFLOW_TRACKING_USERNAME"] = cfg.mlflow.dagshub_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ.get(
        "DAGSHUB_TOKEN", ""
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    print(f" MLflow tracking → {tracking_uri}")
    print(f" Experiment: {cfg.mlflow.experiment_name}")