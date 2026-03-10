"""MLflow experiment tracking backend."""

from __future__ import annotations
from typing import Any

from disttune.tracking.base import BaseTracker
from disttune.utils.logging import get_logger

logger = get_logger(__name__)


class MLflowTracker(BaseTracker):
    def __init__(self, experiment_name: str, run_name: str | None = None, tracking_uri: str | None = None) -> None:
        import mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run(run_name=run_name)
        self._mlflow = mlflow
        logger.info(f"MLflow run started: {self.run.info.run_id}")

    def log_params(self, params: dict[str, Any]) -> None:
        self._mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        self._mlflow.log_artifact(path)

    def finish(self) -> None:
        self._mlflow.end_run()
