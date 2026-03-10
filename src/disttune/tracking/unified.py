"""Unified experiment tracker — logs to multiple backends simultaneously."""

from __future__ import annotations
from typing import Any

from disttune.tracking.base import BaseTracker
from disttune.tracking.mlflow_tracker import MLflowTracker
from disttune.tracking.wandb_tracker import WandbTracker
from disttune.utils.logging import get_logger

logger = get_logger(__name__)


class UnifiedTracker(BaseTracker):
    """Fan-out tracker that logs to multiple backends."""

    def __init__(self, backends: list[BaseTracker] | None = None) -> None:
        self.backends: list[BaseTracker] = backends or []

    def add_backend(self, backend: BaseTracker) -> None:
        self.backends.append(backend)

    def log_params(self, params: dict[str, Any]) -> None:
        for b in self.backends:
            try:
                b.log_params(params)
            except Exception as e:
                logger.warning(f"Tracker {b.__class__.__name__} failed log_params: {e}")

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        for b in self.backends:
            try:
                b.log_metrics(metrics, step=step)
            except Exception as e:
                logger.warning(f"Tracker {b.__class__.__name__} failed log_metrics: {e}")

    def log_artifact(self, path: str) -> None:
        for b in self.backends:
            try:
                b.log_artifact(path)
            except Exception as e:
                logger.warning(f"Tracker {b.__class__.__name__} failed log_artifact: {e}")

    def finish(self) -> None:
        for b in self.backends:
            try:
                b.finish()
            except Exception as e:
                logger.warning(f"Tracker {b.__class__.__name__} failed finish: {e}")


def create_tracker(config: dict[str, Any]) -> UnifiedTracker:
    """Factory to build a UnifiedTracker from config."""
    tracker = UnifiedTracker()
    backends = config.get("backends", [])

    if "mlflow" in backends:
        tracker.add_backend(MLflowTracker(experiment_name=config.get("project", "default"), tracking_uri=config.get("mlflow_uri")))
    if "wandb" in backends:
        tracker.add_backend(WandbTracker(project=config.get("project", "default"), tags=config.get("tags", [])))
    if not tracker.backends:
        logger.warning("No tracking backends configured — metrics will not be persisted")

    return tracker
