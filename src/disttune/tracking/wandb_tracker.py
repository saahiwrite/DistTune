"""Weights & Biases experiment tracking backend."""

from __future__ import annotations
from typing import Any

from disttune.tracking.base import BaseTracker
from disttune.utils.logging import get_logger

logger = get_logger(__name__)


class WandbTracker(BaseTracker):
    def __init__(self, project: str, name: str | None = None, tags: list[str] | None = None) -> None:
        import wandb
        self.run = wandb.init(project=project, name=name, tags=tags or [])
        self._wandb = wandb
        logger.info(f"W&B run started: {self.run.id}")

    def log_params(self, params: dict[str, Any]) -> None:
        self._wandb.config.update(params)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self._wandb.log(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        artifact = self._wandb.Artifact(name="model", type="model")
        artifact.add_file(path)
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        self._wandb.finish()
