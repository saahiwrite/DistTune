"""Abstract tracker interface."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class BaseTracker(ABC):
    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None: ...

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int) -> None: ...

    @abstractmethod
    def log_artifact(self, path: str) -> None: ...

    @abstractmethod
    def finish(self) -> None: ...
