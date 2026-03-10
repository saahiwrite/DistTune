"""Tests for the unified tracking system."""

from disttune.tracking.unified import UnifiedTracker
from disttune.tracking.base import BaseTracker
from typing import Any


class MockTracker(BaseTracker):
    def __init__(self):
        self.params = {}
        self.metrics = []
        self.artifacts = []
        self.finished = False

    def log_params(self, params: dict[str, Any]) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.metrics.append((metrics, step))

    def log_artifact(self, path: str) -> None:
        self.artifacts.append(path)

    def finish(self) -> None:
        self.finished = True


def test_unified_tracker_fanout():
    m1 = MockTracker()
    m2 = MockTracker()
    tracker = UnifiedTracker(backends=[m1, m2])

    tracker.log_params({"lr": 1e-4})
    tracker.log_metrics({"loss": 0.5}, step=1)
    tracker.log_artifact("/path/to/model")
    tracker.finish()

    for m in [m1, m2]:
        assert m.params == {"lr": 1e-4}
        assert len(m.metrics) == 1
        assert m.artifacts == ["/path/to/model"]
        assert m.finished is True
