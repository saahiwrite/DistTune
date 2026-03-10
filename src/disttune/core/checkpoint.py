"""Checkpoint save/load/resume logic with top-K management."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from disttune.utils.logging import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """Manages model checkpoints with automatic pruning of old saves."""

    def __init__(self, output_dir: Path, keep_top_k: int = 3) -> None:
        self.output_dir = output_dir
        self.keep_top_k = keep_top_k
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._history: list[dict[str, Any]] = []

    def save(self, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any, step: int, metrics: dict[str, float]) -> Path:
        ckpt_dir = self.output_dir / f"step-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), ckpt_dir / "model.pt")
        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")

        meta = {"step": step, "metrics": metrics, "path": str(ckpt_dir)}
        with open(ckpt_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        self._history.append(meta)
        self._prune_old()

        latest = self.output_dir / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(ckpt_dir.name)

        logger.info(f"Checkpoint saved: {ckpt_dir}")
        return ckpt_dir

    def load(self, path: str, model: nn.Module, optimizer: torch.optim.Optimizer | None = None, scheduler: Any | None = None) -> dict[str, Any]:
        ckpt_dir = Path(path)
        model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=True))
        if optimizer and (ckpt_dir / "optimizer.pt").exists():
            optimizer.load_state_dict(torch.load(ckpt_dir / "optimizer.pt", weights_only=True))
        if scheduler and (ckpt_dir / "scheduler.pt").exists():
            scheduler.load_state_dict(torch.load(ckpt_dir / "scheduler.pt", weights_only=True))

        meta = {}
        if (ckpt_dir / "meta.json").exists():
            with open(ckpt_dir / "meta.json") as f:
                meta = json.load(f)

        logger.info(f"Checkpoint loaded: {ckpt_dir}")
        return meta

    def _prune_old(self) -> None:
        if len(self._history) <= self.keep_top_k:
            return
        to_remove = self._history[: -self.keep_top_k]
        self._history = self._history[-self.keep_top_k :]
        for entry in to_remove:
            p = Path(entry["path"])
            if p.exists():
                shutil.rmtree(p)
                logger.info(f"Pruned old checkpoint: {p}")
