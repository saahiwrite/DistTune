"""Core distributed training loop with DDP/FSDP support."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from disttune.core.checkpoint import CheckpointManager
from disttune.core.distributed import DistributedContext
from disttune.tracking.unified import UnifiedTracker
from disttune.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainerConfig:
    """Training configuration."""
    epochs: int = 10
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    mixed_precision: str = "fp16"
    gradient_accumulation_steps: int = 1
    eval_steps: int | None = None
    save_steps: int | None = None
    warmup_ratio: float = 0.1
    output_dir: str = "./outputs"
    resume_from: str | None = None
    metric_for_best_model: str = "eval/loss"
    greater_is_better: bool = False


class Trainer:
    """Distributed trainer supporting DDP and FSDP."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        eval_loader: DataLoader | None,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        config: TrainerConfig,
        tracker: UnifiedTracker,
        dist_ctx: DistributedContext,
    ) -> None:
        self.config = config
        self.tracker = tracker
        self.dist_ctx = dist_ctx
        self.device = dist_ctx.device
        self.global_step = 0
        self.best_metric = float("inf") if not config.greater_is_better else float("-inf")

        self.model = self._wrap_model(model)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.use_amp = config.mixed_precision != "none"
        self.amp_dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16
        self.scaler = GradScaler(enabled=self.use_amp and config.mixed_precision == "fp16")

        self.ckpt_manager = CheckpointManager(
            output_dir=Path(config.output_dir) / "checkpoints",
            keep_top_k=3,
        )

        if config.resume_from:
            self._resume(config.resume_from)

    def _wrap_model(self, model: nn.Module) -> nn.Module:
        model = model.to(self.device)
        if self.dist_ctx.world_size > 1:
            model = DDP(model, device_ids=[self.dist_ctx.local_rank])
            logger.info(f"Wrapped model with DDP on rank {self.dist_ctx.rank}")
        return model

    def train(self) -> dict[str, float]:
        """Run full training loop. Returns final metrics."""
        logger.info(f"Starting training for {self.config.epochs} epochs")
        self.tracker.log_params(self._get_hparams())

        total_steps = len(self.train_loader) * self.config.epochs
        self.tracker.log_params({"total_steps": total_steps})

        for epoch in range(self.config.epochs):
            train_metrics = self._train_epoch(epoch)
            eval_metrics = self._evaluate() if self.eval_loader else {}

            metrics = {**train_metrics, **eval_metrics, "epoch": epoch}
            self.tracker.log_metrics(metrics, step=self.global_step)

            if self.dist_ctx.is_main:
                self._maybe_save_checkpoint(metrics)
                logger.info(f"Epoch {epoch}: {metrics}")

        return metrics

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()

        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        n_batches = 0
        t0 = time.perf_counter()

        for step, batch in enumerate(self.train_loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            n_batches += 1

            if self.config.eval_steps and self.global_step % self.config.eval_steps == 0:
                eval_metrics = self._evaluate() if self.eval_loader else {}
                self.tracker.log_metrics(eval_metrics, step=self.global_step)

        elapsed = time.perf_counter() - t0
        avg_loss = total_loss / max(n_batches, 1)
        throughput = n_batches * self.train_loader.batch_size / elapsed

        return {
            "train/loss": avg_loss,
            "train/lr": self.scheduler.get_last_lr()[0],
            "train/epoch_time_s": elapsed,
            "train/throughput_samples_s": throughput,
        }

    @torch.no_grad()
    def _evaluate(self) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.eval_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(**batch)
            total_loss += outputs.loss.item()
            n_batches += 1

        self.model.train()
        return {"eval/loss": total_loss / max(n_batches, 1)}

    def _maybe_save_checkpoint(self, metrics: dict[str, float]) -> None:
        metric_val = metrics.get(self.config.metric_for_best_model)
        if metric_val is None:
            return

        improved = (
            metric_val > self.best_metric
            if self.config.greater_is_better
            else metric_val < self.best_metric
        )

        if improved:
            self.best_metric = metric_val
            unwrapped = self.model.module if isinstance(self.model, DDP) else self.model
            self.ckpt_manager.save(
                model=unwrapped,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                step=self.global_step,
                metrics=metrics,
            )
            logger.info(f"New best model saved (metric={metric_val:.6f})")

    def _resume(self, path: str) -> None:
        state = self.ckpt_manager.load(path, self.model, self.optimizer, self.scheduler)
        self.global_step = state.get("step", 0)
        self.best_metric = state.get("best_metric", self.best_metric)
        logger.info(f"Resumed from {path} at step {self.global_step}")

    def _get_hparams(self) -> dict[str, Any]:
        return {
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "mixed_precision": self.config.mixed_precision,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "world_size": self.dist_ctx.world_size,
        }
