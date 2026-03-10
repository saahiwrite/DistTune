"""Single-run training entry point. Supports DDP via torchrun."""

from __future__ import annotations

import typer
import yaml
import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler

from disttune.core.distributed import init_distributed, cleanup_distributed
from disttune.core.trainer import Trainer, TrainerConfig
from disttune.tracking.unified import create_tracker
from disttune.utils.config import load_config, merge_configs
from disttune.utils.reproducibility import seed_everything
from disttune.utils.logging import get_logger

logger = get_logger(__name__)
app = typer.Typer()


def build_dataloaders(config, tokenizer, dist_ctx):
    ds = load_dataset(config["data"]["dataset"])

    def tokenize(batch):
        return tokenizer(
            batch["sentence1"] if "sentence1" in batch else batch["text"],
            batch.get("sentence2"),
            truncation=True,
            padding="max_length",
            max_length=config["data"].get("max_length", 512),
        )

    ds = ds.map(tokenize, batched=True, remove_columns=ds["train"].column_names - {"label"})
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch")

    bs = config["data"].get("batch_size", 32)
    train_sampler = DistributedSampler(ds["train"], num_replicas=dist_ctx.world_size, rank=dist_ctx.rank, shuffle=True) if dist_ctx.is_distributed else None
    eval_sampler = DistributedSampler(ds["validation"], num_replicas=dist_ctx.world_size, rank=dist_ctx.rank, shuffle=False) if dist_ctx.is_distributed else None

    train_loader = DataLoader(ds["train"], batch_size=bs, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=4, pin_memory=True)
    eval_loader = DataLoader(ds["validation"], batch_size=bs, sampler=eval_sampler, num_workers=4, pin_memory=True)
    return train_loader, eval_loader


@app.command()
def main(
    config: Path = typer.Option(..., "--config", "-c", help="Path to experiment YAML config"),
    overrides: list[str] = typer.Option([], "--set", "-s", help="Override config values (dot notation)"),
) -> None:
    cfg = load_config(config)
    if overrides:
        cfg = merge_configs(cfg, overrides)

    seed_everything(cfg.get("seed", 42))
    dist_ctx = init_distributed()

    try:
        model_name = cfg["model"]["name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        train_loader, eval_loader = build_dataloaders(cfg, tokenizer, dist_ctx)

        tcfg = cfg["training"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg.get("weight_decay", 0.01))
        total_steps = len(train_loader) * tcfg["epochs"]
        warmup_steps = int(total_steps * tcfg.get("warmup_ratio", 0.1))
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        tracker = create_tracker(cfg.get("tracking", {}))

        trainer_config = TrainerConfig(
            epochs=tcfg["epochs"],
            learning_rate=tcfg["learning_rate"],
            weight_decay=tcfg.get("weight_decay", 0.01),
            mixed_precision=tcfg.get("mixed_precision", "fp16"),
            gradient_accumulation_steps=tcfg.get("gradient_accumulation_steps", 1),
            output_dir=cfg.get("output_dir", "./outputs"),
            metric_for_best_model=tcfg.get("metric_for_best_model", "eval/loss"),
        )

        trainer = Trainer(model=model, train_loader=train_loader, eval_loader=eval_loader, optimizer=optimizer, scheduler=scheduler, config=trainer_config, tracker=tracker, dist_ctx=dist_ctx)
        metrics = trainer.train()
        tracker.finish()

        if dist_ctx.is_main:
            logger.info(f"Training complete. Final metrics: {metrics}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    app()
