"""Optuna sweep entry point — runs N trials with hyperparameter search."""

from __future__ import annotations

import typer
import subprocess
import sys
import tempfile
import yaml
from pathlib import Path
from typing import Any

from disttune.optimization.sweep import SweepRunner, load_sweep_config
from disttune.utils.config import load_config
from disttune.utils.logging import get_logger

logger = get_logger(__name__)
app = typer.Typer()


def run_trial(base_config: dict[str, Any], params: dict[str, Any]) -> float:
    trial_cfg = {**base_config}
    trial_cfg.setdefault("training", {})
    trial_cfg.setdefault("model", {})

    param_mapping = {
        "learning_rate": ("training", "learning_rate"),
        "weight_decay": ("training", "weight_decay"),
        "batch_size": ("data", "batch_size"),
        "warmup_ratio": ("training", "warmup_ratio"),
        "lora_rank": ("model", "lora_rank"),
    }

    for k, v in params.items():
        if k in param_mapping:
            section, key = param_mapping[k]
            trial_cfg.setdefault(section, {})[key] = v

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(trial_cfg, f)
        tmp_path = f.name

    gpus = trial_cfg.get("distributed", {}).get("gpus", 1)
    if gpus > 1:
        cmd = ["torchrun", f"--nproc_per_node={gpus}", "scripts/train.py", "--config", tmp_path]
    else:
        cmd = [sys.executable, "scripts/train.py", "--config", tmp_path]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Trial failed:\n{result.stderr}")
        raise RuntimeError("Training subprocess failed")

    import json
    for line in reversed(result.stdout.strip().split("\n")):
        try:
            metrics = json.loads(line)
            metric_name = trial_cfg.get("sweep", {}).get("metric", "eval/loss")
            return metrics[metric_name]
        except (json.JSONDecodeError, KeyError):
            continue

    raise RuntimeError("Could not parse metric from training output")


@app.command()
def main(
    config: Path = typer.Option(..., "--config", "-c", help="Path to sweep YAML config"),
    n_trials: int = typer.Option(50, "--n-trials", "-n", help="Number of trials"),
    base_config: Path = typer.Option(None, "--base-config", "-b", help="Base experiment config"),
    storage: str = typer.Option(None, "--storage", help="Optuna storage URL"),
) -> None:
    sweep_cfg = load_config(config)
    sweep_params = load_sweep_config(sweep_cfg)
    if n_trials:
        sweep_params["n_trials"] = n_trials

    base_cfg = load_config(base_config) if base_config else {}

    def objective(params):
        return run_trial(base_cfg, params)

    runner = SweepRunner(
        study_name=sweep_params["study_name"],
        search_space=sweep_params["search_space"],
        objective_fn=objective,
        direction=sweep_params["direction"],
        n_trials=sweep_params["n_trials"],
        pruner_startup_trials=sweep_params["pruner_startup_trials"],
        storage=storage,
    )

    study = runner.run()
    df = runner.get_results_df()
    df.to_csv("sweep_results.csv", index=False)
    logger.info(f"Sweep complete. Results saved to sweep_results.csv")
    logger.info(f"Best trial: #{study.best_trial.number} — {study.best_value:.6f}")
    logger.info(f"Best params: {study.best_params}")


if __name__ == "__main__":
    app()
