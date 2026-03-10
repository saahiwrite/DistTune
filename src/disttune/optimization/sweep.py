"""Optuna-based hyperparameter sweep orchestration."""

from __future__ import annotations
from typing import Any, Callable

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from disttune.utils.logging import get_logger

logger = get_logger(__name__)

SearchSpace = dict[str, dict[str, Any]]


def build_search_space(trial: optuna.Trial, space: SearchSpace) -> dict[str, Any]:
    """Sample hyperparameters from an Optuna trial given a search space definition."""
    params: dict[str, Any] = {}
    for name, spec in space.items():
        match spec["type"]:
            case "uniform":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"])
            case "loguniform":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
            case "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])
            case "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            case _:
                raise ValueError(f"Unknown search space type: {spec['type']}")
    return params


class SweepRunner:
    """Orchestrates Optuna hyperparameter sweeps."""

    def __init__(self, study_name: str, search_space: SearchSpace, objective_fn: Callable[[dict[str, Any]], float], direction: str = "maximize", n_trials: int = 50, pruner_startup_trials: int = 5, storage: str | None = None) -> None:
        self.search_space = search_space
        self.objective_fn = objective_fn
        self.n_trials = n_trials

        pruner = MedianPruner(n_startup_trials=pruner_startup_trials)
        sampler = TPESampler(seed=42)

        self.study = optuna.create_study(study_name=study_name, direction=direction, sampler=sampler, pruner=pruner, storage=storage, load_if_exists=True)
        logger.info(f"Created Optuna study '{study_name}' ({direction}, {n_trials} trials)")

    def _objective(self, trial: optuna.Trial) -> float:
        params = build_search_space(trial, self.search_space)
        logger.info(f"Trial {trial.number}: {params}")
        try:
            metric = self.objective_fn(params)
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()
        return metric

    def run(self) -> optuna.Study:
        """Execute the sweep and return the completed study."""
        logger.info(f"Starting sweep: {self.n_trials} trials")
        self.study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=True)
        best = self.study.best_trial
        logger.info(f"Best trial #{best.number}: value={best.value:.6f}, params={best.params}")
        return self.study

    def get_results_df(self):
        return self.study.trials_dataframe()


def load_sweep_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "study_name": config["study_name"],
        "direction": config.get("direction", "maximize"),
        "n_trials": config.get("n_trials", 50),
        "metric": config["metric"],
        "pruner_startup_trials": config.get("pruner", {}).get("n_startup_trials", 5),
        "search_space": config["search_space"],
    }
