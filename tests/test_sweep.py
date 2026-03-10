"""Tests for the Optuna sweep module."""

from disttune.optimization.sweep import load_sweep_config


def test_load_sweep_config():
    cfg = {
        "study_name": "test-sweep",
        "direction": "maximize",
        "metric": "eval/f1",
        "n_trials": 10,
        "pruner": {"n_startup_trials": 3},
        "search_space": {
            "learning_rate": {"type": "loguniform", "low": 1e-6, "high": 1e-3},
        },
    }
    result = load_sweep_config(cfg)
    assert result["study_name"] == "test-sweep"
    assert result["n_trials"] == 10
    assert "learning_rate" in result["search_space"]
