"""Pre-defined hyperparameter search space definitions."""

from __future__ import annotations
from typing import Any

BERT_FINETUNE_SPACE: dict[str, dict[str, Any]] = {
    "learning_rate": {"type": "loguniform", "low": 1e-6, "high": 1e-3},
    "weight_decay": {"type": "uniform", "low": 0.0, "high": 0.3},
    "warmup_ratio": {"type": "uniform", "low": 0.0, "high": 0.2},
    "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
}

LORA_SPACE: dict[str, dict[str, Any]] = {
    "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 5e-3},
    "lora_rank": {"type": "categorical", "choices": [4, 8, 16, 32, 64]},
    "lora_alpha": {"type": "categorical", "choices": [16, 32, 64, 128]},
    "weight_decay": {"type": "uniform", "low": 0.0, "high": 0.1},
}
