"""Model registry and factory."""

from __future__ import annotations
from typing import Any

from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM


MODEL_REGISTRY: dict[str, Any] = {
    "sequence_classification": AutoModelForSequenceClassification,
    "causal_lm": AutoModelForCausalLM,
}


def create_model(name: str, task: str = "sequence_classification", **kwargs: Any):
    cls = MODEL_REGISTRY.get(task)
    if cls is None:
        raise ValueError(f"Unknown task: {task}. Available: {list(MODEL_REGISTRY.keys())}")
    return cls.from_pretrained(name, **kwargs)
