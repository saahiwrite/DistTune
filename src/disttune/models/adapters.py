"""LoRA and adapter wrappers using PEFT."""

from __future__ import annotations
from typing import Any

import torch.nn as nn


def apply_lora(model: nn.Module, rank: int = 16, alpha: int = 32, target_modules: list[str] | None = None, **kwargs: Any) -> nn.Module:
    from peft import LoraConfig, get_peft_model, TaskType

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules or ["q_proj", "v_proj"],
        lora_dropout=kwargs.get("dropout", 0.05),
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model
