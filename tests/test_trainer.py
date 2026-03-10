"""Tests for the core trainer module."""

import pytest
import torch
import torch.nn as nn
from disttune.core.trainer import TrainerConfig


def test_trainer_config_defaults():
    cfg = TrainerConfig()
    assert cfg.epochs == 10
    assert cfg.learning_rate == 2e-5
    assert cfg.mixed_precision == "fp16"
    assert cfg.gradient_accumulation_steps == 1
    assert cfg.greater_is_better is False


def test_trainer_config_custom():
    cfg = TrainerConfig(epochs=3, learning_rate=1e-4, mixed_precision="bf16")
    assert cfg.epochs == 3
    assert cfg.learning_rate == 1e-4
    assert cfg.mixed_precision == "bf16"
