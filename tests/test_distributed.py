"""Tests for distributed utilities."""

import torch
from disttune.core.distributed import DistributedContext


def test_default_context():
    ctx = DistributedContext()
    assert ctx.rank == 0
    assert ctx.world_size == 1
    assert ctx.is_main is True
    assert ctx.is_distributed is False


def test_multi_gpu_context():
    ctx = DistributedContext(rank=1, local_rank=1, world_size=4)
    assert ctx.is_main is False
    assert ctx.is_distributed is True
