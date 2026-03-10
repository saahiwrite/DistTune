"""Distributed data loading utilities."""

from __future__ import annotations
from typing import Any

from torch.utils.data import DataLoader, DistributedSampler, Dataset

from disttune.core.distributed import DistributedContext


def create_distributed_loader(dataset: Dataset, batch_size: int, dist_ctx: DistributedContext, shuffle: bool = True, num_workers: int = 4, **kwargs: Any) -> DataLoader:
    sampler = None
    if dist_ctx.is_distributed:
        sampler = DistributedSampler(dataset, num_replicas=dist_ctx.world_size, rank=dist_ctx.rank, shuffle=shuffle)
        shuffle = False

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle, num_workers=num_workers, pin_memory=True, **kwargs)
