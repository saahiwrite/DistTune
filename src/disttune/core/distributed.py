"""Multi-GPU distributed training setup and utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from disttune.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DistributedContext:
    """Holds distributed training state."""
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str = "nccl"

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self.local_rank}")
        return torch.device("cpu")

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


def init_distributed() -> DistributedContext:
    """Initialize distributed training from environment (torchrun sets these)."""
    if "RANK" not in os.environ:
        logger.info("No distributed env detected — running single-process")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return DistributedContext()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    ctx = DistributedContext(rank=rank, local_rank=local_rank, world_size=world_size, backend=backend)
    logger.info(f"Initialized rank {rank}/{world_size} on cuda:{local_rank}")
    return ctx


def cleanup_distributed() -> None:
    """Tear down the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average a tensor across all processes."""
    if not dist.is_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def barrier() -> None:
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()
