# src/utils/distributed_utils.py (FINAL STABLE VERSION WITH TIMEOUT FIX)

import os
import torch
import torch.distributed as dist
from typing import Any
from datetime import timedelta


def setup_distributed(rank: int, world_size: int):
    """
    [ROBUST] Initializes the distributed environment.
    - Sets a very long timeout to accommodate slow operations like model saving.
    """
    # torchrun or your launch script should set these.
    # We can add defaults for local testing.
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12355')

    # 【核心修复】将超时时间从默认的10分钟大幅延长到60分钟。
    # 这为FSDP从所有GPU收集参数到主进程提供了充足的时间，防止因网络或IO延迟导致超时。
    timeout = timedelta(minutes=60)

    try:
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=timeout
        )
        torch.cuda.set_device(rank)
        if rank == 0:
            print(f"✅ Distributed environment initialized with {world_size} GPUs (NCCL backend).")
            print(f"   Timeout set to {timeout.total_seconds() / 60} minutes.")
        print(f"   [Rank {rank}] Process ready on device cuda:{rank}.")
    except Exception as e:
        print(f"❌ [Rank {rank}] Failed to initialize process group: {e}")
        raise


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()
        if get_rank() == 0:
            print("✅ Distributed environment cleaned up.")


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def broadcast_object(obj: Any, src_rank: int = 0) -> Any:
    if get_world_size() > 1:
        object_list = [obj] if get_rank() == src_rank else [None]
        dist.broadcast_object_list(object_list, src=src_rank)
        return object_list[0]
    return obj