# src/utils/distributed_utils.py

import os
import torch
import torch.distributed as dist
from typing import Any

def setup_distributed(rank: int, world_size: int, port: str = "12355"):
    """
    初始化PyTorch分布式环境 (DDP/FSDP)。
    每个工作进程都会调用此函数来加入进程组。

    Args:
        rank (int): 当前进程的全局排名。
        world_size (int): 全局总进程数。
        port (str, optional): 用于通信的主节点端口。Defaults to "12355"。
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    try:
        # 使用NCCL后端
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        if rank == 0:
            print(f"✅ Distributed environment initialized with {world_size} GPUs.")
        print(f"   [Rank {rank}] Process ready on device cuda:{rank}.")
    except Exception as e:
        print(f"❌ [Rank {rank}] Failed to initialize process group: {e}")
        raise

def cleanup_distributed():
    """清理分布式环境，释放所有进程组资源。"""
    if dist.is_initialized():
        dist.destroy_process_group()
        if get_rank() == 0:
            print("✅ Distributed environment cleaned up.")

def get_rank() -> int:
    """获取当前进程的全局排名。如果未初始化，则返回0。"""
    return dist.get_rank() if dist.is_initialized() else 0

def get_world_size() -> int:
    """获取分布式环境中的总进程数。如果未初始化，则返回1。"""
    return dist.get_world_size() if dist.is_initialized() else 1

def is_main_process() -> bool:
    """判断当前进程是否为主进程 (rank 0)，通常用于控制日志打印和文件写入。"""
    return get_rank() == 0

def broadcast_object(obj: Any, src_rank: int = 0) -> Any:
    """
    将一个Python对象从源进程广播到所有其他进程。
    这对于从主进程分发配置或采样数据非常有用。

    Args:
        obj (Any): 需要广播的对象（必须是可pickle的）。
        src_rank (int, optional): 源进程的排名。Defaults to 0。

    Returns:
        Any: 所有进程收到的对象。
    """
    if get_world_size() > 1:
        # 在非源进程上，obj_list中的对象是None
        object_list = [obj] if is_main_process() else [None]
        # broadcast_object_list会用src_rank的对象填充所有进程的列表
        dist.broadcast_object_list(object_list, src=src_rank)
        return object_list[0]
    return obj