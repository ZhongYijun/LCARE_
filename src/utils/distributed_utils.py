# src/utils/distributed_utils.py

import os
import torch
import torch.distributed as dist
from typing import Any

def setup_distributed(rank: int, world_size: int):
    """
    初始化PyTorch分布式环境 (DDP/FSDP)。
    此函数设计为与`torchrun`启动器配合使用。`torchrun`会预先在每个进程中
    设置好`MASTER_ADDR`和`MASTER_PORT`等必要的环境变量。

    Args:
        rank (int): 当前进程的全局排名（由torchrun在os.environ['RANK']中提供）。
        world_size (int): 全局总进程数（由torchrun在os.environ['WORLD_SIZE']中提供）。
    """
    # 检查环境变量是否存在，提供更友好的错误信息
    if 'MASTER_ADDR' not in os.environ or 'MASTER_PORT' not in os.environ:
        raise ConnectionError("MASTER_ADDR and MASTER_PORT environment variables are not set. "
                              "This script is designed to be launched with `torchrun`.")
    
    try:
        # 使用NCCL后端，这是NVIDIA GPU之间通信最高效的选择
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        
        # 将当前进程绑定到指定的GPU设备
        torch.cuda.set_device(rank)
        
        # 仅在主进程打印成功信息，避免日志混乱
        if rank == 0:
            print(f"✅ Distributed environment initialized with {world_size} GPUs (NCCL backend).")
            
        # 每个进程打印自己的状态，便于调试
        print(f"   [Rank {rank}/{world_size}] Process ready, bound to device cuda:{rank}.")

    except Exception as e:
        print(f"❌ [Rank {rank}] Failed to initialize process group: {e}")
        # 抛出异常以终止程序，避免进程卡死
        raise

def cleanup_distributed():
    """清理分布式环境，释放所有进程组资源。"""
    if dist.is_initialized():
        dist.destroy_process_group()
        if get_rank() == 0:
            print("✅ Distributed environment cleaned up.")

def get_rank() -> int:
    """获取当前进程的全局排名。如果分布式环境未初始化，则返回0。"""
    return dist.get_rank() if dist.is_initialized() else 0

def get_world_size() -> int:
    """获取分布式环境中的总进程数。如果未初始化，则返回1。"""
    return dist.get_world_size() if dist.is_initialized() else 1

def is_main_process() -> bool:
    """
    判断当前进程是否为主进程 (rank 0)。
    主进程通常负责日志记录、模型保存和数据加载等任务。
    """
    return get_rank() == 0

def broadcast_object(obj: Any, src_rank: int = 0) -> Any:
    """
    将一个Python对象从源进程广播到所有其他进程。
    这对于从主进程分发配置、采样数据或词表等非常有用。

    Args:
        obj (Any): 需要广播的对象（必须是可pickle序列化的）。
        src_rank (int, optional): 源进程的排名。默认为 0。

    Returns:
        Any: 所有进程收到的对象副本。
    """
    # 仅在多GPU环境下执行广播操作
    if get_world_size() > 1:
        # 将对象放入一个列表中，因为 broadcast_object_list 需要一个列表参数
        # 在非源进程上，obj_list中的对象是None
        object_list = [obj] if get_rank() == src_rank else [None]
        
        # 执行广播。PyTorch会用src_rank进程的对象填充所有其他进程的列表
        dist.broadcast_object_list(object_list, src=src_rank)
        
        # 返回列表中的第一个（也是唯一一个）元素
        return object_list[0]
    
    # 如果是单GPU环境，直接返回原始对象
    return obj

# # src/utils/distributed_utils.py (FINAL STABLE VERSION)

# import os
# import torch
# import torch.distributed as dist
# from typing import Any
# from datetime import timedelta


# def setup_distributed(rank: int, world_size: int):
#     """
#     [ROBUST] Initializes the distributed environment.
#     - Explicitly uses the NCCL backend for NVIDIA GPUs.
#     - Sets a very long timeout to accommodate slow operations.
#     """
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355' # A default port

#     # Define a long timeout for operations like `_sync_local_actor`
#     timeout = timedelta(minutes=60)

#     try:
#         # [CRITICAL FIX] Explicitly set backend to "nccl" and increase timeout
#         dist.init_process_group(
#             backend="nccl", 
#             rank=rank, 
#             world_size=world_size,
#             timeout=timeout
#         )
#         torch.cuda.set_device(rank)
#         if rank == 0:
#             print(f"✅ Distributed environment initialized with {world_size} GPUs (NCCL backend).")
#             print(f"   Timeout set to {timeout.total_seconds() / 60} minutes.")
#         print(f"   [Rank {rank}] Process ready on device cuda:{rank}.")
#     except Exception as e:
#         print(f"❌ [Rank {rank}] Failed to initialize process group: {e}")
#         raise


# def cleanup_distributed():
#     if dist.is_initialized():
#         dist.destroy_process_group()
#         if get_rank() == 0:
#             print("✅ Distributed environment cleaned up.")


# def get_rank() -> int:
#     return dist.get_rank() if dist.is_initialized() else 0


# def get_world_size() -> int:
#     return dist.get_world_size() if dist.is_initialized() else 1


# def is_main_process() -> bool:
#     return get_rank() == 0


# def broadcast_object(obj: Any, src_rank: int = 0) -> Any:
#     # This function remains a potential bottleneck for very large objects
#     # but is necessary for the `local_actor` architecture.
#     if get_world_size() > 1:
#         object_list = [obj] if get_rank() == src_rank else [None]
#         dist.broadcast_object_list(object_list, src=src_rank)
#         return object_list[0]
#     return obj