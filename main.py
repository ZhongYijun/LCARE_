# # main.py (修改后)

# import hydra
# from omegaconf import DictConfig, OmegaConf
# import torch
# # [删除] 不再需要 torch.multiprocessing

# import sys
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).resolve().parent))

# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# from src.data_processing.build_math_datasets import build_datasets
# from src.data_processing.build_enhancement_data import create_enhancement_data
# from evaluate import run_evaluation
# from src.utils.distributed_utils import setup_distributed, cleanup_distributed, is_main_process, get_rank


# def run_distributed_task(config: DictConfig):
#     """
#     [新] 用于分布式任务的执行函数。
#     它由每个 torchrun 启动的进程直接调用。
#     """
#     try:
#         # 从环境变量获取rank和world_size，这是torchrun的标准做法
#         rank = int(os.environ["RANK"])
#         world_size = int(os.environ["WORLD_SIZE"])
        
#         setup_distributed(rank, world_size)

#         # 使用Hydra的instantiate来动态创建训练器
#         trainer = hydra.utils.instantiate(config.trainer, config=config, rank=rank, world_size=world_size)

#         task = config.main.task
#         if task == "train_sft":
#             trainer.train()
#         elif task == "train_rl":
#             trainer.learn()

#     except Exception as e:
#         print(f"FATAL ERROR in worker process rank {get_rank()}: {e}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         cleanup_distributed()


# @hydra.main(version_base=None, config_path="./configs", config_name="lcare_config")
# def main(config: DictConfig) -> None:
#     """
#     [V-FINAL - TORCHRUN COMPATIBLE] 项目主入口。
#     分布式任务不再使用mp.spawn，而是直接执行。
#     """
#     print("=" * 60)
#     print(" L-CARE Project Main Entry (torchrun-compatible) ")
#     print("=" * 60)
#     if is_main_process(): # 只在主进程打印配置
#         print(" Resolved Configuration:")
#         print(OmegaConf.to_yaml(config))
#         print("-" * 60)

#     task = config.main.task
#     valid_tasks = ["process_data", "train_sft", "train_rl", "evaluate", "create_enhancement_data"]
#     if task not in valid_tasks:
#         raise ValueError(f"Unknown task: '{task}'. Available tasks are: {valid_tasks}")

#     if task == "process_data":
#         print(f"🚀 Starting task: {task}...")
#         build_datasets(config)

#     elif task == "create_enhancement_data":
#         print(f"🚀 Starting task: {task}...")
#         create_enhancement_data(config)

#     elif task in ["train_sft", "train_rl"]:
#         if "RANK" not in os.environ:
#              raise EnvironmentError("RANK env var not found. This script should be launched with `torchrun`.")
#         print(f"🚀 Starting distributed task: {task} on Rank {os.environ['RANK']}...")
#         run_distributed_task(config)

#     elif task == "evaluate":
#         print(f"🚀 Starting task: {task}...")
#         run_evaluation(config)

#     # barrier确保所有进程都完成后再打印成功信息
#     if torch.distributed.is_initialized():
#         torch.distributed.barrier()
        
#     if is_main_process():
#         print(f"\n✅ Task '{task}' finished successfully.")


# if __name__ == "__main__":
#     main()

# main.py (FINAL STABLE & ROBUST)

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import os

# 建议设置镜像源，如果您的环境需要
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from src.data_processing.build_math_datasets import build_datasets
from src.data_processing.build_enhancement_data import create_enhancement_data
from evaluate import run_evaluation
from src.utils.distributed_utils import setup_distributed, cleanup_distributed, is_main_process, get_rank
from hydra.utils import get_class # 关键修复：用于动态加载类
from src.utils.logger import SwanLabLogger


def run_distributed_task(config: DictConfig):
    """
    由每个 torchrun 启动的进程直接调用的分布式任务执行函数。
    """
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # 初始化日志记录器
        logger = SwanLabLogger(config, rank)
        
        # 初始化分布式环境
        setup_distributed(rank, world_size)

        # --- [CRITICAL FIX] 替换 instantiate 以避免参数冲突 ---
        # 1. 从配置中获取目标类的完整路径 (e.g., 'src.trainers.sft_trainer.SFTTrainer')
        target_class_path = config.trainer._target_
        
        # 2. 使用 hydra 的工具函数动态地加载这个类
        TrainerClass = get_class(target_class_path)

        # 3. 直接、明确地实例化这个类，传入它需要的参数
        #    这种方式更清晰，且避免了Hydra自动实例化时可能出现的参数名冲突
        trainer = TrainerClass(config=config, rank=rank, world_size=world_size, logger=logger)
        # --- 修复结束 ---

        task = config.main.task
        if task == "train_sft":
            trainer.train()
        elif task == "train_rl":
            trainer.learn()

    except Exception as e:
        # 在出错的进程上打印详细的错误信息
        print(f"FATAL ERROR in worker process rank {get_rank()}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保在任何情况下都清理分布式环境
        cleanup_distributed()


@hydra.main(version_base=None, config_path="./configs", config_name="lcare_config")
def main(config: DictConfig) -> None:
    """
    [FINAL-TORCHRUN COMPATIBLE] 项目主入口。
    """
    task = config.main.task

    # 只有主进程才打印冗长的配置信息，保持日志清洁
    if is_main_process():
        print("=" * 60)
        print(f" L-CARE Project Main Entry: Running Task -> {task}")
        print("=" * 60)
        # 打印完整的、解析后的配置
        print(" Resolved Configuration:")
        print(OmegaConf.to_yaml(config))
        print("-" * 60)
        
    valid_tasks = ["process_data", "train_sft", "train_rl", "evaluate", "create_enhancement_data"]
    if task not in valid_tasks:
        raise ValueError(f"Unknown task: '{task}'. Available tasks are: {valid_tasks}")

    # --- 任务分派 ---
    if task == "process_data":
        # 数据处理是单进程任务
        if is_main_process():
            print(f"🚀 Starting task: {task}...")
            build_datasets(config)

    elif task == "create_enhancement_data":
        # 创建增强数据也是单进程任务
        if is_main_process():
            print(f"🚀 Starting task: {task}...")
            create_enhancement_data(config)

    elif task in ["train_sft", "train_rl"]:
        # 确保脚本是通过 torchrun 启动的
        if "RANK" not in os.environ:
            raise EnvironmentError("RANK env var not found. This script should be launched with `torchrun` for distributed tasks.")
        
        if is_main_process():
            print(f"🚀 Starting distributed task: {task} on {os.environ['WORLD_SIZE']} GPUs...")
        
        run_distributed_task(config)

    elif task == "evaluate":
        # 评估通常是单卡任务
        if is_main_process():
            print(f"🚀 Starting task: {task}...")
            run_evaluation(config)

    # 确保所有进程都完成后再打印成功信息
    if torch.distributed.is_initialized():
        # 这个 barrier 确保所有进程都完成了它们的工作
        # (特别是对于分布式任务) 才继续执行
        torch.distributed.barrier()

    if is_main_process():
        print(f"\n✅ Task '{task}' finished successfully.")


if __name__ == "__main__":
    main()