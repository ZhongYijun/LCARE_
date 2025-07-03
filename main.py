# # main.py (FINAL ROBUST VERSION)
#
# import hydra
# from omegaconf import DictConfig, OmegaConf
# import torch.distributed as dist
#
# import sys
# from pathlib import Path
#
# sys.path.insert(0, str(Path(__file__).resolve().parent))
# import os
#
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
#
# from src.data_processing.build_math_datasets import build_datasets
# from src.data_processing.build_enhancement_data import create_enhancement_data
# from evaluate import run_evaluation  # 我们将直接调用重构后的run_evaluation
# from src.utils.distributed_utils import setup_distributed, cleanup_distributed, is_main_process, get_rank
# from hydra.utils import get_class
# from src.utils.logger import SwanLabLogger
#
#
# def run_distributed_training(config: DictConfig):
#     """
#     用于SFT和RL训练的分布式任务执行函数。
#     """
#     logger = None
#
#     try:
#         rank = int(os.environ["RANK"])
#         world_size = int(os.environ["WORLD_SIZE"])
#
#         setup_distributed(rank, world_size)
#         logger = SwanLabLogger(config, rank)
#
#         target_class_path = config.trainer._target_
#         TrainerClass = get_class(target_class_path)
#         trainer = TrainerClass(config=config, rank=rank, world_size=world_size, logger=logger)
#
#         task = config.main.task
#         if task == "train_sft":
#             trainer.train()
#             dist.barrier()
#             if is_main_process():
#                 trainer.save_model()
#             dist.barrier()
#         elif task == "train_rl":
#             trainer.learn()
#
#     except Exception as e:
#         print(f"FATAL ERROR in worker process rank {get_rank()}: {e}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         if logger is not None and is_main_process():
#             logger.finish()
#         cleanup_distributed()
#
#
# # 【核心修复】为评估任务创建一个新的分布式执行函数
# def run_distributed_evaluation(config: DictConfig):
#     """
#     用于并行化评估的分布式任务执行函数。
#     """
#
#     try:
#         rank = int(os.environ["RANK"])
#         world_size = int(os.environ["WORLD_SIZE"])
#
#         setup_distributed(rank, world_size)
#
#         # 直接调用重构后的、能感知分布式环境的评估函数
#         run_evaluation(config=config, rank=rank, world_size=world_size)
#
#     except Exception as e:
#         print(f"FATAL ERROR in worker process rank {get_rank()}: {e}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         cleanup_distributed()
#
#
# @hydra.main(version_base=None, config_path="./configs", config_name="lcare_config_final")
# def main(config: DictConfig) -> None:
#     task = config.main.task
#     if is_main_process():
#         print("=" * 60)
#         print(f" L-CARE Project Main Entry: Running Task -> {task}")
#         print("=" * 60)
#         print(" Resolved Configuration:")
#         print(OmegaConf.to_yaml(config))
#         print("-" * 60)
#
#     valid_tasks = ["process_data", "train_sft", "train_rl", "evaluate", "create_enhancement_data"]
#     if task not in valid_tasks:
#         raise ValueError(f"Unknown task: '{task}'. Available tasks are: {valid_tasks}")
#
#     # --- 任务分派 ---
#     if task in ["train_sft", "train_rl"]:
#         if "RANK" not in os.environ:
#             raise EnvironmentError(f"Task '{task}' must be launched with `torchrun`.")
#         if is_main_process(): print(f"🚀 Starting distributed training: {task}...")
#         run_distributed_training(config)
#
#     # 【核心修复】将'evaluate'也作为分布式任务处理
#     elif task == "evaluate":
#         if "RANK" not in os.environ:
#             raise EnvironmentError(f"Task '{task}' must be launched with `torchrun`.")
#         if is_main_process(): print(f"🚀 Starting distributed evaluation...")
#         run_distributed_evaluation(config)
#
#     elif task == "process_data":
#         if is_main_process():
#             print(f"🚀 Starting task: {task}...")
#             build_datasets(config)
#
#     elif task == "create_enhancement_data":
#         if is_main_process():
#             print(f"🚀 Starting task: {task}...")
#             create_enhancement_data(config)
#
#     if dist.is_initialized():
#         dist.barrier()
#
#     if is_main_process():
#         print(f"\n✅ Task '{task}' finished successfully.")
#
#
# if __name__ == "__main__":
#     main()

# # main.py (L-CARE V2 - FINAL ROBUST VERSION)
#
# import hydra
# from omegaconf import DictConfig, OmegaConf
# import torch.distributed as dist
# import os
# import sys
# from pathlib import Path
#
# # 将项目根目录添加到Python路径，确保可以正确导入src下的模块
# sys.path.insert(0, str(Path(__file__).resolve().parent))
#
# # 导入项目模块
# from src.utils.distributed_utils import setup_distributed, cleanup_distributed, is_main_process, get_rank
# from src.utils.logger import SwanLabLogger
# from hydra.utils import get_class
#
# # --- 按需导入任务函数 ---
# from src.data_processing.build_math_datasets import build_datasets
# from src.data_processing.build_enhancement_data import create_enhancement_data
# from evaluate import run_evaluation
#
#
# def run_distributed_task(config: DictConfig):
#     """
#     由每个 torchrun 启动的进程直接调用的分布式任务执行函数。
#     此函数负责设置分布式环境、初始化日志和训练器，并执行训练流程。
#     """
#     rank = -1  # 初始化rank以备在错误日志中使用
#     logger = None
#     try:
#         # 从环境变量获取rank和world_size，这是torchrun的标准做法
#         rank = int(os.environ["RANK"])
#         world_size = int(os.environ["WORLD_SIZE"])
#
#         # 初始化分布式环境
#         setup_distributed(rank, world_size)
#
#         # 初始化日志记录器 (只有主进程会真正写入到SwanLab)
#         logger = SwanLabLogger(config, rank)
#
#         # --- [关键修复] 使用 get_class 显式实例化，避免Hydra的参数冲突 ---
#         # 1. 从配置中获取目标训练器类的完整路径
#         target_class_path = config.trainer._target_
#
#         # 2. 使用Hydra的工具函数动态地加载这个类
#         TrainerClass = get_class(target_class_path)
#
#         # 3. 明确地实例化这个类，传入它需要的所有参数
#         trainer = TrainerClass(config=config, rank=rank, world_size=world_size, logger=logger)
#
#         # --- 训练与保存分离，保证FSDP的稳健性 ---
#         task = config.main.task
#         if task == "train_sft":
#             # 步骤 1: 执行完整的训练流程
#             trainer.train()
#
#             # 步骤 2: 训练完全结束后，进行一次同步，确保所有进程都完成了训练
#             dist.barrier()
#             if is_main_process():
#                 logger.info("SFT training finished. Synchronizing all processes before final save.")
#
#             # 步骤 3: 只有主进程负责保存模型，这是最安全的方式
#             if is_main_process():
#                 trainer.save_model()
#
#         elif task == "train_rl":
#             # RL的learn方法内部包含了迭代和保存逻辑，直接调用即可
#             trainer.learn()
#
#     except Exception as e:
#         # 在出错的进程上打印详细的错误信息
#         print(f"FATAL ERROR in worker process rank {get_rank()}: {e}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         # 确保在任何情况下都清理资源
#         if logger and is_main_process():
#             logger.finish()
#         cleanup_distributed()
#
#
# @hydra.main(version_base=None, config_path="./configs", config_name="lcare_config")
# def main(config: DictConfig) -> None:
#     """
#     [L-CARE V2] 项目主入口。
#     - 使用Hydra进行配置管理。
#     - 智能分派分布式任务和单进程任务。
#     """
#     task = config.main.task
#
#     # 只有主进程才打印冗长的配置信息，保持日志清洁
#     if is_main_process():
#         print("=" * 80)
#         print(f" L-CARE Project - Main Entry Point | Task: '{task}'")
#         print("=" * 80)
#         # 使用OmegaConf.to_yaml打印完整的、解析后的配置，更易读
#         print("--- Resolved Configuration ---")
#         print(OmegaConf.to_yaml(config))
#         print("------------------------------\n")
#
#     valid_tasks = ["process_data", "create_enhancement_data", "train_sft", "train_rl", "evaluate"]
#     if task not in valid_tasks:
#         raise ValueError(f"Unknown task: '{task}'. Available tasks are: {valid_tasks}")
#
#     # --- 任务分派逻辑 ---
#
#     # 分布式任务: 必须由 torchrun 启动
#     if task in ["train_sft", "train_rl"]:
#         if "RANK" not in os.environ:
#             raise EnvironmentError(
#                 f"Task '{task}' is a distributed task and must be launched with `torchrun`.\n"
#                 "Example: torchrun --nproc_per_node=2 main.py main.task=train_rl"
#             )
#
#         if is_main_process():
#             print(f"🚀 Starting distributed task: {task} on {os.environ['WORLD_SIZE']} GPUs...")
#
#         run_distributed_task(config)
#
#     # 单进程任务: 只在主进程上执行
#     elif task == "process_data":
#         if is_main_process():
#             print(f"🚀 Starting single-process task: {task}...")
#             build_datasets(config)
#
#     elif task == "create_enhancement_data":
#         if is_main_process():
#             print(f"🚀 Starting single-process task: {task}...")
#             create_enhancement_data(config)
#
#     elif task == "evaluate":
#         if is_main_process():
#             print(f"🚀 Starting single-process task: {task}...")
#             # [CRITICAL FIX] 使用正确的函数签名调用评估函数
#             run_evaluation(config)
#
#     # 使用barrier确保所有进程（尤其是分布式任务）都完成后再打印最终的成功信息
#     if dist.is_initialized():
#         dist.barrier()
#
#     if is_main_process():
#         print(f"\n✅ Task '{task}' finished successfully.")
#
#
# if __name__ == "__main__":
#     main()

# main.py
# [L-CARE V15.6-FIXED - FINAL ROBUST & SYNERGISTIC FORM]

import hydra
from omegaconf import DictConfig
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import logging
import sys
from pathlib import Path

# 将项目根目录添加到Python路径，确保可以正确导入src下的模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Local project imports
from src.utils.config_loader import load_and_resolve_config
from src.utils.distributed_utils import setup_distributed, cleanup_distributed, is_main_process
from src.utils.logger import SwanLabLogger
from src.data_processing import build_math_datasets, build_enhancement_data, generate_rft_data
from src.trainers.sft_trainer import SFTTrainer
from src.trainers.rl_agent import LCARE_Agent
from evaluate import run_evaluation

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_distributed_task(rank: int, world_size: int, config: DictConfig):
    """
    The main entry point for each distributed process.
    """
    try:
        setup_distributed(rank, world_size)

        # Initialize logger for this specific rank
        # SwanLab logger will only activate on rank 0
        swanlab_logger = SwanLabLogger(config, rank)

        task = config.main.task
        logger.info(f"[Rank {rank}] Starting task: {task}")

        if task == "train_sft":
            trainer = SFTTrainer(config, rank, world_size, swanlab_logger)
            trainer.train()
            if is_main_process():
                trainer.save_model()
        elif task == "train_rl":
            agent = LCARE_Agent(config, rank, world_size, swanlab_logger)
            agent.learn()
        else:
            if is_main_process():
                logger.error(f"Unsupported distributed task: {task}")

    except Exception as e:
        logger.error(f"FATAL ERROR in worker process rank {rank}: {e}", exc_info=True)
    finally:
        if dist.is_initialized():
            cleanup_distributed()
        if 'swanlab_logger' in locals() and is_main_process():
            swanlab_logger.finish()


@hydra.main(version_base=None, config_path="configs", config_name="lcare_config")
def main(config: DictConfig) -> None:
    """
    Hydra entry point. Sets up the environment and spawns distributed processes.
    """
    config = load_and_resolve_config(config)

    # --- [CRITICAL FIX for CUDA in subprocesses] ---
    # Set the multiprocessing start method to 'spawn' at the very beginning.
    # This is mandatory for using CUDA in child processes created via multiprocessing.
    # 'spawn' creates a fresh process, avoiding CUDA re-initialization errors that
    # occur with the default 'fork' method.
    try:
        # force=True is important if the context might be set elsewhere
        mp.set_start_method("spawn", force=True)
        logger.info("✅ Multiprocessing start method successfully set to 'spawn'.")
    except RuntimeError as e:
        # This might happen in some environments (like Jupyter notebooks) if already set.
        logger.info(f"ℹ️ Multiprocessing start method was already set. Info: {e}")
    # --- [END OF FIX] ---

    task = config.main.task
    world_size = torch.cuda.device_count()

    if task in ["train_sft", "train_rl"]:
        if world_size > 0:
            logger.info(f"Spawning {world_size} processes for distributed task: {task}")
            mp.spawn(run_distributed_task, args=(world_size, config), nprocs=world_size, join=True)
        else:
            logger.error("No CUDA devices found for distributed training.")
    elif task == "process_data":
        logger.info("Running single-process task: process_data")
        build_math_datasets.build_datasets(config)
    elif task == "generate_rft_data":
        logger.info("Running single-process task: generate_rft_data")
        generate_rft_data.generate_rft_data(config)
    elif task == "create_enhancement_data":
        logger.info("Running single-process task: create_enhancement_data")
        build_enhancement_data.create_enhancement_data(config)
    elif task == "evaluate":
        logger.info("Running evaluation...")
        # Standalone evaluation can be run distributedly if needed, or single-process.
        # Here we assume a standalone call might want to use all GPUs.
        if world_size > 1:
            mp.spawn(run_distributed_task, args=(world_size, config), nprocs=world_size, join=True)
        else:
            run_evaluation(config)
    else:
        logger.error(f"Unknown task specified in config: {task}")

    if is_main_process():
        logger.info(f"✅ Task '{task}' finished successfully.")


if __name__ == "__main__":
    main()