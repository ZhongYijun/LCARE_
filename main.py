# main.py

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.multiprocessing as mp

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from src.data_processing.build_math_datasets import build_datasets
from src.data_processing.build_enhancement_data import create_enhancement_data
from evaluate import run_evaluation
from src.utils.distributed_utils import setup_distributed, cleanup_distributed


def worker_process(rank: int, world_size: int, config: DictConfig):
    """
    一个通用的分布式工作进程函数。
    它根据 'main.task' 的值来实例化并运行相应的训练器。
    """
    setup_distributed(rank, world_size)
    try:
        # 使用Hydra的instantiate来动态创建训练器
        # Hydra的命令行覆盖机制 (e.g., trainer=sft) 会自动选择正确的配置部分
        trainer = hydra.utils.instantiate(config.trainer, config=config, rank=rank, world_size=world_size)

        task = config.main.task
        if task == "train_sft":
            trainer.train()
        elif task == "train_rl":
            trainer.learn()

    except Exception as e:
        print(f"FATAL ERROR in worker process rank {rank}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_distributed()


@hydra.main(version_base=None, config_path="./configs", config_name="lcare_config")
def main(config: DictConfig) -> None:
    """
    [V-FINAL] 项目主入口，任务分发逻辑已最终确定。
    """
    print("=" * 60)
    print(" L-CARE Project Main Entry (Corrected Version)    ")
    print("=" * 60)
    print(" Resolved Configuration:")
    print(OmegaConf.to_yaml(config))
    print("-" * 60)

    task = config.main.task
    valid_tasks = ["process_data", "train_sft", "train_rl", "evaluate", "create_enhancement_data"]
    if task not in valid_tasks:
        raise ValueError(f"Unknown task: '{task}'. Available tasks are: {valid_tasks}")

    if task == "process_data":
        print(f"🚀 Starting task: {task}...")
        build_datasets(config)

    elif task == "create_enhancement_data":
        print(f"🚀 Starting task: {task}...")
        create_enhancement_data(config)

    elif task in ["train_sft", "train_rl"]:
        world_size = torch.cuda.device_count()
        if world_size == 0:
            raise EnvironmentError("No GPUs found for distributed training.")
        print(f"🚀 Starting distributed task: {task} with {world_size} GPUs...")
        mp.spawn(worker_process, args=(world_size, config), nprocs=world_size, join=True)

    elif task == "evaluate":
        print(f"🚀 Starting task: {task}...")
        run_evaluation(config)

    print(f"\n✅ Task '{task}' finished successfully.")


if __name__ == "__main__":
    main()