# src/utils/config_loader.py

import os
from omegaconf import DictConfig, OmegaConf


def load_and_resolve_config(config: DictConfig) -> DictConfig:
    """
    处理从Hydra接收到的配置对象。
    主要职责：
    1. 解析配置中的变量引用 (e.g., ${data.raw_dir})。
    2. 设置基于配置的环境变量，以实现更好的实践。

    Args:
        config (DictConfig): 从Hydra主函数接收的原始配置对象。

    Returns:
        DictConfig: 解析和处理后的配置对象。
    """
    print("Resolving configuration variables...")

    # 解析配置中所有的 ${...} 引用
    OmegaConf.resolve(config)

    # 这是一个良好的实践：将WandB的实体和项目名设置为环境变量
    # 这样可以方便地在不同机器上运行，而无需修改配置文件
    if config.logging.use_wandb:
        if "project_name" in config:
            os.environ["WANDB_PROJECT"] = config.project_name
        if config.logging.get("wandb_entity"):
            os.environ["WANDB_ENTITY"] = config.logging.wandb_entity

    print("Configuration resolved and set up.")
    return config