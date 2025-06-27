# src/utils/logger.py
import wandb
from omegaconf import OmegaConf


class WandbLogger:
    """一个封装了WandB的简单日志记录器"""

    def __init__(self, config, rank: int):
        self.use_wandb = config.logging.use_wandb
        self.rank = rank

        if self.use_wandb and self.rank == 0:
            wandb.init(
                project=config.project_name,
                name=config.experiment_name,
                config=OmegaConf.to_container(config, resolve=True)
            )

    def log(self, data: dict, step: int):
        if self.use_wandb and self.rank == 0:
            wandb.log(data, step=step)

        # 在主进程的控制台上也打印日志
        if self.rank == 0:
            log_str = f"Step: {step:<6} | " + " | ".join(
                [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in data.items()])
            print(log_str)

    def finish(self):
        if self.use_wandb and self.rank == 0:
            wandb.finish()