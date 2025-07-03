# src/utils/logger.py

import swanlab
from omegaconf import OmegaConf, DictConfig


class SwanLabLogger:
    """
    [REPLACEMENT] 一个封装了SwanLab的简单日志记录器。
    它与WandbLogger有几乎相同的接口，使得迁移变得简单。
    """

    def __init__(self, config: DictConfig, rank: int):
        # [MODIFIED] 从配置中读取 `use_swanlab` 开关
        self.use_swanlab = config.logging.get("use_swanlab", False)
        self.rank = rank

        if self.use_swanlab and self.rank == 0:
            # [MODIFIED] 调用 swanlab.init()
            # API参数几乎一一对应
            swanlab.init(
                project=config.project_name,
                experiment_name=config.experiment_name,
                config=OmegaConf.to_container(config, resolve=True)
            )

    def log_config(self, config_dict: dict):
        if self.use_swanlab and self.rank == 0:
            # SwanLab 的 init 已经接收了 config，但如果想更新或覆盖，可以使用 update_config
            # swanlab.config.update(config_dict)
            # 通常在init时传入就够了，这里可以留作一个空接口或打印信息
            print("Logger: Configuration was logged at initialization.")

    def log(self, data: dict, step: int):
        """
        记录一个数据字典。
        SwanLab的log API与WandB兼容。
        """
        if self.use_swanlab and self.rank == 0:
            # [MODIFIED] 调用 swanlab.log()
            swanlab.log(data, step=step)

        # 在主进程的控制台上也打印日志 (此逻辑保持不变)
        if self.rank == 0:
            log_str = f"Step: {step:<6} | " + " | ".join(
                [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in data.items()])
            print(log_str)

    def finish(self):
        """
        结束当前的SwanLab运行。
        """
        if self.use_swanlab and self.rank == 0:
            # [MODIFIED] 调用 swanlab.finish()
            swanlab.finish()

