# # src/rl/buffer.py (REFACTORED & SIMPLIFIED)

# import random
# import numpy as np
# import torch
# import pickle
# import os
# from collections import deque
# from typing import List, Dict, Optional, Tuple
# from omegaconf import DictConfig

# from src.utils.distributed_utils import is_main_process
# from src.rl.sum_tree import SumTree


# class LCAREReplayBuffer:
#     """
#     [REFACTORED] L-CARE Replay Buffer.
#     - 只负责存储和采样原始轨迹数据，不进行任何模型相关的计算。
#     - 所有与模型相关的处理逻辑都已移至 LCARE_Agent。
#     - LGE相关的latent_archive也移至此处，因为它本质上是一种特殊的经验存储。
#     """

#     def __init__(self, config: DictConfig):
#         # 这个Buffer只在主进程上被有效使用
#         if not is_main_process():
#             self.is_main = False
#             return

#         self.is_main = True
#         self.buffer_config = config.trainer.buffer
#         self.use_per = self.buffer_config.get("use_per", True)
#         self.use_lge = config.trainer.exploration.get("use_lge", False)

#         # --- Standard Trajectory Storage ---
#         self.main_storage: List[Optional[List[Dict]]] = [None] * self.buffer_config.capacity
#         self.positive_storage = deque(maxlen=self.buffer_config.get("positive_capacity", 5000))
#         self.next_storage_idx = 0
#         self.storage_size = 0

#         # --- PER (Prioritized Experience Replay) Components ---
#         if self.use_per:
#             self.priority_tree = SumTree(self.buffer_config.capacity)
#             self.alpha = self.buffer_config.get("alpha", 0.6)
#             self.beta_start = self.buffer_config.get("beta", 0.4)
#             self.beta = self.beta_start
#             total_update_steps = config.trainer.exploration.total_iterations * config.trainer.algorithm.ppo_epochs
#             self.beta_increment = (1.0 - self.beta_start) / max(1, total_update_steps)

#         # --- LGE (Latent Go-Explore) Components ---
#         if self.use_lge:
#             self.lge_config = config.trainer.exploration.lge_config
#             # The latent dim needs to be known. We get it from the model config.
#             # This is the only "external" knowledge this class needs.
#             latent_dim = config.model.lora_config.r * 2 if config.model.use_lora else config.model.hidden_size
#             self.latent_state_archive = torch.zeros(
#                 (self.lge_config.archive_capacity, latent_dim), dtype=torch.bfloat16, device='cpu' # Store on CPU to save VRAM
#             )
#             self.latent_archive_size = 0
#             self.next_latent_idx = 0

#     def add_trajectory(self, trajectory: List[Dict]):
#         """
#         向主存储中添加一条完整的轨迹。
#         """
#         if not self.is_main or not trajectory:
#             return

#         traj_idx = self.next_storage_idx
#         self.main_storage[traj_idx] = trajectory
#         self.next_storage_idx = (self.next_storage_idx + 1) % self.buffer_config.capacity
#         if self.storage_size < self.buffer_config.capacity:
#             self.storage_size += 1

#         if trajectory[-1]['external_reward'] == 1.0:
#             self.positive_storage.append(trajectory)

#         if self.use_per:
#             max_p = np.max(self.priority_tree.tree[-self.priority_tree.capacity:]) if self.storage_size > 1 else 1.0
#             self.priority_tree.add(max_p, traj_idx)

#     def sample_trajectories(self, batch_size: int) -> Optional[Tuple[List, List, List]]:
#         """
#         [CORE METHOD] 只采样原始轨迹，不进行任何处理。
#         返回用于分布式处理的原始数据块。
#         """
#         if not self.is_main or self.storage_size < batch_size:
#             return None

#         tree_indices, sampled_trajectories, is_weights = [], [], []
#         if self.use_per and self.priority_tree.size >= batch_size:
#             segment = self.priority_tree.total_priority / batch_size
#             self.beta = min(1.0, self.beta + self.beta_increment)
#             for i in range(batch_size):
#                 s = random.uniform(segment * i, segment * (i + 1))
#                 tree_idx, priority, traj_idx = self.priority_tree.get(s)
#                 sampling_prob = priority / self.priority_tree.total_priority
#                 weight = (self.storage_size * sampling_prob) ** -self.beta
#                 is_weights.append(weight)
#                 tree_indices.append(tree_idx)
#                 sampled_trajectories.append(self.main_storage[traj_idx])
#         else:
#             # Fallback to uniform sampling if PER is not ready
#             sampled_indices = np.random.randint(0, self.storage_size, size=batch_size)
#             sampled_trajectories = [self.main_storage[i] for i in sampled_indices]
#             is_weights = [1.0] * batch_size
#             tree_indices = sampled_indices.tolist()

#         if not sampled_trajectories:
#             return None

#         # Normalize importance sampling weights
#         max_weight = max(is_weights) if is_weights else 1.0
#         normalized_weights = [w / max_weight for w in is_weights]
        
#         return tree_indices, sampled_trajectories, normalized_weights

#     def sample_for_bc(self, batch_size: int) -> Optional[List[List[Dict]]]:
#         """只采样用于BC（行为克隆）的成功轨迹。"""
#         if not self.is_main or len(self.positive_storage) == 0:
#             return None
#         return random.sample(list(self.positive_storage), min(batch_size, len(self.positive_storage)))

#     def update_priorities(self, tree_indices: torch.Tensor, td_errors: torch.Tensor):
#         """根据TD-error更新样本的优先级。"""
#         if not self.use_per or not self.is_main:
#             return
#         priorities = (torch.abs(td_errors) + 1e-6).detach().cpu().numpy() ** self.alpha
#         for idx, p in zip(tree_indices.tolist(), priorities):
#             self.priority_tree.update(idx, p)

#     def save(self, file_path: str):
#         """保存Buffer状态到文件。"""
#         if not self.is_main: return
#         data_to_save = {
#             "main_storage": self.main_storage, 
#             "positive_storage": self.positive_storage,
#             "next_storage_idx": self.next_storage_idx, 
#             "storage_size": self.storage_size,
#             "priority_tree": self.priority_tree.tree if self.use_per else None
#         }
#         if self.use_lge:
#             data_to_save["latent_state_archive"] = self.latent_state_archive
#             data_to_save["latent_archive_size"] = self.latent_archive_size
#             data_to_save["next_latent_idx"] = self.next_latent_idx

#         with open(file_path, 'wb') as f:
#             pickle.dump(data_to_save, f)

#     def load(self, file_path: str):
#         """从文件加载Buffer状态。"""
#         if not self.is_main or not os.path.exists(file_path): return
#         with open(file_path, 'rb') as f:
#             data = pickle.load(f)
        
#         self.main_storage = data["main_storage"]
#         self.positive_storage = data["positive_storage"]
#         self.next_storage_idx = data["next_storage_idx"]
#         self.storage_size = data["storage_size"]
        
#         if self.use_per and data.get("priority_tree") is not None:
#             self.priority_tree.tree = data["priority_tree"]
#             self.priority_tree.size = self.storage_size
        
#         if self.use_lge and data.get("latent_state_archive") is not None:
#             self.latent_state_archive = data["latent_state_archive"]
#             self.latent_archive_size = data["latent_archive_size"]
#             self.next_latent_idx = data["next_latent_idx"]

#     def __len__(self):
#         return self.storage_size if self.is_main else 0

# src/rl/buffer.py (FINAL STABLE VERSION)

import random
import numpy as np
import torch
import pickle
import os
from collections import deque
from typing import List, Dict, Optional, Tuple
from omegaconf import DictConfig

from src.utils.distributed_utils import is_main_process
from src.rl.sum_tree import SumTree


class LCAREReplayBuffer:
    """
    [FINAL STABLE] L-CARE Replay Buffer.
    - Decoupled from the main config object. Receives only necessary parameters.
    - Acts as a pure data container for storage and sampling.
    """

    def __init__(self, buffer_config: DictConfig, exploration_config: DictConfig, latent_dim: int):
        if not is_main_process():
            self.is_main = False
            return

        self.is_main = True
        self.buffer_config = buffer_config
        self.use_per = self.buffer_config.get("use_per", True)
        self.use_lge = exploration_config.get("use_lge", False)

        self.main_storage: List[Optional[List[Dict]]] = [None] * self.buffer_config.capacity
        self.positive_storage = deque(maxlen=self.buffer_config.get("positive_capacity", 5000))
        self.next_storage_idx = 0
        self.storage_size = 0

        if self.use_per:
            self.priority_tree = SumTree(self.buffer_config.capacity)
            self.alpha = self.buffer_config.get("alpha", 0.6)
            self.beta_start = self.buffer_config.get("beta", 0.4)
            self.beta = self.beta_start
            # This calculation is now safe as it only depends on sub-configs
            total_update_steps = exploration_config.total_iterations * buffer_config.get("ppo_epochs", 4) # Assuming ppo_epochs is in buffer_config or has a default
            self.beta_increment = (1.0 - self.beta_start) / max(1, total_update_steps)

        if self.use_lge:
            self.lge_config = exploration_config.lge_config
            # [ROBUST FIX] latent_dim is now passed directly as an integer.
            self.latent_dim = latent_dim
            self.latent_state_archive = torch.zeros(
                (self.lge_config.archive_capacity, self.latent_dim), dtype=torch.bfloat16, device='cpu'
            )
            self.latent_archive_size = 0
            self.next_latent_idx = 0

    # ... The rest of the file is identical to the previous stable version ...
    def add_trajectory(self, trajectory: List[Dict]):
        if not self.is_main or not trajectory: return
        traj_idx = self.next_storage_idx
        self.main_storage[traj_idx] = trajectory
        self.next_storage_idx = (self.next_storage_idx + 1) % self.buffer_config.capacity
        if self.storage_size < self.buffer_config.capacity:
            self.storage_size += 1
        if trajectory[-1]['external_reward'] == 1.0:
            self.positive_storage.append(trajectory)
        if self.use_per:
            max_p = np.max(self.priority_tree.tree[-self.priority_tree.capacity:]) if self.storage_size > 1 else 1.0
            self.priority_tree.add(max_p, traj_idx)

    def sample_trajectories(self, batch_size: int) -> Optional[Tuple[List, List, List]]:
        if not self.is_main or self.storage_size < batch_size: return None
        tree_indices, sampled_trajectories, is_weights = [], [], []
        if self.use_per and self.priority_tree.size >= batch_size:
            segment = self.priority_tree.total_priority / batch_size
            self.beta = min(1.0, self.beta + self.beta_increment)
            for i in range(batch_size):
                s = random.uniform(segment * i, segment * (i + 1))
                tree_idx, priority, traj_idx = self.priority_tree.get(s)
                sampling_prob = priority / self.priority_tree.total_priority
                weight = (self.storage_size * sampling_prob) ** -self.beta
                is_weights.append(weight)
                tree_indices.append(tree_idx)
                sampled_trajectories.append(self.main_storage[traj_idx])
        else:
            sampled_indices = np.random.randint(0, self.storage_size, size=batch_size)
            sampled_trajectories = [self.main_storage[i] for i in sampled_indices]
            is_weights = [1.0] * batch_size
            tree_indices = sampled_indices.tolist()
        if not sampled_trajectories: return None
        max_weight = max(is_weights) if is_weights else 1.0
        normalized_weights = [w / max_weight for w in is_weights]
        return tree_indices, sampled_trajectories, normalized_weights

    def sample_for_bc(self, batch_size: int) -> Optional[List[List[Dict]]]:
        if not self.is_main or len(self.positive_storage) == 0: return None
        return random.sample(list(self.positive_storage), min(batch_size, len(self.positive_storage)))

    def update_priorities(self, tree_indices: torch.Tensor, td_errors: torch.Tensor):
        if not self.use_per or not self.is_main: return
        priorities = (torch.abs(td_errors) + 1e-6).detach().cpu().numpy() ** self.alpha
        for idx, p in zip(tree_indices.tolist(), priorities):
            self.priority_tree.update(idx, p)

    def save(self, file_path: str):
        if not self.is_main: return
        data_to_save = {
            "main_storage": self.main_storage, "positive_storage": self.positive_storage,
            "next_storage_idx": self.next_storage_idx, "storage_size": self.storage_size,
            "priority_tree": self.priority_tree.tree if self.use_per else None
        }
        if self.use_lge:
            data_to_save.update({
                "latent_state_archive": self.latent_state_archive,
                "latent_archive_size": self.latent_archive_size,
                "next_latent_idx": self.next_latent_idx
            })
        with open(file_path, 'wb') as f: pickle.dump(data_to_save, f)

    def load(self, file_path: str):
        if not self.is_main or not os.path.exists(file_path): return
        with open(file_path, 'rb') as f: data = pickle.load(f)
        self.main_storage, self.positive_storage = data["main_storage"], data["positive_storage"]
        self.next_storage_idx, self.storage_size = data["next_storage_idx"], data["storage_size"]
        if self.use_per and data.get("priority_tree") is not None:
            self.priority_tree.tree, self.priority_tree.size = data["priority_tree"], self.storage_size
        if self.use_lge and data.get("latent_state_archive") is not None:
            self.latent_state_archive = data["latent_state_archive"]
            self.latent_archive_size = data["latent_archive_size"]
            self.next_latent_idx = data["next_latent_idx"]

    def __len__(self):
        return self.storage_size if self.is_main else 0