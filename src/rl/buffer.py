# src/rl/buffer.py

import random
import numpy as np
import torch
import pickle
import os
from collections import deque, defaultdict
from typing import List, Dict, Optional, Any
from omegaconf import DictConfig
from transformers import AutoTokenizer

from src.models.actor_critic import LCARE_Critic, LCARE_TokenRewardModel
from src.models.lge_encoder import LGE_Encoder
from src.rl.algorithm import compute_gae
from src.utils.distributed_utils import is_main_process
from src.rl.sum_tree import SumTree


class LCAREReplayBuffer:
    """
    [V-LGE-FINAL] 完整实现LGE内在奖励机制的Replay Buffer。
    此版本修复了所有已知bug并保持了原始代码的函数顺序。
    """

    def __init__(self, config: DictConfig, encoder: LGE_Encoder, critic: LCARE_Critic,
                 token_reward_model: Optional[LCARE_TokenRewardModel], tokenizer: AutoTokenizer):
        if not is_main_process():
            self.is_main = False
            return

        self.is_main = True
        self.config = config.trainer
        self.buffer_config = self.config.buffer
        self.algo_config = self.config.algorithm
        self.device = torch.device(f"cuda:{config.rank}")

        self.use_her = self.buffer_config.get("use_her", True)
        self.use_per = self.buffer_config.get("use_per", True)
        self.use_trm = self.config.exploration.get("use_token_reward_model", False)
        self.use_lge = self.config.exploration.get("use_lge", False)

        self.encoder = encoder
        self.critic = critic
        self.token_reward_model = token_reward_model
        self.tokenizer = tokenizer

        self.main_storage: List[Optional[List[Dict]]] = [None] * self.buffer_config.capacity
        self.positive_storage = deque(maxlen=self.buffer_config.get("positive_capacity", 5000))
        self.next_storage_idx = 0
        self.storage_size = 0

        if self.use_per:
            self.priority_tree = SumTree(self.buffer_config.capacity)
            self.alpha = self.buffer_config.get("alpha", 0.6)
            self.beta_start = self.buffer_config.get("beta", 0.4)
            self.beta = self.beta_start
            total_update_steps = self.config.exploration.total_iterations * self.config.algorithm.ppo_epochs
            self.beta_increment = (1.0 - self.beta_start) / max(1, total_update_steps)

        if self.use_lge:
            self.lge_config = self.config.exploration.lge_config
            latent_dim = self.encoder.embedding_layer.embedding_dim
            self.latent_state_archive = torch.zeros(
                (self.lge_config.archive_capacity, latent_dim), dtype=torch.bfloat16, device=self.device
            )
            self.latent_archive_size = 0
            self.next_latent_idx = 0

    def add_trajectory(self, trajectory: List[Dict]):
        if not self.is_main or not trajectory: return

        if self.use_lge:
            trajectory = self._compute_and_add_intrinsic_rewards(trajectory)

        traj_idx = self.next_storage_idx
        self.main_storage[traj_idx] = trajectory
        self.next_storage_idx = (self.next_storage_idx + 1) % self.buffer_config.capacity
        if self.storage_size < self.buffer_config.capacity:
            self.storage_size += 1

        if trajectory[-1]['external_reward'] == 1.0:
            self.positive_storage.append(trajectory)

        if self.use_per:
            max_p = np.max(
                self.priority_tree.tree[-self.priority_tree.capacity:]) if self.priority_tree.size > 0 else 1.0
            self.priority_tree.add(max_p, traj_idx)

    def sample_for_rl_update(self, batch_size: int) -> Optional[Dict[str, Any]]:
        if not self.is_main or self.storage_size < batch_size: return None

        tree_indices, sampled_trajectories, is_weights = [], [], []
        if self.use_per and self.priority_tree.size >= batch_size:
            segment = self.priority_tree.total_priority / batch_size
            self.beta = min(1., self.beta + self.beta_increment)
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

        processed_batch = self._process_and_collate_rl_batch(sampled_trajectories)
        processed_batch['tree_indices'] = torch.tensor(tree_indices, dtype=torch.long)
        max_weight = max(is_weights) if is_weights else 1.0
        processed_batch['weights'] = torch.tensor(is_weights, dtype=torch.float32) / max_weight
        return processed_batch

    def sample_for_bc_update(self, batch_size: int) -> Optional[Dict[str, Any]]:
        if not self.is_main or len(self.positive_storage) == 0: return None
        actual_batch_size = min(batch_size, len(self.positive_storage))
        sampled_trajectories = random.sample(list(self.positive_storage), actual_batch_size)
        return self._collate_fn_for_bc(sampled_trajectories)

    # --- Helper methods start here ---

    def _compute_and_add_intrinsic_rewards(self, trajectory: List[Dict]) -> List[Dict]:
        all_states_text = [t['state_text'] for t in trajectory]

        with torch.no_grad():
            new_latent_vectors = self.encoder.encode(all_states_text).to(torch.bfloat16)

        intrinsic_rewards = []
        for latent_vec in new_latent_vectors:
            reward = 0.0
            if self.latent_archive_size > self.lge_config.k_nearest_neighbors:
                distances = torch.norm(self.latent_state_archive[:self.latent_archive_size] - latent_vec, dim=1)
                k = self.lge_config.k_nearest_neighbors
                # [修复] 正确解包 torch.topk 的返回值，我们只关心距离值
                knn_distances, _ = torch.topk(distances, k, largest=False)
                reward = knn_distances.mean().item()

            intrinsic_rewards.append(reward)

            self.latent_state_archive[self.next_latent_idx] = latent_vec
            self.next_latent_idx = (self.next_latent_idx + 1) % self.lge_config.archive_capacity
            if self.latent_archive_size < self.lge_config.archive_capacity:
                self.latent_archive_size += 1

        for i, transition in enumerate(trajectory):
            transition['intrinsic_reward'] = intrinsic_rewards[i]
        return trajectory

    def _process_and_collate_rl_batch(self, trajectories: List[List[Dict]]) -> Dict[str, torch.Tensor]:
        collated = defaultdict(list)
        her_k = self.buffer_config.her_k_relabel if self.use_her else 0

        for trajectory in trajectories:
            final_reward = trajectory[-1]['external_reward']
            pass_rate = trajectory[0]['metadata'].get('pass_rate', 0.5)
            self._process_single_goal_trajectory(collated, trajectory, final_reward, pass_rate)

            if self.use_her and her_k > 0 and len(trajectory) > 1:
                future_indices = np.random.randint(0, len(trajectory), size=her_k)
                for future_idx in future_indices:
                    self._process_single_goal_trajectory(collated, trajectory, 1.0, 0.5, achieved_state_idx=future_idx)

        return self._collate_fn(collated)

    def _process_single_goal_trajectory(self, collated_batch: defaultdict, trajectory: List[Dict], final_reward: float,
                                        pass_rate: float, achieved_state_idx: Optional[int] = None):
        current_trajectory = trajectory
        if achieved_state_idx is not None:
            current_trajectory = trajectory[:achieved_state_idx + 1]

        if not current_trajectory: return

        full_texts = [trans['state_text'] + self.tokenizer.decode(trans['action_ids'], skip_special_tokens=True) for
                      trans in current_trajectory]
        dones = torch.tensor([float(trans['done']) for trans in current_trajectory], device=self.device)

        if achieved_state_idx is not None:
            dones[-1] = 1.0

        with torch.no_grad():
            tokenized = self.tokenizer(full_texts, padding='longest', truncation=True, max_length=1024,
                                       return_tensors="pt").to(self.device)
            rewards = torch.zeros_like(tokenized.input_ids, dtype=torch.float, device=self.device)

            if self.use_trm:
                rewards += self.token_reward_model(tokenized.input_ids, tokenized.attention_mask).squeeze(-1)

            if self.use_lge:
                intrinsic_rewards = torch.tensor([t.get('intrinsic_reward', 0.0) for t in current_trajectory],
                                                 device=self.device)
                sequence_lengths = (tokenized.attention_mask == 1).sum(dim=1) - 1
                for i in range(len(current_trajectory)):
                    rewards[i, sequence_lengths[i]] += self.lge_config.bonus_coef * intrinsic_rewards[i]

            extrinsic_reward_val = final_reward if achieved_state_idx is None else 1.0
            if achieved_state_idx is None and final_reward != 1.0:
                extrinsic_reward_val = -pass_rate
            reward_token_pos = (tokenized.attention_mask[-1] == 1).sum() - 1
            rewards[-1, reward_token_pos] += extrinsic_reward_val

            values = self.critic(tokenized.input_ids, tokenized.attention_mask).squeeze(-1)

        advantages, returns = compute_gae(rewards, values, dones, tokenized.attention_mask, self.algo_config.gamma,
                                          self.algo_config.tau_gae)

        collated_batch['full_text'].extend(full_texts)
        collated_batch['prompt_text'].extend([trans['state_text'] for trans in current_trajectory])
        collated_batch['behavior_log_prob'].extend([trans['behavior_log_prob'] for trans in current_trajectory])
        collated_batch['advantages'].append(advantages)
        collated_batch['returns'].append(returns)
        collated_batch['outcome_labels'].extend([final_reward] * len(current_trajectory))

    def _collate_fn(self, batch_dict: Dict) -> Dict[str, torch.Tensor]:
        padded_batch = {}
        tokenized_full = self.tokenizer(
            batch_dict['full_text'], padding='longest', truncation=True, max_length=8192, return_tensors='pt'
        )
        padded_batch['input_ids'] = tokenized_full['input_ids']
        padded_batch['attention_mask'] = tokenized_full['attention_mask']

        prompts_tokenized = self.tokenizer(batch_dict['prompt_text'], padding='longest', truncation=True,
                                           max_length=4096)
        prompt_lengths = [len(ids) for ids in prompts_tokenized['input_ids']]

        labels = padded_batch['input_ids'].clone()
        for i, p_len in enumerate(prompt_lengths):
            labels[i, :p_len] = -100
        labels[padded_batch['input_ids'] == self.tokenizer.pad_token_id] = -100
        padded_batch['labels'] = labels

        for key in ['behavior_log_prob', 'outcome_labels']:
            padded_batch[key] = torch.tensor(batch_dict[key], dtype=torch.float32)

        for key in ['advantages', 'returns']:
            padded_batch[key] = torch.nn.utils.rnn.pad_sequence(
                batch_dict[key], batch_first=True, padding_value=0
            )
        return padded_batch

    def _collate_fn_for_bc(self, trajectories: List[List[Dict]]) -> Dict[str, torch.Tensor]:
        full_texts = []
        prompts_text = [traj[0]['state_text'] for traj in trajectories]
        for trajectory in trajectories:
            full_text = trajectory[0]['state_text']
            for transition in trajectory:
                action_text = self.tokenizer.decode(transition['action_ids'], skip_special_tokens=True)
                full_text += action_text
            full_texts.append(full_text + self.tokenizer.eos_token)

        tokenized = self.tokenizer(full_texts, padding='longest', truncation=True, max_length=8192, return_tensors='pt')
        labels = tokenized['input_ids'].clone()

        prompts_tokenized = self.tokenizer(prompts_text, padding='longest', truncation=True, max_length=4096)
        prompt_lengths = [len(ids) for ids in prompts_tokenized['input_ids']]
        for i, p_len in enumerate(prompt_lengths):
            labels[i, :p_len] = -100

        labels[tokenized['input_ids'] == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": tokenized.input_ids,
            "attention_mask": tokenized.attention_mask,
            "labels": labels
        }

    def update_priorities(self, tree_indices: torch.Tensor, td_errors: torch.Tensor):
        if not self.use_per or not self.is_main: return
        priorities = (torch.abs(td_errors) + 1e-6).detach().cpu().numpy() ** self.alpha
        for idx, p in zip(tree_indices.tolist(), priorities):
            self.priority_tree.update(idx, p)

    def save(self, file_path: str):
        if not self.is_main: return
        with open(file_path, 'wb') as f:
            pickle.dump({
                "main_storage": self.main_storage, "positive_storage": self.positive_storage,
                "next_storage_idx": self.next_storage_idx, "storage_size": self.storage_size,
                "priority_tree": self.priority_tree.tree if self.use_per else None
            }, f)

    def load(self, file_path: str):
        if not self.is_main or not os.path.exists(file_path): return
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.main_storage = data['main_storage']
            self.positive_storage = data['positive_storage']
            self.next_storage_idx = data['next_storage_idx']
            self.storage_size = data['storage_size']
            if self.use_per and data['priority_tree'] is not None:
                self.priority_tree.tree = data['priority_tree']
                self.priority_tree.size = self.storage_size

    def __len__(self):
        return self.storage_size if self.is_main else 0