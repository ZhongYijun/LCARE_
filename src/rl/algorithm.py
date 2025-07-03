# # src/rl/algorithm.py

# import torch
# import torch.nn.functional as F
# from typing import Tuple, Dict, Optional
# from transformers import AutoTokenizer

# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


# def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int = None) -> torch.Tensor:
#     if dim is not None:
#         return (values * mask).sum(dim=dim) / (mask.sum(dim=dim) + 1e-8)
#     return (values * mask).sum() / (mask.sum() + 1e-8)


# def masked_var(values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
#     mean = masked_mean(values, mask)
#     centered_values = values - mean
#     variance = masked_mean(centered_values ** 2, mask)
#     if unbiased:
#         mask_sum = mask.sum()
#         if mask_sum > 1:
#             variance = variance * mask_sum / (mask_sum - 1)
#     return variance


# def compute_gae(
#         rewards: torch.Tensor,
#         values: torch.Tensor,
#         dones: torch.Tensor,
#         response_mask: torch.Tensor,
#         gamma: float,
#         tau: float
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     advantages_reversed = []
#     last_gae_lam = torch.zeros_like(values[:, -1])
#     next_values = torch.cat([values[:, 1:], torch.zeros_like(values[:, -1:])], dim=1)
#     for t in reversed(range(rewards.shape[1])):
#         next_non_terminal = 1.0 - dones[:, t]
#         delta = rewards[:, t] + gamma * next_values[:, t] * next_non_terminal - values[:, t]
#         last_gae_lam = delta + gamma * tau * next_non_terminal * last_gae_lam
#         advantages_reversed.append(last_gae_lam)
#     advantages = torch.stack(advantages_reversed[::-1], dim=1)
#     returns = advantages + values
#     whitened_advantages = (advantages - masked_mean(advantages, response_mask)) / (
#             torch.sqrt(masked_var(advantages, response_mask)) + 1e-8)
#     whitened_advantages = whitened_advantages * response_mask
#     return whitened_advantages, returns


# class OffPolicyPPO_Trainer:
#     """
#     [V-FINAL-INIT-FIX] PPO训练器。
#     - 不再负责创建old_actor，只接收已经准备好的模型。
#     - update_old_policy只负责同步参数。
#     """
#     def __init__(self, actor: FSDP, old_actor: FSDP, critic: FSDP, optimizer: torch.optim.Optimizer,
#                  config, tokenizer: AutoTokenizer, rank: int, world_size: int):
#         self.actor = actor
#         self.old_actor = old_actor
#         self.critic = critic
#         self.optimizer = optimizer
#         self.config = config
#         self.tokenizer = tokenizer
#         self.rank = rank
#         self.world_size = world_size
#         self.kl_coef = config.get("kl_coef", 0.01)

#     def update_old_policy(self):
#         """
#         [SIMPLIFIED] 只负责同步参数。这是一个快速且安全的集体操作。
#         """
#         self.old_actor.load_state_dict(self.actor.state_dict())

#     def train_step(self, batch: Dict[str, torch.Tensor], return_loss: bool = False) -> Tuple[
#         Dict, Optional[torch.Tensor]]:
#         input_ids = batch['input_ids']
#         attention_mask = batch['attention_mask']
#         labels = batch['labels']
#         advantages = batch['advantages']
#         returns = batch['returns']
#         is_weights = batch.get('weights', torch.ones_like(advantages[:, 0]))

#         current_log_probs, entropy = self.actor(input_ids, attention_mask, labels)
#         predicted_values = self.critic(input_ids, attention_mask).squeeze(-1)

#         with torch.no_grad():
#             old_log_probs, _ = self.old_actor(input_ids, attention_mask, labels)

#         action_mask = (labels != -100).float()
#         current_seq_log_probs = (current_log_probs * action_mask).sum(dim=-1)
#         old_seq_log_probs = (old_log_probs * action_mask).sum(dim=-1)

#         seq_advantages = (advantages * action_mask).sum(dim=-1)
#         seq_advantages = (seq_advantages - seq_advantages.mean()) / (seq_advantages.std() + 1e-8)

#         ratio = torch.exp(current_seq_log_probs - old_seq_log_probs)
#         clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)

#         loss_unclipped = ratio * seq_advantages
#         loss_clipped = clipped_ratio * seq_advantages

#         policy_loss = - (torch.min(loss_unclipped, loss_clipped) * is_weights).mean()

#         value_loss = F.mse_loss(predicted_values, returns.sum(dim=-1))
#         entropy_loss = -masked_mean(entropy, action_mask)
#         kl_div = masked_mean(current_log_probs - old_log_probs, action_mask)

#         loss = policy_loss + self.config.vf_coef * value_loss + self.config.entropy_coef * entropy_loss + self.kl_coef * kl_div

#         with torch.no_grad():
#             clip_fraction = torch.mean(torch.gt(torch.abs(ratio - 1.0), self.config.clip_epsilon).float()).item()

#         log_dict = {
#             'loss/policy': policy_loss.item(),
#             'loss/value': value_loss.item(),
#             'loss/entropy': entropy_loss.item(),
#             'loss/kl_div': kl_div.item(),
#             'loss/total_ppo': loss.item(),
#             'stats/clip_fraction': clip_fraction,
#             'stats/is_weights_mean': is_weights.mean().item(),
#         }

#         if return_loss:
#             return log_dict, loss
#         return log_dict, None

# src/rl/algorithm.py (最终的、完整的、适配无old_actor架构的版本)

import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import numpy as np
from transformers import AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP

from src.models.actor_critic import LCARE_Actor, LCARE_Critic

def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int = None) -> torch.Tensor:
    if dim is not None:
        return (values * mask).sum(dim=dim) / (mask.sum(dim=dim).clamp(min=1e-8))
    return (values * mask).sum() / (mask.sum().clamp(min=1e-8))

def masked_var(values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values ** 2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum > 1:
            variance = variance * mask_sum / (mask_sum - 1)
    return variance

def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, response_mask: np.ndarray, gamma: float, tau: float) -> Tuple[np.ndarray, np.ndarray]:
    advantages_reversed = []
    last_gae_lam = 0
    
    next_values = np.concatenate([values[1:], [0]])
    
    for t in reversed(range(len(rewards))):
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] * next_non_terminal - values[t]
        last_gae_lam = delta + gamma * tau * next_non_terminal * last_gae_lam
        advantages_reversed.append(last_gae_lam)
    
    advantages = np.array(advantages_reversed[::-1])
    returns = advantages + values
    
    # 对优势进行白化 (Normalization)
    response_mask_bool = response_mask.astype(bool)
    if response_mask_bool.any():
        # 只有在response_mask中至少有一个True时，才进行白化操作
        adv_tensor = torch.from_numpy(advantages)
        mask_tensor = torch.from_numpy(response_mask)
        whitened_advantages = (adv_tensor - masked_mean(adv_tensor, mask_tensor)) / (
                    torch.sqrt(masked_var(adv_tensor, mask_tensor)) + 1e-8)
        whitened_advantages = whitened_advantages.numpy() * response_mask
    else:
        # 如果response_mask全为False，直接返回原始的advantages，不进行白化
        whitened_advantages = advantages

    return whitened_advantages, returns

class OffPolicyPPO_Trainer:
    def __init__(self, actor: DDP, critic: DDP, optimizer: torch.optim.Optimizer,
                 config, tokenizer: AutoTokenizer, rank: int, world_size: int):
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.config = config
        self.tokenizer = tokenizer
        self.rank = rank
        self.world_size = world_size
        self.kl_coef = config.get("kl_coef", 0.01)

    def train_step(self, batch: Dict[str, torch.Tensor], return_loss: bool = False) -> Tuple[
        Dict, Optional[torch.Tensor]]:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        advantages = batch['advantages']
        returns = batch['returns']
        old_log_probs = batch['behavior_log_probs']
        is_weights = batch.get('weights', torch.ones_like(advantages[:, 0]))

        current_log_probs, entropy = self.actor(input_ids, attention_mask, labels)
        predicted_values = self.critic(input_ids, attention_mask).squeeze(-1)

        action_mask = (labels != -100).float()
        
        current_seq_log_probs = (current_log_probs * action_mask).sum(dim=-1)
        old_seq_log_probs = (old_log_probs * action_mask).sum(dim=-1)

        seq_advantages = (advantages * action_mask).sum(dim=-1)
        seq_advantages = (seq_advantages - seq_advantages.mean()) / (seq_advantages.std() + 1e-8)

        ratio = torch.exp(current_seq_log_probs - old_seq_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)

        loss_unclipped = ratio * seq_advantages
        loss_clipped = clipped_ratio * seq_advantages

        policy_loss = - (torch.min(loss_unclipped, loss_clipped) * is_weights).mean()

        value_loss = F.mse_loss(predicted_values, returns.sum(dim=-1))
        entropy_loss = -masked_mean(entropy, action_mask)
        kl_div = masked_mean(current_log_probs - old_log_probs, action_mask)

        loss = policy_loss + self.config.vf_coef * value_loss + self.config.entropy_coef * entropy_loss + self.kl_coef * kl_div

        with torch.no_grad():
            clip_fraction = torch.mean(torch.gt(torch.abs(ratio - 1.0), self.config.clip_epsilon).float()).item()

        log_dict = {
            'loss/policy': policy_loss.item(),
            'loss/value': value_loss.item(),
            'loss/entropy': entropy_loss.item(),
            'loss/kl_div': kl_div.item(),
            'loss/total_ppo': loss.item(),
            'stats/clip_fraction': clip_fraction,
            'stats/is_weights_mean': is_weights.mean().item(),
        }

        if return_loss:
            return log_dict, loss
        return log_dict, None