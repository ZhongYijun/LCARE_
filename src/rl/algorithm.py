# src/rl/algorithm.py

import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from copy import deepcopy
from transformers import AutoTokenizer

from src.models.actor_critic import LCARE_Actor, LCARE_Critic


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int = None) -> torch.Tensor:
    """计算带掩码的张量的平均值。"""
    if dim is not None:
        return (values * mask).sum(dim=dim) / (mask.sum(dim=dim) + 1e-8)
    return (values * mask).sum() / (mask.sum() + 1e-8)


def masked_var(values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    """计算带掩码的张量的方差。"""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values ** 2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum > 1:
            variance = variance * mask_sum / (mask_sum - 1)
    return variance


def compute_gae(
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        response_mask: torch.Tensor,
        gamma: float,
        tau: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算广义优势估计 (GAE) 和回报 (Returns)。"""
    advantages_reversed = []
    last_gae_lam = torch.zeros_like(values[:, -1])

    next_values = torch.cat([values[:, 1:], torch.zeros_like(values[:, -1:])], dim=1)

    for t in reversed(range(rewards.shape[1])):
        next_non_terminal = 1.0 - dones[:, t]
        delta = rewards[:, t] + gamma * next_values[:, t] * next_non_terminal - values[:, t]
        last_gae_lam = delta + gamma * tau * next_non_terminal * last_gae_lam
        advantages_reversed.append(last_gae_lam)

    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values

    # 对优势进行白化 (Normalization)
    whitened_advantages = (advantages - masked_mean(advantages, response_mask)) / (
            torch.sqrt(masked_var(advantages, response_mask)) + 1e-8)
    whitened_advantages = whitened_advantages * response_mask

    return whitened_advantages, returns


class OffPolicyPPO_Trainer:
    """[V-PER-FINAL] 应用了重要性采样权重的PPO训练器。"""

    def __init__(self, actor: LCARE_Actor, critic: LCARE_Critic, optimizer: torch.optim.Optimizer,
                 config, tokenizer: AutoTokenizer, rank: int, world_size: int):
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.config = config
        self.tokenizer = tokenizer
        self.rank = rank
        self.world_size = world_size
        self.kl_coef = config.get("kl_coef", 0.01)
        self.old_actor = deepcopy(self.actor).eval()

    def update_old_policy(self):
        self.old_actor.load_state_dict(self.actor.state_dict())

    def train_step(self, batch: Dict[str, torch.Tensor], return_loss: bool = False) -> Tuple[
        Dict, Optional[torch.Tensor]]:
        """执行一步完整的PPO更新，包含PER的重要性采样权重。"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        advantages = batch['advantages']
        returns = batch['returns']
        is_weights = batch.get('weights', torch.ones_like(advantages[:, 0]))

        current_log_probs, entropy = self.actor(input_ids, attention_mask, labels)
        predicted_values = self.critic(input_ids, attention_mask).squeeze(-1)

        with torch.no_grad():
            old_log_probs, _ = self.old_actor(input_ids, attention_mask, labels)

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