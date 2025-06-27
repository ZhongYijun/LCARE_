# src/trainers/rl_agent.py

import os
import torch
import torch.distributed as dist
from torch.optim import AdamW
from omegaconf import DictConfig, OmegaConf
from tqdm import trange
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer

from src.models.actor_critic import LCARE_Actor, LCARE_Critic, LCARE_TokenRewardModel
from src.models.lge_encoder import LGE_Encoder
from src.envs.math_reasoning_env import MathReasoningEnv
from src.rl.buffer import LCAREReplayBuffer
from src.rl.algorithm import OffPolicyPPO_Trainer
from src.datasets.rl_prompt_dataset import RLPromptDataset
from src.utils.logger import WandbLogger
from src.utils.verifier import Verifier
from src.utils.prompt_constructor import PromptConstructor
from src.utils.distributed_utils import is_main_process, broadcast_object
import torch.nn.functional as F


class LCARE_Agent:
    """
    [V-LGE-1] 集成了LGE基础设施的Agent。
    - LGE Encoder和Buffer现在只在主进程初始化。
    - 数据收集流程调整为在主进程上进行，以便后续计算内在奖励。
    """

    def __init__(self, config: DictConfig, rank: int, world_size: int):
        self.config = config
        self.agent_config = config.trainer
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")

        self.logger = WandbLogger(config, self.rank)
        self.verifier = Verifier(config)

        OmegaConf.set_struct(config, False)
        if self.agent_config.get("use_lora") is not None:
            config.model.use_lora = self.agent_config.use_lora
            if is_main_process():
                print(f"Overriding model 'use_lora' with trainer's config: {config.model.use_lora}")

        tokenizer = AutoTokenizer.from_pretrained(
            config.model.path, trust_remote_code=True, padding_side='left'
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.prompt_constructor = PromptConstructor(config, tokenizer)
        self.actor = LCARE_Actor(config.model, tokenizer).to(self.device)
        self.critic = LCARE_Critic(config.model.critic).to(self.device)

        self.use_trm = self.agent_config.exploration.get("use_token_reward_model", False)
        self.token_reward_model = LCARE_TokenRewardModel(config.model.token_reward_model).to(
            self.device) if self.use_trm else None

        if os.path.isdir(self.agent_config.initial_policy_path):
            try:
                self.actor.model.load_adapter(self.agent_config.initial_policy_path, "default")
                if self.use_trm:
                    self.token_reward_model.model.load_state_dict(self.actor.model.state_dict(), strict=False)
            except Exception as e:
                print(f"Rank {rank}: WARNING - Load adapter failed. Error: {e}")

        # [LGE] Encoder, Env, Buffer只在主进程初始化
        self.encoder = None
        self.env = None
        self.replay_buffer = None
        if is_main_process():
            base_actor_model = self.actor.model.base_model.model if config.model.use_lora else self.actor.model
            self.encoder = LGE_Encoder(base_actor_model, tokenizer, self.device)
            problem_set = list(RLPromptDataset(self.agent_config.env.problem_set_path))
            self.env = MathReasoningEnv(problem_set, self.verifier, self.prompt_constructor,
                                        self.agent_config.env.max_steps_per_episode)
            self.replay_buffer = LCAREReplayBuffer(config, self.encoder, self.critic, self.token_reward_model,
                                                   tokenizer)

        ac_params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.ac_optimizer = AdamW(ac_params, lr=self.agent_config.algorithm.learning_rate)

        self.trm_lr_scheduler = None
        if self.use_trm:
            self.trm_optimizer = AdamW(self.token_reward_model.parameters(),
                                       lr=self.agent_config.algorithm.trm_learning_rate)
            trm_warmup_steps = self.agent_config.algorithm.get("trm_warmup_steps", 10)
            self.trm_lr_scheduler = LambdaLR(self.trm_optimizer, lr_lambda=lambda step: min(1.0,
                                                                                            float(step + 1) / max(1,
                                                                                                                  trm_warmup_steps)))

        self.ppo_trainer = OffPolicyPPO_Trainer(self.actor, self.critic, self.ac_optimizer, self.agent_config.algorithm,
                                                tokenizer, self.rank, self.world_size)
        self.timesteps = 0
        self.start_iteration = 0
        # self.load_checkpoint() # 等待所有逻辑实现后再开启

    def learn(self):
        pbar = trange(self.start_iteration, self.agent_config.exploration.total_iterations,
                      disable=not is_main_process(), desc="RL Training")

        for iteration in pbar:
            rollout_data_list = self._collect_data_on_main_process()

            if is_main_process():
                rollout_stats = self._process_rollout_data(rollout_data_list)
                self.logger.log({**rollout_stats, 'iteration': iteration}, step=iteration)
                pbar.set_description(
                    f"Iter {iteration} | Buffer: {len(self.replay_buffer)} | Avg Acc: {rollout_stats.get('rollout/correctness', 0):.2f}")

            dist.barrier()

            buffer_size = broadcast_object(len(self.replay_buffer) if is_main_process() else 0, self.rank)
            if buffer_size >= self.agent_config.exploration.learning_starts:
                self._update_models_distributed(iteration)

            dist.barrier()
            if is_main_process() and iteration > 0 and (iteration + 1) % self.agent_config.saving.save_interval == 0:
                self.save_checkpoint(iteration)

    def _collect_data_on_main_process(self) -> List[Dict]:
        if not is_main_process(): return []

        self.actor.eval()
        self.critic.eval()
        if self.use_trm: self.token_reward_model.eval()

        collected_data = []
        num_rollouts = self.agent_config.exploration.rollouts_per_iteration
        for _ in range(num_rollouts):
            trajectory, info = self._collect_one_trajectory()
            if trajectory:
                collected_data.append({'trajectory': trajectory, 'info': info})
        return collected_data

    def _process_rollout_data(self, rollout_data_list: List[Dict]) -> Dict:
        if not is_main_process() or not rollout_data_list: return {}
        rollout_stats = defaultdict(list)
        for item in rollout_data_list:
            self.replay_buffer.add_trajectory(item['trajectory'])
            self.timesteps += len(item['trajectory'])
            rollout_stats['rollout/episode_length'].append(len(item['trajectory']))
            rollout_stats['rollout/correctness'].append(1.0 if item['info'].get('is_correct') else 0.0)
            avg_intrinsic_reward = np.mean([t.get('intrinsic_reward', 0) for t in item['trajectory']])
            rollout_stats['rollout/avg_intrinsic_reward'].append(avg_intrinsic_reward)

        stats = {k: np.mean(v) for k, v in rollout_stats.items()}
        stats['rollout/timesteps'] = self.timesteps
        return stats

    def _collect_one_trajectory(self) -> Tuple[List[Dict], Dict]:
        trajectory, final_info = [], {}
        obs_text, env_info = self.env.reset()

        for _ in range(self.agent_config.env.max_steps_per_episode):
            state_tokens = self.actor.tokenizer(obs_text, return_tensors="pt").to(self.device)
            sampling_params = {'max_new_tokens': 2048, 'do_sample': True, 'temperature': 0.9, 'top_p': 0.95}
            action_ids, behavior_log_prob = self.actor.generate(state_tokens['input_ids'],
                                                                state_tokens['attention_mask'], sampling_params)
            action_text = self.actor.tokenizer.decode(action_ids[0], skip_special_tokens=True)

            next_obs_text, _, terminated, truncated, step_info = self.env.step(action_text)
            done = terminated or truncated

            transition = {
                'state_text': obs_text, 'action_ids': action_ids[0].cpu(), 'next_state_text': next_obs_text,
                'done': done, 'behavior_log_prob': behavior_log_prob[0].cpu(),
                'external_reward': 1.0 if step_info.get('is_correct') else 0.0, 'metadata': env_info
            }
            trajectory.append(transition)
            obs_text = next_obs_text
            if done:
                final_info = step_info
                break
        return trajectory, final_info

    def _train_trm_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.trm_optimizer.zero_grad()
        token_rewards = self.token_reward_model(batch['input_ids'], batch['attention_mask']).squeeze(-1)
        masked_rewards = token_rewards * batch['attention_mask']
        seq_logits = torch.sum(masked_rewards, dim=-1) / (torch.sum(batch['attention_mask'], dim=-1) + 1e-8)
        loss = F.binary_cross_entropy_with_logits(seq_logits, batch['outcome_labels'].float())
        loss.backward()
        self.trm_optimizer.step()
        return loss

    def _update_models_distributed(self, iteration: int):
        self.actor.train()
        self.critic.train()
        if self.use_trm: self.token_reward_model.train()
        self.ppo_trainer.update_old_policy()

        for epoch in range(self.agent_config.algorithm.ppo_epochs):
            log_dict = defaultdict(list)

            # [LGE] 数据采样和广播现在分开进行
            rl_batch_main = self.replay_buffer.sample_for_rl_update(
                self.agent_config.algorithm.batch_size) if is_main_process() else None
            bc_batch_main = self.replay_buffer.sample_for_bc_update(
                self.agent_config.algorithm.batch_size) if is_main_process() else None

            # 主进程将采样到的数据广播给所有进程
            rl_batch = broadcast_object(rl_batch_main, self.rank)
            bc_batch = broadcast_object(bc_batch_main, self.rank)

            if not rl_batch: continue

            rl_dataset = TensorDataset(*[v for _, v in sorted(rl_batch.items())])
            rl_dataloader = DataLoader(rl_dataset, sampler=DistributedSampler(rl_dataset, num_replicas=self.world_size,
                                                                              rank=self.rank),
                                       batch_size=self.agent_config.algorithm.batch_size // self.world_size)

            bc_iter = None
            if bc_batch and len(bc_batch['input_ids']) > 0:
                bc_dataset = TensorDataset(*[v for _, v in sorted(bc_batch.items())])
                bc_dataloader = DataLoader(bc_dataset,
                                           sampler=DistributedSampler(bc_dataset, num_replicas=self.world_size,
                                                                      rank=self.rank),
                                           batch_size=max(1, (
                                                       self.agent_config.algorithm.batch_size // self.world_size) // 4))
                bc_iter = iter(bc_dataloader)

            for micro_rl_batch_tensors in rl_dataloader:
                micro_rl_batch = {k: v.to(self.device) for k, v in zip(sorted(rl_batch.keys()), micro_rl_batch_tensors)}

                if self.use_trm:
                    log_dict['loss/trm'].append(self._train_trm_step(micro_rl_batch).item())

                ppo_log, ppo_loss = self.ppo_trainer.train_step(micro_rl_batch, return_loss=True)
                for k, v in ppo_log.items(): log_dict[k].append(v)

                total_loss = ppo_loss
                if bc_iter:
                    try:
                        micro_bc_batch_tensors = next(bc_iter)
                        micro_bc_batch = {k: v.to(self.device) for k, v in
                                          zip(sorted(bc_batch.keys()), micro_bc_batch_tensors)}
                        bc_loss = self.actor.forward_sft(micro_bc_batch['input_ids'], micro_bc_batch['attention_mask'],
                                                         micro_bc_batch['labels'])
                        log_dict['loss/bc'].append(bc_loss.item())
                        total_loss += self.agent_config.algorithm.bc_loss_weight * bc_loss
                    except StopIteration:
                        pass

                self.ac_optimizer.zero_grad()
                total_loss.backward()
                self.ac_optimizer.step()

                if self.replay_buffer.use_per and is_main_process():
                    with torch.no_grad():
                        new_values = self.critic(micro_rl_batch['input_ids'], micro_rl_batch['attention_mask']).squeeze(
                            -1)
                        td_errors = micro_rl_batch['returns'].sum(dim=-1) - new_values
                    self.replay_buffer.update_priorities(micro_rl_batch['tree_indices'], td_errors)

            if is_main_process():
                final_log = {k: np.mean(v) for k, v in log_dict.items() if v}
                final_log['epoch'] = epoch
                self.logger.log(final_log, step=iteration * self.agent_config.algorithm.ppo_epochs + epoch)

        if self.use_trm and self.trm_lr_scheduler: self.trm_lr_scheduler.step()

    def save_checkpoint(self, iteration: int):
        output_dir = os.path.join(self.agent_config.saving.checkpoint_dir, f"iter_{iteration}")
        if is_main_process():
            print(f"Saving checkpoint to {output_dir}...")
            os.makedirs(output_dir, exist_ok=True)
        dist.barrier()

        # [修复] 正确保存LoRA权重
        if self.config.model.use_lora:
            self.actor.model.save_pretrained(output_dir)
        else:  # 保存完整模型
            torch.save(self.actor.state_dict(), os.path.join(output_dir, "actor.pt"))

        torch.save(self.critic.state_dict(), os.path.join(output_dir, "critic.pt"))
        torch.save(self.ac_optimizer.state_dict(), os.path.join(output_dir, "ac_optimizer.pt"))

        if self.use_trm:
            torch.save(self.token_reward_model.state_dict(), os.path.join(output_dir, "token_reward_model.pt"))
            torch.save(self.trm_optimizer.state_dict(), os.path.join(output_dir, "trm_optimizer.pt"))

        if is_main_process():
            self.replay_buffer.save(os.path.join(output_dir, "replay_buffer.pkl"))
            with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
                json.dump({'iteration': iteration, 'timesteps': self.timesteps}, f)
        dist.barrier()

    def load_checkpoint(self):
        checkpoint_dir = self.agent_config.saving.checkpoint_dir
        if not os.path.isdir(checkpoint_dir): return

        dirs = [d for d in os.listdir(checkpoint_dir) if d.startswith("iter_")]
        if not dirs: return

        latest_iter = max([int(d.split('_')[1]) for d in dirs])
        latest_ckpt_path = os.path.join(checkpoint_dir, f"iter_{latest_iter}")

        print(f"Rank {self.rank}: Resuming training from checkpoint: {latest_ckpt_path}")

        # [修复] 正确加载LoRA权重
        if self.config.model.use_lora:
            self.actor.model.load_adapter(latest_ckpt_path, "default")
        else:
            self.actor.load_state_dict(torch.load(os.path.join(latest_ckpt_path, "actor.pt"), map_location=self.device))

        self.critic.load_state_dict(torch.load(os.path.join(latest_ckpt_path, "critic.pt"), map_location=self.device))
        self.ac_optimizer.load_state_dict(torch.load(os.path.join(latest_ckpt_path, "ac_optimizer.pt")))

        if self.use_trm and os.path.exists(os.path.join(latest_ckpt_path, "token_reward_model.pt")):
            self.token_reward_model.load_state_dict(
                torch.load(os.path.join(latest_ckpt_path, "token_reward_model.pt"), map_location=self.device))
            self.trm_optimizer.load_state_dict(torch.load(os.path.join(latest_ckpt_path, "trm_optimizer.pt")))

        # 只有主进程加载和广播状态
        if is_main_process():
            self.replay_buffer.load(os.path.join(latest_ckpt_path, "replay_buffer.pkl"))
            with open(os.path.join(latest_ckpt_path, "metadata.json"), 'r') as f:
                metadata = json.load(f)
                self.start_iteration = metadata['iteration'] + 1
                self.timesteps = metadata['timesteps']

        # 广播恢复的迭代步数和时间步
        dist.barrier()
        self.start_iteration = broadcast_object(self.start_iteration if is_main_process() else None, self.rank)
        self.timesteps = broadcast_object(self.timesteps if is_main_process() else None, self.rank)
        dist.barrier()
        if is_main_process():
            print(f"Resumed from iteration {self.start_iteration - 1}. Current timesteps: {self.timesteps}")