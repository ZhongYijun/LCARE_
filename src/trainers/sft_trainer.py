# # src/trainers/rl_agent.py (FINAL & NO-COMPILE)

# import os
# import torch
# import torch.distributed as dist
# from torch.optim import AdamW
# from omegaconf import DictConfig
# from tqdm import trange
# from collections import defaultdict
# from typing import Dict, List, Tuple
# import numpy as np
# import json
# from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
# from copy import deepcopy

# from transformers import AutoTokenizer, PreTrainedModel
# from peft import PeftModel

# import functools
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
# from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
# from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
# from torch.distributed.checkpoint import save, load

# from src.models.actor_critic import LCARE_Actor, LCARE_Critic, LCARE_TokenRewardModel
# from src.models.lge_encoder import LGE_Encoder
# from src.envs.math_reasoning_env import MathReasoningEnv
# from src.rl.buffer import LCAREReplayBuffer
# from src.rl.algorithm import OffPolicyPPO_Trainer, compute_gae
# from src.datasets.rl_prompt_dataset import RLPromptDataset
# from src.utils.logger import SwanLabLogger
# from src.utils.verifier import Verifier
# from src.utils.prompt_constructor import PromptConstructor
# from src.utils.distributed_utils import is_main_process, broadcast_object
# import torch.nn.functional as F


# def _get_transformer_layer_class(model: PreTrainedModel) -> type:
#     if isinstance(model, PeftModel): model = model.get_base_model()
#     for name, module in model.named_modules():
#         if "DecoderLayer" in module.__class__.__name__ or "TransformerBlock" in module.__class__.__name__: return module.__class__
#     raise ValueError("Could not find a valid transformer layer class name.")


# class LCARE_Agent:
#     """
#     [FINAL & NO-COMPILE] L-CARE Reinforcement Learning Agent.
#     - Implements parallel data collection.
#     - torch.compile has been removed to simplify debugging.
#     """

#     def __init__(self, config: DictConfig, rank: int, world_size: int, logger: SwanLabLogger):
#         self.config, self.agent_config, self.model_config = config, config.trainer, config.model
#         self.rank, self.world_size, self.device, self.logger = rank, world_size, torch.device(f"cuda:{rank}"), logger
#         self.verifier = Verifier(config)
#         use_lora = self.model_config.get("use_lora", False)
#         if self.agent_config.get("use_lora") is not None: use_lora = self.agent_config.use_lora
#         self.tokenizer = AutoTokenizer.from_pretrained(config.model.path, trust_remote_code=True, padding_side='left')
#         if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.prompt_constructor = PromptConstructor(config, self.tokenizer)

#         models_for_fsdp_cpu = [
#             LCARE_Actor(self.model_config, self.tokenizer).to('cpu'),
#             LCARE_Critic(config.model.critic).to('cpu'),
#             LCARE_Actor(self.model_config, self.tokenizer).to('cpu')
#         ]

#         if use_lora and os.path.isdir(self.agent_config.initial_policy_path):
#             models_for_fsdp_cpu[0].model.load_adapter(self.agent_config.initial_policy_path, "default")
#             models_for_fsdp_cpu[2].model.load_adapter(self.agent_config.initial_policy_path, "default")

#         self.local_actor = deepcopy(models_for_fsdp_cpu[0]).to(self.device) if is_main_process() else None

#         models_gpu = [m.to(self.device) for m in models_for_fsdp_cpu]
#         fsdp_dtype = torch.bfloat16
#         models_gpu = [m.to(dtype=fsdp_dtype) for m in models_gpu]

#         # [REMOVED] torch.compile calls have been removed for simplification.

#         policies = [functools.partial(transformer_auto_wrap_policy,
#                                       transformer_layer_cls={_get_transformer_layer_class(m.model)}) for m in
#                     models_gpu]
#         fsdp_config = {'auto_wrap_policy': None, 'device_id': self.device, 'use_orig_params': use_lora,
#                        'mixed_precision': MixedPrecision(param_dtype=fsdp_dtype, reduce_dtype=fsdp_dtype,
#                                                          buffer_dtype=fsdp_dtype)}
#         self.actor, self.critic, self.old_actor = (
#             FSDP(m, **{**fsdp_config, 'auto_wrap_policy': p}) for m, p in zip(models_gpu, policies))

#         self.old_actor.load_state_dict(self.actor.state_dict())
#         self.old_actor.eval()

#         self.use_trm = self.agent_config.exploration.get("use_token_reward_model", False)
#         if self.use_trm:
#             trm = LCARE_TokenRewardModel(config.model.token_reward_model).to(self.device)
#             self.token_reward_model = torch.nn.parallel.DistributedDataParallel(trm, device_ids=[rank])
#             if use_lora: self.token_reward_model.module.model.load_state_dict(self.actor.module.model.state_dict(),
#                                                                               strict=False)
#         else:
#             self.token_reward_model = None

#         self.replay_buffer = LCAREReplayBuffer(config) if is_main_process() else None
#         self.encoder = None
#         if is_main_process():
#             # [SIMPLIFIED] Unwrapping logic is simpler without compile.
#             unwrapped_local_actor = self.local_actor
#             base_model_for_encoder = unwrapped_local_actor.model.base_model.model if use_lora else unwrapped_local_actor.model
#             self.encoder = LGE_Encoder(base_model_for_encoder, self.tokenizer, self.device)

#         p_path = str(os.path.join(config.data.processed_dir, config.data.rl_prompt_file))
#         p_set = list(RLPromptDataset(p_path))
#         self.env = MathReasoningEnv(p_set, self.verifier, self.prompt_constructor,
#                                     self.agent_config.env.max_steps_per_episode, self.config)

#         ac_params = list(self.actor.parameters()) + list(self.critic.parameters())
#         self.ac_optimizer = AdamW(ac_params, lr=self.agent_config.algorithm.learning_rate)
#         if self.use_trm: self.trm_optimizer = AdamW(self.token_reward_model.parameters(),
#                                                     lr=self.agent_config.algorithm.trm_learning_rate)
#         self.ppo_trainer = OffPolicyPPO_Trainer(self.actor, self.old_actor, self.critic, self.ac_optimizer,
#                                                 self.agent_config.algorithm, self.tokenizer, self.rank, self.world_size)
#         self.timesteps, self.start_iteration = 0, 0
#         self.load_checkpoint()

#     def learn(self):
#         pbar = trange(self.start_iteration, self.agent_config.exploration.total_iterations,
#                       disable=not is_main_process(), desc="RL Training")
#         for iteration in pbar:
#             self._sync_local_actor()

#             collected_trajs = self._collect_rollouts()

#             all_gathered_trajs = [None] * self.world_size
#             dist.all_gather_object(all_gathered_trajs, collected_trajs)

#             if is_main_process():
#                 total_trajs_in_iter = 0
#                 correct_in_iter = 0
#                 for rank_trajs in all_gathered_trajs:
#                     for traj in rank_trajs:
#                         if traj:
#                             self.replay_buffer.add_trajectory(traj)
#                             self.timesteps += len(traj)
#                             total_trajs_in_iter += 1
#                             if traj[-1]['external_reward'] == 1.0:
#                                 correct_in_iter += 1

#                 avg_acc = (correct_in_iter / total_trajs_in_iter) if total_trajs_in_iter > 0 else 0
#                 self.logger.log({
#                     'rollout/correctness': avg_acc,
#                     'rollout/total_trajectories': total_trajs_in_iter,
#                     'rollout/timesteps_total': float(self.timesteps),
#                     'iteration': float(iteration)
#                 }, step=iteration)
#                 pbar.set_description(
#                     f"Iter {iteration} | Buffer: {len(self.replay_buffer)} | Acc: {avg_acc:.2f}")

#             dist.barrier()

#             buffer_size = broadcast_object(len(self.replay_buffer) if is_main_process() else 0)

#             if buffer_size >= self.agent_config.exploration.learning_starts:
#                 self._update_models_distributed(iteration)

#             if is_main_process() and iteration > 0 and (iteration + 1) % self.agent_config.saving.save_interval == 0:
#                 self.save_checkpoint(iteration)

#     def _sync_local_actor(self):
#         fsdp_state_dict = self.actor.state_dict()
#         if is_main_process():
#             self.local_actor.load_state_dict(fsdp_state_dict)

#     def _collect_one_trajectory(self) -> Tuple[List[Dict], Dict]:
#         trajectory, final_info = [], {}
#         obs_text, env_info = self.env.reset()
#         for _ in range(self.agent_config.env.max_steps_per_episode):
#             state_tokens = self.tokenizer(obs_text, return_tensors="pt").to(self.device)
#             sampling_params = {'max_new_tokens': 2048, 'do_sample': True, 'temperature': 0.9, 'top_p': 0.95}
#             action_ids, behavior_log_prob = self.local_actor.generate(state_tokens['input_ids'],
#                                                                       state_tokens['attention_mask'], sampling_params)
#             action_text = self.tokenizer.decode(action_ids[0], skip_special_tokens=True)
#             next_obs_text, _, terminated, truncated, step_info = self.env.step(action_text)
#             done = terminated or truncated
#             trajectory.append({'state_text': obs_text, 'action_ids': action_ids[0].cpu(),
#                                'behavior_log_prob': behavior_log_prob[0].cpu(),
#                                'external_reward': 1.0 if step_info.get('is_correct') else 0.0, 'metadata': env_info,
#                                'done': done})
#             obs_text = next_obs_text
#             if done:
#                 final_info = step_info
#                 break
#         return trajectory, final_info

#     def _collect_rollouts(self) -> List[List[Dict]]:
#         self.local_actor.eval()

#         total_rollouts = self.agent_config.exploration.rollouts_per_iteration
#         rollouts_per_rank = total_rollouts // self.world_size
#         if self.rank < total_rollouts % self.world_size:
#             rollouts_per_rank += 1

#         local_trajectories = []
#         for _ in range(rollouts_per_rank):
#             traj, _ = self._collect_one_trajectory()
#             if traj:
#                 local_trajectories.append(traj)

#         return local_trajectories

#     def _process_single_traj(self, collated, trajectory, final_reward, pass_rate, achieved_state_idx=None):
#         current_traj = trajectory[:achieved_state_idx + 1] if achieved_state_idx is not None else trajectory
#         if not current_traj: return
#         full_texts = [t['state_text'] + self.tokenizer.decode(t['action_ids'], skip_special_tokens=True) for t in
#                       current_traj]
#         dones = torch.tensor([float(t['done']) for t in current_traj], device=self.device)
#         if achieved_state_idx is not None: dones[-1] = 1.0
#         with torch.no_grad():
#             tokenized = self.tokenizer(full_texts, padding='longest', truncation=True, max_length=1024,
#                                        return_tensors="pt").to(self.device)
#             values = self.critic(tokenized.input_ids, tokenized.attention_mask).squeeze(-1)
#             rewards = torch.zeros_like(tokenized.input_ids, dtype=torch.float, device=self.device)
#             if self.use_trm: rewards += self.token_reward_model.module(tokenized.input_ids,
#                                                                        tokenized.attention_mask).squeeze(-1)
#             extrinsic_reward = 1.0 if achieved_state_idx is not None else (
#                 final_reward if final_reward == 1.0 else -pass_rate)
#             rewards[-1, (tokenized.attention_mask[-1] == 1).sum() - 1] += extrinsic_reward
#             advantages, returns = compute_gae(rewards, values, dones, tokenized.attention_mask,
#                                               self.agent_config.algorithm.gamma, self.agent_config.algorithm.tau_gae)
#         collated['full_text'].extend(full_texts)
#         collated['prompt_text'].extend([t['state_text'] for t in current_traj])
#         collated['behavior_log_prob'].extend([t['behavior_log_prob'] for t in current_traj])
#         collated['advantages'].append(advantages)
#         collated['returns'].append(returns)
#         collated['outcome_labels'].extend([final_reward] * len(current_traj))

#     def _collate_rl_batch(self, trajectories: List[List[Dict]]) -> Dict[str, torch.Tensor]:
#         collated = defaultdict(list)
#         her_k = self.agent_config.buffer.her_k_relabel
#         for traj in trajectories:
#             self._process_single_traj(collated, traj, traj[-1]['external_reward'],
#                                       traj[0]['metadata'].get('pass_rate', 0.5))
#             if self.agent_config.buffer.use_her and her_k > 0 and len(traj) > 1:
#                 for idx in np.random.randint(0, len(traj), size=her_k): self._process_single_traj(collated, traj, 1.0,
#                                                                                                   0.5,
#                                                                                                   achieved_state_idx=idx)
#         padded = {}
#         tokenized = self.tokenizer(collated['full_text'], padding='longest', truncation=True, max_length=8192,
#                                    return_tensors='pt')
#         padded['input_ids'], padded['attention_mask'] = tokenized['input_ids'], tokenized['attention_mask']
#         prompt_lens = [len(ids) for ids in
#                        self.tokenizer(collated['prompt_text'], padding='longest', truncation=True, max_length=4096)[
#                            'input_ids']]
#         labels = padded['input_ids'].clone()
#         for i, p_len in enumerate(prompt_lens): labels[i, :p_len] = -100
#         labels[padded['input_ids'] == self.tokenizer.pad_token_id] = -100
#         padded['labels'] = labels
#         for k in ['behavior_log_prob', 'outcome_labels']: padded[k] = torch.tensor(collated[k], dtype=torch.float32)
#         for k in ['advantages', 'returns']: padded[k] = torch.nn.utils.rnn.pad_sequence(collated[k], batch_first=True,
#                                                                                         padding_value=0)
#         return padded

#     def _collate_bc_batch(self, trajectories: List[List[Dict]]) -> Dict[str, torch.Tensor]:
#         full_texts, prompts_text = [], [t[0]['state_text'] for t in trajectories]
#         for traj in trajectories:
#             full_text = traj[0]['state_text']
#             full_text += "".join([self.tokenizer.decode(t['action_ids'], skip_special_tokens=True) for t in traj])
#             full_texts.append(full_text + self.tokenizer.eos_token)
#         tokenized = self.tokenizer(full_texts, padding='longest', truncation=True, max_length=8192, return_tensors='pt')
#         labels = tokenized['input_ids'].clone()
#         prompt_lens = [len(ids) for ids in
#                        self.tokenizer(prompts_text, padding='longest', truncation=True, max_length=4096)['input_ids']]
#         for i, p_len in enumerate(prompt_lens): labels[i, :p_len] = -100
#         labels[tokenized['input_ids'] == self.tokenizer.pad_token_id] = -100
#         return {"input_ids": tokenized.input_ids, "attention_mask": tokenized.attention_mask, "labels": labels}

#     def _train_trm_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
#         self.trm_optimizer.zero_grad()
#         token_rewards = self.token_reward_model.module(batch['input_ids'], batch['attention_mask']).squeeze(-1)
#         masked_rewards = token_rewards * batch['attention_mask']
#         seq_logits = torch.sum(masked_rewards, dim=-1) / (torch.sum(batch['attention_mask'], dim=-1) + 1e-8)
#         loss = F.binary_cross_entropy_with_logits(seq_logits, batch['outcome_labels'].float())
#         loss.backward()
#         self.trm_optimizer.step()
#         return loss

#     def _update_models_distributed(self, iteration: int):
#         self.actor.train()
#         self.critic.train()
#         if self.use_trm: self.token_reward_model.train()
#         self.ppo_trainer.update_old_policy()
#         for epoch in range(self.agent_config.algorithm.ppo_epochs):
#             log_dict = defaultdict(list)
#             sampled_data = self.replay_buffer.sample_trajectories(
#                 self.agent_config.algorithm.batch_size) if is_main_process() else None
#             sampled_data = broadcast_object(sampled_data)
#             if not sampled_data: continue
#             tree_indices, trajectories, is_weights = sampled_data
#             rl_batch = self._collate_rl_batch(trajectories)
#             rl_batch['tree_indices'], rl_batch['weights'] = torch.tensor(tree_indices, dtype=torch.long), torch.tensor(
#                 is_weights, dtype=torch.float32)
#             bc_trajs = self.replay_buffer.sample_for_bc(
#                 self.agent_config.algorithm.batch_size) if is_main_process() else None
#             bc_trajs = broadcast_object(bc_trajs)
#             rl_dataset = TensorDataset(*[v for _, v in sorted(rl_batch.items())])
#             rl_dataloader = DataLoader(rl_dataset, sampler=DistributedSampler(rl_dataset, num_replicas=self.world_size,
#                                                                               rank=self.rank, drop_last=True),
#                                        batch_size=self.agent_config.algorithm.batch_size // self.world_size)
#             bc_dataloader = None
#             if bc_trajs and len(bc_trajs) > 0:
#                 bc_batch_full = self._collate_bc_batch(bc_trajs)
#                 bc_dataset = TensorDataset(*[v for _, v in sorted(bc_batch_full.items())])
#                 bc_dataloader = DataLoader(bc_dataset,
#                                            sampler=DistributedSampler(bc_dataset, num_replicas=self.world_size,
#                                                                       rank=self.rank, drop_last=True),
#                                            batch_size=max(1, (
#                                                    self.agent_config.algorithm.batch_size // self.world_size) // 4))

#             if bc_dataloader:
#                 loop_iterator = zip(rl_dataloader, bc_dataloader)
#                 process_bc = True
#             else:
#                 loop_iterator = rl_dataloader
#                 process_bc = False

#             for batch_data in loop_iterator:
#                 if process_bc:
#                     micro_rl_batch_tensors, micro_bc_batch_tensors = batch_data
#                     micro_bc_batch = {k: v.to(self.device) for k, v in
#                                       zip(sorted(bc_batch_full.keys()), micro_bc_batch_tensors)}
#                 else:
#                     micro_rl_batch_tensors = batch_data

#                 micro_rl_batch = {k: v.to(self.device) for k, v in zip(sorted(rl_batch.keys()), micro_rl_batch_tensors)}

#                 if self.use_trm: log_dict['loss/trm'].append(self._train_trm_step(micro_rl_batch).item())

#                 ppo_log, ppo_loss = self.ppo_trainer.train_step(micro_rl_batch, return_loss=True)
#                 for k, v in ppo_log.items(): log_dict[k].append(v)
#                 total_loss = ppo_loss

#                 if process_bc:
#                     bc_loss = self.actor.module.forward_sft(micro_bc_batch['input_ids'],
#                                                             micro_bc_batch['attention_mask'],
#                                                             micro_bc_batch['labels'])
#                     log_dict['loss/bc'].append(bc_loss.item())
#                     total_loss += self.agent_config.algorithm.bc_loss_weight * bc_loss

#                 self.ac_optimizer.zero_grad()
#                 total_loss.backward()
#                 self.ac_optimizer.step()

#                 if self.replay_buffer and self.replay_buffer.use_per:
#                     with torch.no_grad():
#                         new_values = self.critic(micro_rl_batch['input_ids'], micro_rl_batch['attention_mask']).squeeze(
#                             -1)
#                         td_errors = micro_rl_batch['returns'].sum(dim=-1) - new_values
#                     self.replay_buffer.update_priorities(micro_rl_batch['tree_indices'], td_errors)

#             if is_main_process():
#                 final_log = {k: np.mean(v) for k, v in log_dict.items() if v}
#                 final_log['epoch'] = float(epoch)
#                 self.logger.log(final_log, step=iteration * self.agent_config.algorithm.ppo_epochs + epoch)

#     def save_checkpoint(self, iteration: int):
#         checkpoint_dir = os.path.join(self.agent_config.saving.checkpoint_dir, f"iter_{iteration}")
#         if is_main_process(): print(f"\nSaving distributed checkpoint to {checkpoint_dir}...")

#         state_dict = {"actor": self.actor, "critic": self.critic, "old_actor": self.old_actor,
#                       "ac_optimizer": self.ac_optimizer}
#         if self.use_trm: state_dict["trm"] = self.token_reward_model; state_dict["trm_optimizer"] = self.trm_optimizer

#         writer = FileSystemWriter(checkpoint_dir)
#         save(state_dict=state_dict, storage_writer=writer)

#         if is_main_process():
#             self.replay_buffer.save(os.path.join(checkpoint_dir, "replay_buffer.pkl"))
#             metadata = {'iteration': iteration, 'timesteps': self.timesteps}
#             with open(os.path.join(checkpoint_dir, "metadata.json"), 'w') as f: json.dump(metadata, f)
#         dist.barrier(device_ids=[self.rank])
#         if is_main_process(): print(f"✅ Checkpoint saved successfully.")

#     def load_checkpoint(self):
#         checkpoint_dir = self.agent_config.saving.checkpoint_dir
#         if not os.path.isdir(checkpoint_dir):
#             if is_main_process(): print("No checkpoint directory found, starting from scratch.")
#             return

#         dirs = [d for d in os.listdir(checkpoint_dir) if
#                 d.startswith("iter_") and os.path.isdir(os.path.join(checkpoint_dir, d))]
#         if not dirs:
#             if is_main_process(): print("No iteration checkpoints found, starting from scratch.")
#             return

#         latest_iter = max([int(d.split('_')[1]) for d in dirs])
#         latest_ckpt_path = os.path.join(checkpoint_dir, f"iter_{latest_iter}")
#         if is_main_process(): print(f"Resuming training from checkpoint: {latest_ckpt_path}")

#         state_dict = {"actor": self.actor, "critic": self.critic, "old_actor": self.old_actor,
#                       "ac_optimizer": self.ac_optimizer}
#         if self.use_trm: state_dict["trm"] = self.token_reward_model; state_dict["trm_optimizer"] = self.trm_optimizer

#         reader = FileSystemReader(latest_ckpt_path)
#         load(state_dict=state_dict, storage_reader=reader)

#         if is_main_process():
#             self.replay_buffer.load(os.path.join(latest_ckpt_path, "replay_buffer.pkl"))
#             with open(os.path.join(latest_ckpt_path, "metadata.json"), 'r') as f:
#                 metadata = json.load(f)
#                 self.start_iteration = metadata['iteration'] + 1
#                 self.timesteps = metadata['timesteps']

#         self.start_iteration = broadcast_object(self.start_iteration if is_main_process() else 0)
#         self.timesteps = broadcast_object(self.timesteps if is_main_process() else 0)
#         dist.barrier()
#         if is_main_process(): print(
#             f"Resumed from iteration {self.start_iteration - 1}. Current timesteps: {self.timesteps}")

# src/trainers/sft_trainer.py (FINAL & OPTIMIZED)

import os
import functools
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.fsdp import FullStateDictConfig
from peft import get_peft_model, LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, PreTrainedModel
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from functools import partial

from src.datasets.sft_dataset import SFTDataset, collate_fn_sft
from src.utils.logger import SwanLabLogger
from src.utils.distributed_utils import is_main_process, get_rank


class SFTTrainer:
    """
    [FINAL & OPTIMIZED] SFT Trainer.
    - Integrated with torch.compile, Flash Attention 2, Fused Optimizers, and DataLoader optimizations.
    """

    def __init__(self, config: DictConfig, rank: int, world_size: int, logger: SwanLabLogger):
        self.config = config
        self.sft_config = config.trainer
        self.model_config = config.model
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")
        self.logger = logger
        model_path = self.sft_config.get("model_path", self.model_config.path)
        use_lora = self.sft_config.get("use_lora", self.model_config.get("use_lora", False))
        if is_main_process(): print(f"SFT Trainer using model path: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token

        # OPTIMIZATION 1: Use Flash Attention 2 if available
        model_kwargs = {'trust_remote_code': True}
        if config.get("use_flash_attention_2", True):
            model_kwargs['attn_implementation'] = "flash_attention_2"
            if is_main_process(): print("Attempting to enable Flash Attention 2...")

        if is_main_process(): print(f"Loading base model from {model_path} for SFT...")
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, **model_kwargs)
        model.gradient_checkpointing_enable()

        if use_lora:
            if is_main_process(): print("Applying LoRA...")
            if not hasattr(self.model_config, 'lora_config'): raise ValueError(
                "`use_lora` is true, but `lora_config` is not defined in the model configuration.")
            lora_config_dict = OmegaConf.to_container(self.model_config.lora_config, resolve=True)
            lora_config = LoraConfig(**lora_config_dict)
            model = get_peft_model(model, lora_config)
            if is_main_process(): model.print_trainable_parameters()

        # OPTIMIZATION 2: Use torch.compile
        if self.config.get("use_torch_compile", True):
            if is_main_process(): print("Applying torch.compile to the SFT model...")
            try:
                model = torch.compile(model)
            except Exception as e:
                if is_main_process(): print(f"Warning: torch.compile failed, proceeding without it. Error: {e}")

        try:
            transformer_block_class = self._get_transformer_layer_class(model)
            if is_main_process(): print(
                f"Found transformer layer class to wrap with FSDP: {transformer_block_class.__name__}")
        except ValueError as e:
            print(f"Error: Could not automatically determine the transformer layer class for FSDP wrapping. {e}")
            if dist.is_initialized(): dist.destroy_process_group()
            exit(1)

        transformer_layer_cls_set = {transformer_block_class}
        auto_wrap_policy = functools.partial(transformer_auto_wrap_policy,
                                             transformer_layer_cls=transformer_layer_cls_set)
        use_orig_params_for_fsdp = use_lora
        fsdp_dtype = torch.bfloat16
        model = model.to(dtype=fsdp_dtype)
        mixed_precision_policy = torch.distributed.fsdp.MixedPrecision(param_dtype=fsdp_dtype, reduce_dtype=fsdp_dtype,
                                                                       buffer_dtype=fsdp_dtype)
        self.model = FSDP(model, auto_wrap_policy=auto_wrap_policy, device_id=self.device,
                          use_orig_params=use_orig_params_for_fsdp, mixed_precision=mixed_precision_policy)
        dist.barrier(device_ids=[self.rank])
        if is_main_process(): print("✅ FSDP model wrapped successfully.")

        train_dataset = SFTDataset(self.sft_config.sft_data_path, self.tokenizer, max_length=self.sft_config.max_length)
        train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        collate_with_tokenizer = partial(collate_fn_sft, tokenizer=self.tokenizer)

        # OPTIMIZATION 3: Improve DataLoader performance
        dl_kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.sft_config.batch_size_per_gpu,
                                           sampler=train_sampler, collate_fn=collate_with_tokenizer, **dl_kwargs)

        # OPTIMIZATION 4: Use Fused AdamW
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.sft_config.learning_rate, fused=True)

        num_training_steps = self.sft_config.epochs * len(self.train_dataloader)
        self.lr_scheduler = get_scheduler(name="cosine", optimizer=self.optimizer,
                                          num_warmup_steps=int(0.03 * num_training_steps),
                                          num_training_steps=num_training_steps)

    @staticmethod
    def _get_transformer_layer_class(model: PreTrainedModel) -> type:
        # Correctly handle compiled model
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
        if isinstance(model, PeftModel):
            model = model.get_base_model()

        found_class = None
        for name, module in model.named_modules():
            if "DecoderLayer" in module.__class__.__name__ or "TransformerBlock" in module.__class__.__name__:
                found_class = module.__class__
                break
        if found_class is None: raise ValueError("Could not find a valid transformer layer class name.")
        return found_class

    def train(self):
        self.model.train()
        total_steps = 0
        for epoch in range(self.sft_config.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler): self.train_dataloader.sampler.set_epoch(
                epoch)
            pbar = tqdm(self.train_dataloader, disable=(not is_main_process()),
                        desc=f"SFT Epoch {epoch + 1}/{self.sft_config.epochs} | Rank {get_rank()}")
            for batch in pbar:
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                if is_main_process():
                    pbar.set_postfix({"loss": loss.item()})
                    self.logger.log({"sft/loss": loss.item(), "sft/lr": self.lr_scheduler.get_last_lr()[0]},
                                    step=total_steps)
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                total_steps += 1
            dist.barrier(device_ids=[self.rank])
        self.save_model()

    def save_model(self):
        output_dir = self.sft_config.output_dir
        if is_main_process():
            print(f"Starting model saving process to {output_dir}...")
            os.makedirs(output_dir, exist_ok=True)

        dist.barrier()

        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
            unwrapped_model = self.model.module

            if is_main_process():
                print("Model state dict collected on Rank 0. Now saving to disk...")
                use_lora = self.sft_config.get("use_lora", self.model_config.get("use_lora", False))

                # Correctly unwrap the compiled model before saving
                if hasattr(unwrapped_model, '_orig_mod'):
                    unwrapped_model = unwrapped_model._orig_mod

                if use_lora and isinstance(unwrapped_model, PeftModel):
                    unwrapped_model.save_pretrained(output_dir)
                    print("LoRA adapter saved using `save_pretrained`.")
                else:
                    cpu_state_dict = unwrapped_model.state_dict()
                    torch.save(cpu_state_dict, os.path.join(output_dir, "pytorch_model.bin"))
                    print("Full model state dict saved.")

                self.tokenizer.save_pretrained(output_dir)
                print("Tokenizer saved.")
                print(f"✅ SFT model saved successfully to {output_dir}.")

        # dist.barrier()