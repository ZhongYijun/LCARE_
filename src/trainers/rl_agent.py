# # src/trainers/rl_agent.py (最终的、完整的、釜底抽薪的稳健版本)
#
# import os
# import torch
# import torch.distributed as dist
# from torch.optim import AdamW
# from torch.nn.parallel import DistributedDataParallel as DDP
# from omegaconf import DictConfig
# from tqdm import trange
# from collections import defaultdict
# from typing import Dict, List, Any, Optional, Tuple
# import numpy as np
# import json
# from copy import deepcopy
#
# from transformers import AutoTokenizer, AutoConfig
#
#
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
#
#
# class LCARE_Agent:
#     def __init__(self, config: DictConfig, rank: int, world_size: int, logger: SwanLabLogger):
#         self.config, self.agent_config, self.model_config = config, config.trainer, config.model
#         self.rank, self.world_size, self.device, self.logger = rank, world_size, torch.device(f"cuda:{rank}"), logger
#         self.verifier = Verifier(config.verifier)
#         self.use_lora = self.model_config.get("use_lora", False)
#         self.use_trm = self.agent_config.exploration.get("use_token_reward_model", False)
#
#         # 在初始化时就检查任务分配的对称性
#         if self.agent_config.exploration.rollouts_per_iteration % self.world_size != 0:
#             raise ValueError(
#                 f"CRITICAL ERROR: `rollouts_per_iteration` ({self.agent_config.exploration.rollouts_per_iteration}) "
#                 f"must be divisible by `world_size` ({self.world_size}) to ensure symmetric workload.")
#         if not self.use_lora: raise NotImplementedError("此Agent专为LoRA训练设计。")
#
#         self.tokenizer = AutoTokenizer.from_pretrained(config.model.path, trust_remote_code=True, padding_side='left')
#         model_generation_config = AutoConfig.from_pretrained(config.model.path, trust_remote_code=True)
#         self.eos_token_ids = getattr(model_generation_config, 'eos_token_id', self.tokenizer.eos_token_id)
#         if not isinstance(self.eos_token_ids, list): self.eos_token_ids = [self.eos_token_ids]
#
#         if self.tokenizer.pad_token_id is None:
#             pad_token_id_from_config = getattr(model_generation_config, 'pad_token_id', self.eos_token_ids[0])
#             if is_main_process(): print(
#                 f"⚠️ WARNING: Tokenizer's `pad_token_id` not set. Setting to `{pad_token_id_from_config}`.")
#             self.tokenizer.pad_token_id = pad_token_id_from_config
#             self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(self.tokenizer.pad_token_id)
#
#         if is_main_process(): print(
#             f"✅ EOS tokens: {self.eos_token_ids}, PAD token: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")
#
#         self.prompt_constructor = PromptConstructor(config, self.tokenizer)
#
#         actor_cpu_template = LCARE_Actor(self.model_config, self.tokenizer)
#         if os.path.isdir(self.agent_config.initial_policy_path):
#             if is_main_process(): print(f"Loading initial LoRA adapter from {self.agent_config.initial_policy_path}")
#             actor_cpu_template.model.load_adapter(self.agent_config.initial_policy_path, "default")
#
#         self.local_actor = deepcopy(actor_cpu_template).to(self.device).eval()
#         actor_train = deepcopy(actor_cpu_template).to(self.device)
#         critic_train = LCARE_Critic(config.model.critic).to(self.device)
#
#         self.actor = DDP(actor_train, device_ids=[self.rank])
#         self.critic = DDP(critic_train, device_ids=[self.rank])
#
#         self.token_reward_model, self.trm_optimizer = None, None
#         if self.use_trm:
#             trm_train = LCARE_TokenRewardModel(config.model.token_reward_model).to(self.device)
#             self.token_reward_model = DDP(trm_train, device_ids=[self.rank])
#             self.token_reward_model.module.load_state_dict(self.actor.module.state_dict(), strict=False)
#
#         dist.barrier()
#         if is_main_process(): print("✅ 所有模型已成功使用DDP包装。")
#
#         self.replay_buffer, self.encoder = None, None
#         if is_main_process():
#             self.replay_buffer = LCAREReplayBuffer(self.agent_config.buffer, self.agent_config.exploration,
#                                                    actor_cpu_template.model.config.hidden_size)
#             self.encoder = LGE_Encoder(actor_cpu_template.model.get_base_model(), self.tokenizer, self.device)
#
#         prompt_path = os.path.join(config.data.processed_dir, config.data.rl_prompt_file)
#         self.prompt_set = list(RLPromptDataset(prompt_path)) if os.path.exists(prompt_path) else []
#         self.env = MathReasoningEnv(self.prompt_set, self.verifier, self.prompt_constructor,
#                                     self.agent_config.env.max_steps_per_episode, self.config)
#
#         params_to_optimize = list(self.actor.parameters()) + list(self.critic.parameters())
#         self.ac_optimizer = AdamW(params_to_optimize, lr=self.agent_config.algorithm.learning_rate,
#                                   fused=torch.cuda.is_available())
#         if self.use_trm: self.trm_optimizer = AdamW(self.token_reward_model.parameters(),
#                                                     lr=self.agent_config.algorithm.trm_learning_rate,
#                                                     fused=torch.cuda.is_available())
#
#         self.ppo_trainer = OffPolicyPPO_Trainer(self.actor, self.critic, self.ac_optimizer, self.agent_config.algorithm,
#                                                 self.tokenizer, self.rank, self.world_size)
#         self.timesteps, self.start_iteration = 0, 0
#         self.load_checkpoint()
#
#     def learn(self):
#         pbar = trange(self.start_iteration, self.agent_config.exploration.total_iterations,
#                       disable=not is_main_process(), desc="RL训练")
#         for iteration in pbar:
#             self.local_actor.load_state_dict(self.actor.module.state_dict())
#
#             # [终极架构] 数据采集、填充、通信的全新流程
#             trajectories = self._collect_rollouts()
#
#             gathered_trajs = [None] * self.world_size
#             dist.all_gather_object(gathered_trajs, trajectories)
#
#             if is_main_process():
#                 flat_trajs = [t for rank_trajs in gathered_trajs for t in rank_trajs]
#                 self._process_and_store_rollouts(flat_trajs, iteration, pbar)
#
#             dist.barrier()
#             buffer_size = broadcast_object(len(self.replay_buffer) if is_main_process() else 0)
#
#             if buffer_size >= self.agent_config.exploration.learning_starts:
#                 self._update_models_distributed(iteration)
#
#             if is_main_process() and (iteration + 1) % self.agent_config.saving.save_interval == 0:
#                 self.save_checkpoint(iteration)
#
#     def _collect_rollouts(self) -> List[List[Dict]]:
#         rollouts_per_rank = self.agent_config.exploration.rollouts_per_iteration // self.world_size
#         return [self._collect_one_trajectory() for _ in range(rollouts_per_rank)]
#
#     def _collect_one_trajectory(self) -> List[Dict]:
#         trajectory, (obs_text, env_info) = [], self.env.reset()
#         for _ in range(self.agent_config.env.max_steps_per_episode):
#             state_tokens = self.tokenizer(obs_text, return_tensors="pt", truncation=True, max_length=2048).to(
#                 self.device)
#             with torch.no_grad():
#                 sampling_params = {'max_new_tokens': 1024, 'do_sample': True, 'temperature': 0.9, 'top_p': 0.95,
#                                    'eos_token_id': self.eos_token_ids}
#                 action_ids, behavior_log_prob = self.local_actor.generate(state_tokens['input_ids'],
#                                                                           state_tokens['attention_mask'],
#                                                                           sampling_params)
#             action_text = self.tokenizer.decode(action_ids[0], skip_special_tokens=True)
#             next_obs_text, _, terminated, truncated, step_info = self.env.step(action_text)
#             done = terminated or truncated
#             trajectory.append({'state_text': obs_text, 'action_ids': action_ids[0].cpu(),
#                                'behavior_log_prob': behavior_log_prob.cpu(),
#                                'external_reward': 1.0 if step_info.get('is_correct') else 0.0, 'metadata': env_info,
#                                'done': done})
#             obs_text = next_obs_text
#             if done: break
#         return trajectory
#
#     def _process_and_store_rollouts(self, trajectories: List[List[Dict]], iteration: int, pbar: trange):
#         if not is_main_process() or not self.replay_buffer: return
#         total_trajs, correct_trajs = 0, 0
#         for traj in trajectories:
#             if not traj: continue
#             if self.agent_config.exploration.get("use_lge", False):
#                 traj = self._compute_and_add_intrinsic_rewards(traj)
#             self.replay_buffer.add_trajectory(traj)
#             self.timesteps += len(traj)
#             total_trajs += 1
#             if traj[-1]['external_reward'] == 1.0: correct_trajs += 1
#         avg_acc = (correct_trajs / total_trajs) if total_trajs > 0 else 0
#         self.logger.log({'rollout/correctness': avg_acc, 'rollout/total_trajectories': float(total_trajs),
#                          'rollout/timesteps_total': float(self.timesteps)}, step=iteration)
#         pbar.set_description(f"Iter {iteration} | Buffer: {len(self.replay_buffer)} | Acc: {avg_acc:.2f}")
#
#     def _update_models_distributed(self, iteration: int):
#         self.actor.train()
#         self.critic.train()
#         if self.use_trm and self.token_reward_model: self.token_reward_model.train()
#         for epoch in range(self.agent_config.algorithm.ppo_epochs):
#             data_chunk = self._sample_and_scatter_batch()
#             if not data_chunk or not data_chunk.get('trajs'): continue
#
#             batch = self._collate_and_compute_rewards(data_chunk)
#             if not batch: continue
#
#             log_dict, _ = self._perform_train_step(batch)
#             self._update_per_priorities(batch)
#             if is_main_process():
#                 final_log = {k: float(np.mean(v)) for k, v in log_dict.items() if v}
#                 self.logger.log(final_log, step=iteration * self.agent_config.algorithm.ppo_epochs + epoch)
#
#     def _sample_and_scatter_batch(self) -> Optional[Dict[str, Any]]:
#         batch_size = self.agent_config.algorithm.batch_size
#         local_bs = batch_size // self.world_size
#         data_to_scatter = [None] * self.world_size
#         if is_main_process() and self.replay_buffer and len(self.replay_buffer) >= batch_size:
#             sampled_data = self.replay_buffer.sample_trajectories(batch_size)
#             if sampled_data:
#                 tree_indices, trajectories, is_weights = sampled_data
#                 for i in range(self.world_size):
#                     start, end = i * local_bs, (i + 1) * local_bs
#                     data_to_scatter[i] = {'trajs': trajectories[start:end], 'indices': tree_indices[start:end],
#                                           'weights': is_weights[start:end]}
#         scattered_data = broadcast_object(data_to_scatter)
#         return scattered_data[self.rank] if scattered_data else None
#
#     def _collate_and_compute_rewards(self, data_chunk: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
#         trajectories = data_chunk['trajs']
#         if not trajectories: return None
#
#         collated = defaultdict(list)
#         use_her, her_k = self.agent_config.buffer.use_her, self.agent_config.buffer.her_k_relabel
#         for traj in trajectories:
#             self._process_single_traj_for_batching(collated, traj, traj[-1]['external_reward'],
#                                                    traj[0]['metadata'].get('pass_rate', 0.5))
#             if use_her and her_k > 0 and traj[-1]['external_reward'] < 1.0 and len(traj) > 1:
#                 for _ in range(her_k):
#                     split_point = np.random.randint(1, len(traj))
#                     self._process_single_traj_for_batching(collated, traj[:split_point], 1.0, 0.5, is_her=True)
#
#         if not collated['prompts']: return None
#
#         # 批量Tokenize
#         prompt_tok = self.tokenizer(collated['prompts'], padding=True, truncation=True, max_length=1024,
#                                     return_tensors='pt')
#         action_tok = self.tokenizer.pad({'input_ids': collated['actions']}, padding=True, return_tensors='pt')
#
#         input_ids = torch.cat([prompt_tok.input_ids, action_tok.input_ids], dim=1).to(self.device)
#         attention_mask = torch.cat([prompt_tok.attention_mask, action_tok.attention_mask], dim=1).to(self.device)
#
#         labels = input_ids.clone()
#         labels[:, :prompt_tok.input_ids.shape[1]] = -100
#         labels[labels == self.tokenizer.pad_token_id] = -100
#
#         # 批量计算Values和Rewards
#         with torch.no_grad():
#             values = self.critic.module(input_ids, attention_mask).squeeze(-1)
#             rewards = torch.zeros_like(values)
#             if self.use_trm and self.token_reward_model:
#                 token_rewards = self.token_reward_model.module(input_ids, attention_mask).squeeze(-1)
#                 sequence_lengths = torch.sum(attention_mask, dim=1) - 1
#                 rewards = token_rewards[torch.arange(len(token_rewards)), sequence_lengths]
#
#         advantages, returns = [], []
#         start_idx = 0
#         for i in range(len(collated['seq_lens'])):
#             seq_len, final_reward, pass_rate, is_her = collated['seq_lens'][i], collated['final_rewards'][i], \
#             collated['pass_rates'][i], collated['is_her'][i]
#             end_idx = start_idx + seq_len
#
#             seq_rewards = rewards[start_idx:end_idx].clone()
#             extrinsic_reward = 1.0 if is_her else (final_reward if final_reward == 1.0 else -pass_rate)
#             seq_rewards[-1] += extrinsic_reward
#
#             seq_dones = torch.zeros_like(seq_rewards)
#             seq_dones[-1] = 1.0
#
#             response_mask_bool = torch.zeros_like(attention_mask[start_idx:end_idx], dtype=torch.bool)
#             response_mask_bool[:, prompt_tok.input_ids.shape[1]:] = action_tok.attention_mask[start_idx:end_idx] > 0
#
#             adv, ret = compute_gae(seq_rewards.cpu().numpy(), values[start_idx:end_idx].cpu().numpy(),
#                                    seq_dones.cpu().numpy(),
#                                    response_mask_bool.cpu().numpy(),
#                                    self.agent_config.algorithm.gamma, self.agent_config.algorithm.tau_gae)
#
#             advantages.append(torch.from_numpy(adv))
#             returns.append(torch.from_numpy(ret))
#             start_idx = end_idx
#
#         return {
#             'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels,
#             'advantages': torch.cat(advantages).to(self.device),
#             'returns': torch.cat(returns).to(self.device),
#             'behavior_log_probs': torch.cat(collated['logprobs']).to(self.device),
#             'tree_indices': torch.tensor(data_chunk['indices'], dtype=torch.long, device=self.device),
#             'weights': torch.tensor(data_chunk['weights'], dtype=torch.float32, device=self.device),
#             'outcome_labels': torch.tensor(collated['outcome_labels'], dtype=torch.float32).to(self.device)
#         }
#
#     def _process_single_traj_for_batching(self, collated: defaultdict, trajectory: List[Dict], final_reward: float,
#                                           pass_rate: float, is_her: bool = False):
#         collated['prompts'].extend([s['state_text'] for s in trajectory])
#         collated['actions'].extend([s['action_ids'] for s in trajectory])
#         collated['logprobs'].extend([s['behavior_log_prob'] for s in trajectory])
#         collated['seq_lens'].append(len(trajectory))
#         collated['final_rewards'].append(final_reward)
#         collated['pass_rates'].append(pass_rate)
#         collated['is_her'].append(is_her)
#         collated['outcome_labels'].extend([final_reward] * len(trajectory))
#
#     def _perform_train_step(self, batch: Dict) -> Tuple[defaultdict, Optional[torch.Tensor]]:
#         log_dict, total_loss = defaultdict(list), None
#         if self.use_trm and self.trm_optimizer and self.token_reward_model:
#             self.trm_optimizer.zero_grad()
#             trm_loss = self.token_reward_model(batch['input_ids'], batch['attention_mask'], batch['outcome_labels'])
#             trm_loss.backward()
#             self.trm_optimizer.step()
#             log_dict['loss/trm'].append(trm_loss.item())
#
#         self.ac_optimizer.zero_grad()
#         ppo_log, ppo_loss = self.ppo_trainer.train_step(batch, return_loss=True)
#         if ppo_loss is not None:
#             total_loss = ppo_loss
#             total_loss.backward()
#             self.ac_optimizer.step()
#         for k, v in ppo_log.items(): log_dict[k].append(v)
#         return log_dict, total_loss
#
#     def _update_per_priorities(self, batch: Dict):
#         if not (self.replay_buffer and self.replay_buffer.use_per): return
#         with torch.no_grad():
#             new_values = self.critic.module(batch['input_ids'], batch['attention_mask']).squeeze(-1)
#             td_errors = batch['returns'] - new_values
#         all_td_errors, all_indices = [torch.empty_like(td_errors) for _ in range(self.world_size)], [
#             torch.empty_like(batch['tree_indices']) for _ in range(self.world_size)]
#         dist.all_gather(all_td_errors, td_errors)
#         dist.all_gather(all_indices, batch['tree_indices'])
#         if is_main_process(): self.replay_buffer.update_priorities(torch.cat(all_indices), torch.cat(all_td_errors))
#
#     def _compute_and_add_intrinsic_rewards(self, trajectory: List[Dict]) -> List[Dict]:
#         if not self.encoder or not self.replay_buffer: return trajectory
#         all_states_text = [t['state_text'] for t in trajectory]
#         lge_config = self.agent_config.exploration.lge_config
#         with torch.no_grad():
#             new_latent_vectors = self.encoder.encode(all_states_text).to(torch.bfloat16)
#         for i, latent_vec in enumerate(new_latent_vectors):
#             reward = 0.0
#             if self.replay_buffer.latent_archive_size > lge_config.k_nearest_neighbors:
#                 archive_tensor = self.replay_buffer.latent_state_archive[:self.replay_buffer.latent_archive_size].to(
#                     self.device)
#                 distances = torch.norm(archive_tensor - latent_vec, dim=1)
#                 knn_distances, _ = torch.topk(distances, lge_config.k_nearest_neighbors, largest=False)
#                 reward = knn_distances.mean().item()
#             trajectory[i]['intrinsic_reward'] = reward
#             self.replay_buffer.latent_state_archive[self.replay_buffer.next_latent_idx] = latent_vec.cpu()
#             self.replay_buffer.next_latent_idx = (self.replay_buffer.next_latent_idx + 1) % lge_config.archive_capacity
#             if self.replay_buffer.latent_archive_size < lge_config.archive_capacity: self.replay_buffer.latent_archive_size += 1
#         return trajectory
#
#     def save_checkpoint(self, iteration: int):
#         if not is_main_process(): return
#         checkpoint_dir = os.path.join(self.agent_config.saving.checkpoint_dir, f"iter_{iteration}")
#         os.makedirs(checkpoint_dir, exist_ok=True)
#         print(f"\nSaving checkpoint on rank 0 to {checkpoint_dir}...")
#         self.actor.module.model.save_pretrained(checkpoint_dir)
#         torch.save(self.critic.module.state_dict(), os.path.join(checkpoint_dir, 'critic.pt'))
#         if self.use_trm and self.token_reward_model: torch.save(self.token_reward_model.module.state_dict(),
#                                                                 os.path.join(checkpoint_dir, 'trm.pt'))
#         torch.save(self.ac_optimizer.state_dict(), os.path.join(checkpoint_dir, 'ac_optimizer.pt'))
#         if self.trm_optimizer: torch.save(self.trm_optimizer.state_dict(),
#                                           os.path.join(checkpoint_dir, 'trm_optimizer.pt'))
#         if self.replay_buffer: self.replay_buffer.save(os.path.join(checkpoint_dir, "replay_buffer.pkl"))
#         metadata = {'iteration': iteration, 'timesteps': self.timesteps}
#         with open(os.path.join(checkpoint_dir, "metadata.json"), 'w') as f:
#             json.dump(metadata, f)
#         print(f"✅ Checkpoint saved.")
#
#     def load_checkpoint(self):
#         checkpoint_dir = self.agent_config.saving.checkpoint_dir
#         if not os.path.isdir(checkpoint_dir):
#             if is_main_process(): print("No checkpoint directory found, starting from scratch.")
#             return
#         iter_dirs = [d for d in os.listdir(checkpoint_dir) if
#                      d.startswith("iter_") and os.path.isdir(os.path.join(checkpoint_dir, d))]
#         if not iter_dirs:
#             if is_main_process(): print("No checkpoints found, starting from scratch.")
#             return
#         latest_iter = max(int(d.split('_')[1]) for d in iter_dirs)
#         load_path = os.path.join(checkpoint_dir, f"iter_{latest_iter}")
#         if is_main_process(): print(f"Resuming from latest checkpoint: {load_path}")
#
#         self.actor.module.model.load_adapter(load_path, "default")
#         self.critic.module.load_state_dict(torch.load(os.path.join(load_path, 'critic.pt'), map_location='cpu'))
#         if self.use_trm and self.token_reward_model and os.path.exists(os.path.join(load_path, 'trm.pt')):
#             self.token_reward_model.module.load_state_dict(
#                 torch.load(os.path.join(load_path, 'trm.pt'), map_location='cpu'))
#
#         ac_optim_state_dict = torch.load(os.path.join(load_path, 'ac_optimizer.pt'), map_location='cpu',
#                                          weights_only=True)
#         self.ac_optimizer.load_state_dict(ac_optim_state_dict)
#         if self.trm_optimizer and os.path.exists(os.path.join(load_path, 'trm_optimizer.pt')):
#             trm_optim_state_dict = torch.load(os.path.join(load_path, 'trm_optimizer.pt'), map_location='cpu',
#                                               weights_only=True)
#             self.trm_optimizer.load_state_dict(trm_optim_state_dict)
#
#         if is_main_process() and self.replay_buffer:
#             self.replay_buffer.load(os.path.join(load_path, "replay_buffer.pkl"))
#             with open(os.path.join(load_path, "metadata.json"), 'r') as f:
#                 metadata = json.load(f)
#                 self.start_iteration = metadata['iteration'] + 1
#                 self.timesteps = metadata['timesteps']
#
#         self.start_iteration = int(broadcast_object(self.start_iteration if is_main_process() else 0))
#         self.timesteps = int(broadcast_object(self.timesteps if is_main_process() else 0))
#
#         dist.barrier()
#         if is_main_process(): print(
#             f"✅ Successfully resumed from iteration {self.start_iteration - 1}. Current timesteps: {self.timesteps}")

# # src/trainers/rl_agent.py
# # [L-CARE V15-SYNERGY - FINAL IDEAL FORM]
#
# import os
# import torch
# import torch.distributed as dist
# from torch.optim import AdamW
# from torch.nn.parallel import DistributedDataParallel as DDP
# from omegaconf import DictConfig, OmegaConf
# from tqdm import trange
# from collections import defaultdict
# from typing import Dict, List, Any, Optional, Tuple
# import numpy as np
# import json
# from copy import deepcopy
# import logging
# import atexit
# from multiprocessing import Process
#
# from transformers import AutoTokenizer, AutoConfig
# from peft import PeftModel
#
# # Local project imports
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
# from evaluate import run_evaluation as run_evaluation_process
#
# # Setup a dedicated logger for this module
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
#
# class LCARE_Agent:
#     """
#     [V15-SYNERGY] The definitive, synergistic L-CARE Agent.
#     - Fuses V14's synergistic algorithms (LGE-Driven-Explore, Hybrid-Priority-PER, Difficulty-Aware-HER)
#       with V15.2's robust engineering (Memory-Safe Loading, Async-Eval).
#     - This is the final, ideal form of the L-CARE project.
#     """
#
#     def __init__(self, config: DictConfig, rank: int, world_size: int, swanlab_logger: SwanLabLogger):
#         self.config = config
#         self.agent_config = config.trainer
#         self.model_config = config.model
#         self.rank = rank
#         self.world_size = world_size
#         self.device = torch.device(f"cuda:{rank}")
#         self.swanlab_logger = swanlab_logger
#
#         self.verifier = Verifier(config.verifier)
#         self.use_lora = self.model_config.get("use_lora", True)
#         self.use_trm = self.agent_config.exploration.get("use_token_reward_model", False)
#         self.use_lge = self.agent_config.exploration.get("use_lge", False)
#
#         if self.agent_config.exploration.rollouts_per_iteration % self.world_size != 0:
#             raise ValueError(
#                 f"Config Error: `rollouts_per_iteration` ({self.agent_config.exploration.rollouts_per_iteration}) must be divisible by `world_size` ({self.world_size}).")
#         if not self.use_lora:
#             raise NotImplementedError("This Agent is specifically designed for LoRA training.")
#
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.path, trust_remote_code=True,
#                                                        padding_side='left')
#         self._setup_tokenizer()
#         self.prompt_constructor = PromptConstructor(config, self.tokenizer)
#
#         with torch.device("cpu"):
#             actor_cpu_template = self._initialize_actor_template()
#
#         self.local_actor = deepcopy(actor_cpu_template).to(self.device).eval()
#
#         actor_train = deepcopy(actor_cpu_template).to(self.device)
#         critic_train = LCARE_Critic(config.model.critic).to(self.device)
#         self.actor = DDP(actor_train, device_ids=[self.rank], find_unused_parameters=True)
#         self.critic = DDP(critic_train, device_ids=[self.rank], find_unused_parameters=True)
#
#         self.token_reward_model, self.trm_optimizer = None, None
#         if self.use_trm:
#             trm_train = LCARE_TokenRewardModel(config.model.token_reward_model).to(self.device)
#             self.token_reward_model = DDP(trm_train, device_ids=[self.rank], find_unused_parameters=True)
#
#         dist.barrier()
#         if is_main_process(): logger.info("✅ Training models wrapped with DDP.")
#
#         prompt_path = os.path.join(config.data.processed_dir, config.data.rl_prompt_file)
#         prompt_set = list(RLPromptDataset(prompt_path)) if os.path.exists(prompt_path) else []
#         self.env = MathReasoningEnv(prompt_set, self.verifier, self.prompt_constructor,
#                                     self.agent_config.env.max_steps_per_episode, self.config)
#
#         self.replay_buffer, self.encoder = None, None
#         if is_main_process():
#             self.replay_buffer = LCAREReplayBuffer(self.agent_config.buffer, self.agent_config.exploration,
#                                                    actor_cpu_template.model.config.hidden_size)
#             if self.agent_config.exploration.get("use_lge", False):
#                 base_model = actor_cpu_template.model.get_base_model() if self.use_lora else actor_cpu_template.model
#                 self.encoder = LGE_Encoder(base_model, self.tokenizer, self.device)
#
#         params_to_optimize = list(self.actor.parameters()) + list(self.critic.parameters())
#         self.ac_optimizer = AdamW(params_to_optimize, lr=self.agent_config.algorithm.learning_rate,
#                                   fused=torch.cuda.is_available())
#         if self.use_trm: self.trm_optimizer = AdamW(self.token_reward_model.parameters(),
#                                                     lr=self.agent_config.algorithm.trm_learning_rate,
#                                                     fused=torch.cuda.is_available())
#         self.ppo_trainer = OffPolicyPPO_Trainer(self.actor, self.critic, self.ac_optimizer, self.agent_config.algorithm,
#                                                 self.tokenizer, self.rank, self.world_size)
#
#         self.timesteps, self.start_iteration = 0, 0
#         self.load_checkpoint()
#
#         self.evaluation_processes = []
#         atexit.register(self._cleanup_processes)
# src/trainers/rl_agent.py
# [L-CARE V15.7-MEMFIX - FINAL PRODUCTION-READY FORM]

import os
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import DictConfig, OmegaConf
from tqdm import trange
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import json
from copy import deepcopy
import logging
import atexit
from multiprocessing import Process

from transformers import AutoTokenizer, AutoConfig
from peft import PeftModel

# Local project imports
from src.models.actor_critic import LCARE_Actor, LCARE_Critic, LCARE_TokenRewardModel
from src.models.lge_encoder import LGE_Encoder
from src.envs.math_reasoning_env import MathReasoningEnv
from src.rl.buffer import LCAREReplayBuffer
from src.rl.algorithm import OffPolicyPPO_Trainer, compute_gae
from src.datasets.rl_prompt_dataset import RLPromptDataset
from src.utils.logger import SwanLabLogger
from src.utils.verifier import Verifier
from src.utils.prompt_constructor import PromptConstructor
from src.utils.distributed_utils import is_main_process, broadcast_object
from evaluate import run_evaluation as run_evaluation_process

# Setup a dedicated logger for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LCARE_Agent:
    """
    [V15.7-MEMFIX] The definitive, memory-safe, synergistic L-CARE Agent.
    - Implements a memory-aware initialization strategy to prevent OOM in 'spawn' mode.
    - Models are loaded on CPU first, then moved to GPU just-in-time.
    - Retains all synergistic algorithms and robust engineering from previous versions.
    """

    def __init__(self, config: DictConfig, rank: int, world_size: int, swanlab_logger: SwanLabLogger):
        self.config = config
        self.agent_config = config.trainer
        self.model_config = config.model
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")
        self.swanlab_logger = swanlab_logger

        # --- Basic Setup (remains the same) ---
        self.verifier = Verifier(config.verifier)
        self.use_lora = self.model_config.get("use_lora", True)
        self.use_trm = self.agent_config.exploration.get("use_token_reward_model", False)

        if self.agent_config.exploration.rollouts_per_iteration % self.world_size != 0:
            raise ValueError(f"Config Error: `rollouts_per_iteration` must be divisible by `world_size`.")
        if not self.use_lora:
            raise NotImplementedError("This Agent is specifically designed for LoRA training.")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.path, trust_remote_code=True,
                                                       padding_side='left')
        self._setup_tokenizer()
        self.prompt_constructor = PromptConstructor(config, self.tokenizer)

        # --- [CRITICAL FIX] Memory-Aware Model Initialization ---
        logger.info(f"[Rank {rank}] Starting memory-aware model initialization...")

        # Step 1: Load all model templates onto CPU first to save GPU memory.
        with torch.device("cpu"):
            logger.info(f"[Rank {rank}] Loading base actor template on CPU...")
            actor_cpu_template = self._initialize_actor_template()

        # Step 2: Create models for DDP. They will be moved to GPU just before wrapping.
        actor_train = deepcopy(actor_cpu_template)
        critic_train = LCARE_Critic(config.model.critic)

        # Step 3: Create local actor for rollouts (no DDP). Move it to its device.
        # This is the only model that needs to be on the device before DDP setup.
        self.local_actor = deepcopy(actor_cpu_template).to(self.device).eval()
        logger.info(f"[Rank {rank}] Local actor for rollouts placed on device {self.device}.")

        # Step 4: Wrap training models with DDP. This handles moving them to the correct GPU.
        logger.info(f"[Rank {rank}] Wrapping training actor with DDP...")
        self.actor = DDP(actor_train.to(self.device), device_ids=[self.rank], find_unused_parameters=True)

        logger.info(f"[Rank {rank}] Wrapping training critic with DDP...")
        self.critic = DDP(critic_train.to(self.device), device_ids=[self.rank], find_unused_parameters=True)

        self.token_reward_model = None
        if self.use_trm:
            with torch.device("cpu"):
                trm_cpu = LCARE_TokenRewardModel(config.model.token_reward_model)
            logger.info(f"[Rank {rank}] Wrapping token reward model with DDP...")
            self.token_reward_model = DDP(trm_cpu.to(self.device), device_ids=[self.rank], find_unused_parameters=True)

        dist.barrier()
        if is_main_process(): logger.info("✅ All training models successfully wrapped with DDP.")

        # Step 5: Initialize components that are only needed on the main process.
        self.replay_buffer, self.encoder = None, None
        if is_main_process():
            logger.info("[Rank 0] Initializing main-process-only components (Buffer, Encoder)...")
            # The encoder needs a CPU model template and its own device.
            self.replay_buffer = LCAREReplayBuffer(self.agent_config.buffer, self.agent_config.exploration,
                                                   actor_cpu_template.model.config.hidden_size)
            if self.agent_config.exploration.get("use_lge", False):
                base_model_cpu = actor_cpu_template.model.get_base_model() if self.use_lora else actor_cpu_template.model
                self.encoder = LGE_Encoder(base_model_cpu, self.tokenizer, self.device)
        # --- [END OF FIX] ---

        prompt_path = os.path.join(config.data.processed_dir, config.data.rl_prompt_file)
        prompt_set = list(RLPromptDataset(prompt_path)) if os.path.exists(prompt_path) else []
        self.env = MathReasoningEnv(prompt_set, self.verifier, self.prompt_constructor,
                                    self.agent_config.env.max_steps_per_episode, self.config)

        params_to_optimize = list(self.actor.parameters()) + list(self.critic.parameters())
        self.ac_optimizer = AdamW(params_to_optimize, lr=self.agent_config.algorithm.learning_rate,
                                  fused=torch.cuda.is_available())

        self.trm_optimizer = None
        if self.use_trm:
            self.trm_optimizer = AdamW(self.token_reward_model.parameters(),
                                       lr=self.agent_config.algorithm.trm_learning_rate,
                                       fused=torch.cuda.is_available())

        self.ppo_trainer = OffPolicyPPO_Trainer(self.actor, self.critic, self.ac_optimizer, self.agent_config.algorithm,
                                                self.tokenizer, self.rank, self.world_size)

        self.timesteps, self.start_iteration = 0, 0
        self.load_checkpoint()

        self.evaluation_processes = []
        atexit.register(self._cleanup_processes)
    def learn(self):
        pbar = trange(self.start_iteration, self.agent_config.exploration.total_iterations,
                      disable=not is_main_process(), desc="RL Training Iteration")
        for iteration in pbar:
            self.local_actor.load_state_dict(self.actor.module.state_dict())
            trajectories = self._collect_rollouts()
            gathered_trajs = [None] * self.world_size
            dist.all_gather_object(gathered_trajs, trajectories)

            if is_main_process():
                flat_trajs = [t for rank_trajs in gathered_trajs for t in rank_trajs if t]
                self._process_and_store_rollouts(flat_trajs, iteration, pbar)

            dist.barrier()
            buffer_size = broadcast_object(len(self.replay_buffer) if is_main_process() else 0)

            if buffer_size >= self.agent_config.exploration.learning_starts:
                self._update_models_distributed(iteration)

            eval_interval = self.agent_config.saving.get("evaluation_interval", -1)
            if is_main_process() and eval_interval > 0 and (iteration + 1) % eval_interval == 0:
                self.save_checkpoint(iteration)
                self._start_async_evaluation(iteration)

        if is_main_process():
            logger.info("Training loop finished. Waiting for final evaluation processes...")
            for p in self.evaluation_processes:
                if p.is_alive(): p.join()
            logger.info("All tasks completed.")
    # [V15.3-FIXED]
    def _collect_rollouts(self) -> List[List[Dict]]:
        rollouts_per_rank = self.agent_config.exploration.rollouts_per_iteration // self.world_size
        trajectories = []

        for _ in range(rollouts_per_rank):
            # --- [V15.3-FIX] 修复分布式决策同步问题 ---
            # 1. 决策只在主进程上做出
            decision = False
            if is_main_process():
                decision = np.random.rand() < self.agent_config.exploration.lge_config.get("frontier_rollout_ratio",
                                                                                           0.0)

            # 2. 将这个唯一的决策广播给所有进程
            is_frontier_rollout = broadcast_object(decision)
            # --- [END OF FIX] ---

            start_info = None
            if is_frontier_rollout:
                # 现在，所有进程要么都进入这个if，要么都跳过，因为is_frontier_rollout在所有进程上都相同
                if is_main_process():
                    start_info = self.replay_buffer.sample_frontier_state()

                # 这个broadcast现在是安全的，因为所有进程都会调用它
                start_info = broadcast_object(start_info)

            trajectories.append(self._collect_one_trajectory(start_info))

        return trajectories

    def _collect_one_trajectory(self, start_info: Optional[Dict] = None) -> List[Dict]:
        trajectory = []
        if start_info and 'state_text' in start_info and 'metadata' in start_info:
            obs_text, env_info = start_info['state_text'], start_info['metadata']
        else:
            obs_text, env_info = self.env.reset()

        for _ in range(self.agent_config.env.max_steps_per_episode):
            state_tokens = self.tokenizer(obs_text, return_tensors="pt", truncation=True, max_length=4096).to(
                self.device)
            with torch.no_grad():
                sampling_params = {'max_new_tokens': 2048, 'do_sample': True, 'temperature': 0.9, 'top_p': 0.95,
                                   'eos_token_id': self.eos_token_ids}
                action_ids, behavior_log_prob = self.local_actor.generate(state_tokens['input_ids'],
                                                                          state_tokens['attention_mask'],
                                                                          sampling_params)
            action_text = self.tokenizer.decode(action_ids[0], skip_special_tokens=True)
            next_obs_text, _, terminated, truncated, step_info = self.env.step(action_text)
            done = terminated or truncated
            trajectory.append({'state_text': obs_text, 'action_ids': action_ids[0].cpu(),
                               'behavior_log_prob': behavior_log_prob.cpu(),
                               'external_reward': 1.0 if step_info.get('is_correct') else 0.0, 'metadata': env_info,
                               'done': done})
            obs_text = next_obs_text
            if done: break
        return trajectory

    def _process_and_store_rollouts(self, trajectories: List[List[Dict]], iteration: int, pbar: trange):
        if not is_main_process() or not self.replay_buffer: return
        total_trajs, correct_trajs = 0, 0
        for traj in trajectories:
            if self.agent_config.exploration.get("use_lge", False):
                traj = self._compute_and_add_intrinsic_rewards(traj)
                frontier_candidates = [{'state_text': step['state_text'], 'metadata': step['metadata'],
                                        'novelty': step.get('intrinsic_reward', 0)} for step in traj]
                if frontier_candidates:
                    frontier_candidates.sort(key=lambda x: x['novelty'], reverse=True)
                    num_to_take = int(
                        len(frontier_candidates) * self.agent_config.exploration.lge_config.get("frontier_top_k_ratio",
                                                                                                0.05))
                    if num_to_take > 0:
                        self.replay_buffer.add_frontier_states(frontier_candidates[:num_to_take])
            self.replay_buffer.add_trajectory(traj)
            self.timesteps += len(traj)
            total_trajs += 1
            if traj[-1]['external_reward'] == 1.0: correct_trajs += 1
        avg_acc = (correct_trajs / total_trajs) if total_trajs > 0 else 0
        frontier_size = len(self.replay_buffer.frontier_states) if self.use_lge and hasattr(self.replay_buffer,
                                                                                            'frontier_states') else 0
        self.swanlab_logger.log({'rollout/correctness': avg_acc, 'rollout/total_trajectories': float(total_trajs),
                                 'rollout/timesteps_total': float(self.timesteps),
                                 'rollout/frontier_buffer_size': float(frontier_size)}, step=iteration)
        pbar.set_description(
            f"Iter {iteration} | Buffer: {len(self.replay_buffer)} | Frontiers: {frontier_size} | Acc: {avg_acc:.2f}")

    def _collate_and_compute_rewards(self, data_chunk: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        trajectories = data_chunk['trajs']
        if not trajectories: return None
        collated = defaultdict(list)
        use_her, her_k = self.agent_config.buffer.use_her, self.agent_config.buffer.her_k_relabel
        her_effort_bonus = self.agent_config.buffer.get("her_effort_bonus", 0.0)
        for traj in trajectories:
            final_reward, pass_rate = traj[-1]['external_reward'], traj[0]['metadata'].get('pass_rate', 0.5)
            extrinsic_reward = final_reward if final_reward == 1.0 else -pass_rate
            self._process_single_traj_for_batching(collated, traj, extrinsic_reward)
            if use_her and her_k > 0 and final_reward < 1.0 and len(traj) > 1:
                for _ in range(her_k):
                    split_point = np.random.randint(1, len(traj))
                    sub_traj = traj[:split_point]
                    effort_proxy = len(sub_traj) / len(traj)
                    her_extrinsic_reward = 1.0 + her_effort_bonus * effort_proxy
                    self._process_single_traj_for_batching(collated, sub_traj, her_extrinsic_reward)
        if not collated['prompts']: return None
        prompt_tok = self.tokenizer(collated['prompts'], padding=True, truncation=True, max_length=2048,
                                    return_tensors='pt')
        action_tok = self.tokenizer.pad({'input_ids': collated['actions']}, padding=True, return_tensors='pt')
        input_ids = torch.cat([prompt_tok.input_ids, action_tok.input_ids], dim=1).to(self.device)
        attention_mask = torch.cat([prompt_tok.attention_mask, action_tok.attention_mask], dim=1).to(self.device)
        labels = input_ids.clone()
        labels[:, :prompt_tok.input_ids.shape[1]] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        with torch.no_grad():
            values = self.critic.module(input_ids, attention_mask).squeeze(-1)
            rewards = torch.zeros_like(values)
            if self.use_trm and self.token_reward_model:
                token_rewards = self.token_reward_model.module(input_ids, attention_mask).squeeze(-1)
                sequence_lengths = torch.sum(attention_mask, dim=1) - 1
                rewards = token_rewards[torch.arange(len(token_rewards)), sequence_lengths]
        advantages, returns = [], []
        start_idx = 0
        for i in range(len(collated['seq_lens'])):
            seq_len, extrinsic_reward = collated['seq_lens'][i], collated['extrinsic_rewards'][i]
            end_idx = start_idx + seq_len
            seq_rewards = rewards[start_idx:end_idx].clone()
            seq_rewards[-1] += extrinsic_reward
            seq_dones = torch.zeros_like(seq_rewards)
            seq_dones[-1] = 1.0
            response_mask_bool = torch.zeros_like(attention_mask[start_idx:end_idx], dtype=torch.bool)
            response_mask_bool[:, prompt_tok.input_ids.shape[1]:] = action_tok.attention_mask[start_idx:end_idx] > 0
            adv, ret = compute_gae(seq_rewards.cpu().numpy(), values[start_idx:end_idx].cpu().numpy(),
                                   seq_dones.cpu().numpy(), response_mask_bool.cpu().numpy(),
                                   self.agent_config.algorithm.gamma, self.agent_config.algorithm.tau_gae)
            advantages.append(torch.from_numpy(adv))
            returns.append(torch.from_numpy(ret))
            start_idx = end_idx
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels,
                'advantages': torch.cat(advantages).to(self.device), 'returns': torch.cat(returns).to(self.device),
                'behavior_log_probs': torch.cat(collated['logprobs']).to(self.device),
                'tree_indices': torch.tensor(data_chunk['indices'], dtype=torch.long, device=self.device),
                'weights': torch.tensor(data_chunk['weights'], dtype=torch.float32, device=self.device),
                'outcome_labels': torch.tensor(collated['outcome_labels'], dtype=torch.float32).to(self.device)}

    def _process_single_traj_for_batching(self, collated: defaultdict, trajectory: List[Dict], extrinsic_reward: float):
        collated['prompts'].extend([s['state_text'] for s in trajectory])
        collated['actions'].extend([s['action_ids'] for s in trajectory])
        collated['logprobs'].extend([s['behavior_log_prob'] for s in trajectory])
        collated['seq_lens'].append(len(trajectory))
        collated['extrinsic_rewards'].append(extrinsic_reward)
        final_outcome = 1.0 if extrinsic_reward > 0 else 0.0
        collated['outcome_labels'].extend([final_outcome] * len(trajectory))

    def _update_models_distributed(self, iteration: int):
        self.actor.train()
        self.critic.train()
        if self.use_trm and self.token_reward_model: self.token_reward_model.train()
        for epoch in range(self.agent_config.algorithm.ppo_epochs):
            data_chunk = self._sample_and_scatter_batch()
            if not data_chunk or not data_chunk.get('trajs'): continue
            batch = self._collate_and_compute_rewards(data_chunk)
            if not batch: continue
            log_dict, _ = self._perform_train_step(batch)
            self._update_per_priorities(batch)
            if is_main_process():
                final_log = {k: float(np.mean(v)) for k, v in log_dict.items() if v}
                self.swanlab_logger.log(final_log, step=iteration * self.agent_config.algorithm.ppo_epochs + epoch)

    def save_checkpoint(self, iteration: int):
        if not is_main_process(): return
        checkpoint_dir = os.path.join(self.agent_config.saving.checkpoint_dir, f"iter_{iteration}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"\nSaving checkpoint on Rank 0 to {checkpoint_dir}...")
        self.actor.module.model.save_pretrained(checkpoint_dir)
        torch.save(self.critic.module.state_dict(), os.path.join(checkpoint_dir, 'critic.pt'))
        if self.use_trm and self.token_reward_model: torch.save(self.token_reward_model.module.state_dict(),
                                                                os.path.join(checkpoint_dir, 'trm.pt'))
        torch.save(self.ac_optimizer.state_dict(), os.path.join(checkpoint_dir, 'ac_optimizer.pt'))
        if self.trm_optimizer: torch.save(self.trm_optimizer.state_dict(),
                                          os.path.join(checkpoint_dir, 'trm_optimizer.pt'))
        if self.replay_buffer: self.replay_buffer.save(os.path.join(checkpoint_dir, "replay_buffer.pkl"))
        metadata = {'iteration': iteration, 'timesteps': self.timesteps}
        with open(os.path.join(checkpoint_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f)
        logger.info(f"✅ Checkpoint saved.")

    def load_checkpoint(self):
        """
        [V15-SYNERGY] Memory-safe checkpoint loading.
        Loads all state dicts to CPU first to prevent GPU OOM on rank 0,
        then allows DDP to distribute them to the correct devices.
        """
        load_path = self.agent_config.saving.checkpoint_dir
        if not os.path.isdir(load_path):
            if is_main_process(): logger.info("Checkpoint directory not found. Starting from initial policy.")
            return
        iter_dirs = [d for d in os.listdir(load_path) if
                     d.startswith("iter_") and os.path.isdir(os.path.join(load_path, d))]
        if not iter_dirs:
            if is_main_process(): logger.info("No iteration checkpoints found. Starting from initial policy.")
            return

        latest_iter = max(int(d.split('_')[1]) for d in iter_dirs)
        latest_ckpt_path = os.path.join(load_path, f"iter_{latest_iter}")
        if is_main_process(): logger.info(f"Detected latest checkpoint, resuming from {latest_ckpt_path}...")

        # Load LoRA adapter directly, Peft handles this efficiently
        self.actor.module.model.load_adapter(latest_ckpt_path, "default")

        # --- Memory-Safe Loading for Standard State Dicts ---
        map_location_cpu = {'cuda:%d' % 0: 'cpu'}

        critic_path = os.path.join(latest_ckpt_path, 'critic.pt')
        self.critic.module.load_state_dict(torch.load(critic_path, map_location=map_location_cpu))

        trm_path = os.path.join(latest_ckpt_path, 'trm.pt')
        if self.use_trm and self.token_reward_model and os.path.exists(trm_path):
            self.token_reward_model.module.load_state_dict(torch.load(trm_path, map_location=map_location_cpu))

        ac_optim_path = os.path.join(latest_ckpt_path, 'ac_optimizer.pt')
        if os.path.exists(ac_optim_path):
            self.ac_optimizer.load_state_dict(torch.load(ac_optim_path, map_location=map_location_cpu))

        trm_optim_path = os.path.join(latest_ckpt_path, 'trm_optimizer.pt')
        if self.trm_optimizer and os.path.exists(trm_optim_path):
            self.trm_optimizer.load_state_dict(torch.load(trm_optim_path, map_location=map_location_cpu))

        if is_main_process():
            self.replay_buffer.load(os.path.join(latest_ckpt_path, "replay_buffer.pkl"))
            with open(os.path.join(latest_ckpt_path, "metadata.json"), 'r') as f:
                metadata = json.load(f)
                self.start_iteration = metadata['iteration'] + 1
                self.timesteps = metadata['timesteps']

        self.start_iteration = int(broadcast_object(self.start_iteration if is_main_process() else 0))
        self.timesteps = int(broadcast_object(self.timesteps if is_main_process() else 0))

        dist.barrier()
        if is_main_process(): logger.info(
            f"✅ Successfully resumed from iteration {self.start_iteration - 1}. Current timesteps: {self.timesteps}")

    # --- Unchanged Helper Methods ---
    def _cleanup_processes(self):
        if is_main_process():
            logger.info("Initiating cleanup of all child processes...")
            for p in self.evaluation_processes:
                if p.is_alive():
                    logger.info(f"Terminating evaluation process {p.pid}...")
                    p.terminate()
                    p.join(timeout=10)
            logger.info("✅ All processes cleaned up successfully.")

    def _sample_and_scatter_batch(self) -> Optional[Dict[str, Any]]:
        batch_size = self.agent_config.algorithm.batch_size
        local_bs = batch_size // self.world_size
        data_to_scatter = [None] * self.world_size
        if is_main_process() and self.replay_buffer and len(self.replay_buffer) >= batch_size:
            sampled_data = self.replay_buffer.sample_trajectories(batch_size)
            if sampled_data:
                tree_indices, trajectories, is_weights = sampled_data
                for i in range(self.world_size):
                    start, end = i * local_bs, (i + 1) * local_bs
                    data_to_scatter[i] = {'trajs': trajectories[start:end], 'indices': tree_indices[start:end],
                                          'weights': is_weights[start:end]}
        scattered_data = broadcast_object(data_to_scatter)
        return scattered_data[self.rank] if scattered_data else None

    # def _start_async_evaluation(self, iteration: int):
    #     if not is_main_process(): return
    #     logger.info(f"--- Iteration {iteration + 1}: Triggering asynchronous evaluation ---")
    #     eval_config = self.config.copy()
    #     checkpoint_dir = os.path.join(self.agent_config.saving.checkpoint_dir, f"iter_{iteration}")
    #     OmegaConf.update(eval_config, "evaluation.model_path", checkpoint_dir)
    #     OmegaConf.update(eval_config, "evaluation.load_lora_adapter", True)
    #     base_run_name = eval_config.evaluation.get("run_name", "eval")
    #     eval_run_name = f"{base_run_name}_iter_{iteration + 1}"
    #     OmegaConf.update(eval_config, "evaluation.run_name", eval_run_name)
    #     p = Process(target=run_evaluation_process, args=(eval_config, self.swanlab_logger, iteration + 1))
    #     p.start()
    #     self.evaluation_processes.append(p)
    #     logger.info(f"Started evaluation subprocess (PID: {p.pid}) for model at iteration {iteration + 1}.")
    def _start_async_evaluation(self, iteration: int):
        if not is_main_process(): return

        logger.info(f"--- Iteration {iteration + 1}: Triggering asynchronous evaluation ---")
        eval_config = self.config.copy()
        checkpoint_dir = os.path.join(self.agent_config.saving.checkpoint_dir, f"iter_{iteration}")

        # --- [CRITICAL FIX] ---
        # Set the adapter_path to the specific checkpoint directory.
        # The base_model_path is already correctly set in the config.
        OmegaConf.update(eval_config, "evaluation.adapter_path", checkpoint_dir)
        # --- [END OF FIX] ---

        base_run_name = eval_config.evaluation.get("run_name", "eval")
        eval_run_name = f"{base_run_name}_iter_{iteration + 1}"
        OmegaConf.update(eval_config, "evaluation.run_name", eval_run_name)

        p = Process(target=run_evaluation_process, args=(eval_config, self.swanlab_logger, iteration + 1, True))

        p.start()
        self.evaluation_processes.append(p)
        logger.info(f"Started evaluation subprocess (PID: {p.pid}) for adapter at {checkpoint_dir}.")

    def _initialize_actor_template(self) -> LCARE_Actor:
        with torch.device("cpu"):
            if is_main_process(): logger.info(f"Loading base model: '{self.model_config.path}'")
            base_model = LCARE_Actor(self.model_config, self.tokenizer)
            sft_checkpoint_path = self.agent_config.initial_policy.path
            if os.path.isdir(sft_checkpoint_path):
                if is_main_process(): logger.info(f"Loading SFT checkpoint as LoRA adapter: '{sft_checkpoint_path}'")
                actor_template_model = PeftModel.from_pretrained(base_model.model, sft_checkpoint_path,
                                                                 is_trainable=True)
                base_model.model = actor_template_model
                if is_main_process(): logger.info(
                    "✅ SFT adapter loaded successfully. RL will only train adapter parameters.")
            else:
                if is_main_process(): logger.warning(
                    f"SFT checkpoint path not found: '{sft_checkpoint_path}'. Training new LoRA layers on the base model.")
        return base_model

    def _setup_tokenizer(self):
        try:
            model_gen_config = AutoConfig.from_pretrained(self.model_config.path, trust_remote_code=True)
            eos_token_id = getattr(model_gen_config, 'eos_token_id', self.tokenizer.eos_token_id)
            self.eos_token_ids = [eos_token_id] if not isinstance(eos_token_id, list) else eos_token_id
            if self.tokenizer.pad_token_id is None:
                pad_token_id = getattr(model_gen_config, 'pad_token_id', self.eos_token_ids[0])
                self.tokenizer.pad_token_id = pad_token_id
                self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(pad_token_id)
        except Exception as e:
            if is_main_process(): logger.warning(f"Auto-setup for tokenizer failed: {e}. Falling back to defaults.")
            self.eos_token_ids = [self.tokenizer.eos_token_id] if isinstance(self.tokenizer.eos_token_id,
                                                                             int) else self.tokenizer.eos_token_id
            if self.tokenizer.pad_token_id is None: self.tokenizer.pad_token_id = self.eos_token_ids[0]

    def _perform_train_step(self, batch: Dict) -> Tuple[defaultdict, Optional[torch.Tensor]]:
        log_dict, total_loss = defaultdict(list), None
        if self.use_trm and self.trm_optimizer and self.token_reward_model:
            self.trm_optimizer.zero_grad()
            trm_loss = self.token_reward_model(batch['input_ids'], batch['attention_mask'], batch['outcome_labels'])
            trm_loss.backward()
            self.trm_optimizer.step()
            log_dict['loss/trm'].append(trm_loss.item())
        self.ac_optimizer.zero_grad()
        ppo_log, ppo_loss = self.ppo_trainer.train_step(batch, return_loss=True)
        if ppo_loss is not None:
            total_loss = ppo_loss
            total_loss.backward()
            self.ac_optimizer.step()
        for k, v in ppo_log.items(): log_dict[k].append(v)
        return log_dict, total_loss

    def _update_per_priorities(self, batch: Dict):
        if not (self.replay_buffer and self.replay_buffer.use_per): return
        with torch.no_grad():
            new_values = self.critic.module(batch['input_ids'], batch['attention_mask']).squeeze(-1)
            td_errors = batch['returns'] - new_values
        all_td_errors, all_indices = [torch.empty_like(td_errors) for _ in range(self.world_size)], [
            torch.empty_like(batch['tree_indices']) for _ in range(self.world_size)]
        dist.all_gather(all_td_errors, td_errors)
        dist.all_gather(all_indices, batch['tree_indices'])
        if is_main_process(): self.replay_buffer.update_priorities(torch.cat(all_indices), torch.cat(all_td_errors))

    def _compute_and_add_intrinsic_rewards(self, trajectory: List[Dict]) -> List[Dict]:
        if not self.encoder or not self.replay_buffer: return trajectory
        all_states_text = [t['state_text'] for t in trajectory]
        lge_config = self.agent_config.exploration.lge_config
        with torch.no_grad():
            new_latent_vectors = self.encoder.encode(all_states_text).to(torch.bfloat16)
        for i, latent_vec in enumerate(new_latent_vectors):
            reward = 0.0
            if self.replay_buffer.latent_archive_size > lge_config.k_nearest_neighbors:
                archive_tensor = self.replay_buffer.latent_state_archive[:self.replay_buffer.latent_archive_size].to(
                    self.device)
                distances = torch.norm(archive_tensor - latent_vec, dim=1)
                knn_distances, _ = torch.topk(distances, lge_config.k_nearest_neighbors, largest=False)
                reward = knn_distances.mean().item()
            trajectory[i]['intrinsic_reward'] = reward
            self.replay_buffer.latent_state_archive[self.replay_buffer.next_latent_idx] = latent_vec.cpu()
            self.replay_buffer.next_latent_idx = (self.replay_buffer.next_latent_idx + 1) % lge_config.archive_capacity
            if self.replay_buffer.latent_archive_size < lge_config.archive_capacity: self.replay_buffer.latent_archive_size += 1
        return trajectory