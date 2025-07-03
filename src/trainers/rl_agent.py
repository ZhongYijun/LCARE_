# # src/trainers/rl_agent.py (ULTRA-STABLE VERSION, NO OPTIMIZATIONS)
#
# import os
# import torch
# import torch.distributed as dist
# from torch.optim import AdamW
# from omegaconf import DictConfig
# from tqdm import trange
# from collections import defaultdict
# from typing import Dict, List, Tuple, Any, Optional
# import numpy as np
# import json
# from copy import deepcopy
#
# from transformers import AutoTokenizer, PreTrainedModel
# from peft import PeftModel
#
# import functools
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
# from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
# from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType, FullStateDictConfig
# from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
# from torch.distributed.checkpoint import save, load
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
# import torch.nn.functional as F
#
#
# def _get_transformer_layer_class(model: PreTrainedModel) -> type:
#     if isinstance(model, PeftModel): model = model.get_base_model()
#     for _, module in model.named_modules():
#         if "DecoderLayer" in module.__class__.__name__ or "TransformerBlock" in module.__class__.__name__:
#             return module.__class__
#     raise ValueError("Could not find a valid transformer layer class name.")
#
#
# class LCARE_Agent:
#     def __init__(self, config: DictConfig, rank: int, world_size: int, logger: SwanLabLogger):
#         self.config, self.agent_config, self.model_config = config, config.trainer, config.model
#         self.rank, self.world_size, self.device, self.logger = rank, world_size, torch.device(f"cuda:{rank}"), logger
#         self.verifier = Verifier(config)
#         self.use_lora = self.model_config.get("use_lora", False)
#         if self.agent_config.get("use_lora") is not None: self.use_lora = self.agent_config.use_lora
#         self.tokenizer = AutoTokenizer.from_pretrained(config.model.path, trust_remote_code=True, padding_side='left')
#         if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.prompt_constructor = PromptConstructor(config, self.tokenizer)
#
#         # --- [ULTRA-STABLE INIT] ---
#         # No optimizations, no inter-process communication during init.
#         # Each process creates its models independently.
#
#         # 1. Create models directly on CPU.
#         actor_cpu = LCARE_Actor(self.model_config, self.tokenizer)
#         critic_cpu = LCARE_Critic(config.model.critic)
#
#         # 2. Load adapter if path exists. This is done independently on each rank.
#         if self.use_lora and os.path.isdir(self.agent_config.initial_policy_path):
#             print(f"[Rank {rank}] Loading LoRA adapter from {self.agent_config.initial_policy_path}")
#             actor_cpu.model.load_adapter(self.agent_config.initial_policy_path, "default")
#
#         # 3. Use deepcopy for local_actor and old_actor, as requested.
#         self.local_actor = deepcopy(actor_cpu).to(self.device)
#         old_actor_cpu = deepcopy(actor_cpu)
#
#         # 4. Move models to GPU for FSDP wrapping.
#         actor_gpu = actor_cpu.to(self.device)
#         critic_gpu = critic_cpu.to(self.device)
#         old_actor_gpu = old_actor_cpu.to(self.device)
#
#         models_gpu = [actor_gpu, critic_gpu, old_actor_gpu]
#         fsdp_dtype = torch.bfloat16
#         models_gpu = [m.to(dtype=fsdp_dtype) for m in models_gpu]
#
#         policies = [functools.partial(transformer_auto_wrap_policy,
#                                       transformer_layer_cls={_get_transformer_layer_class(m.model)}) for m in
#                     models_gpu]
#         fsdp_config = {'auto_wrap_policy': None, 'device_id': self.device, 'use_orig_params': self.use_lora,
#                        'mixed_precision': MixedPrecision(param_dtype=fsdp_dtype, reduce_dtype=fsdp_dtype,
#                                                          buffer_dtype=fsdp_dtype)}
#         self.actor, self.critic, self.old_actor = (
#             FSDP(m, **{**fsdp_config, 'auto_wrap_policy': p}) for m, p in zip(models_gpu, policies))
#
#         # 5. Disable torch.compile to remove a major source of non-determinism.
#         # if self.config.get("use_torch_compile", True): ...
#
#         self.old_actor.load_state_dict(self.actor.state_dict())
#         self.old_actor.eval()
#
#         self.use_trm = self.agent_config.exploration.get("use_token_reward_model", False)
#         if self.use_trm:
#             trm = LCARE_TokenRewardModel(config.model.token_reward_model).to(self.device)
#             self.token_reward_model = torch.nn.parallel.DistributedDataParallel(trm, device_ids=[rank])
#             if self.use_lora: self.token_reward_model.module.model.load_state_dict(self.actor.module.state_dict(),
#                                                                                    strict=False)
#         else:
#             self.token_reward_model = None
#
#         self.replay_buffer: Optional[LCAREReplayBuffer] = LCAREReplayBuffer(config) if is_main_process() else None
#         self.encoder: Optional[LGE_Encoder] = None
#         if is_main_process() and self.replay_buffer:
#             base_model_for_encoder = actor_cpu.model.base_model.model if self.use_lora else actor_cpu.model
#             self.encoder = LGE_Encoder(base_model_for_encoder, self.tokenizer, self.device)
#
#         p_path = str(os.path.join(config.data.processed_dir, config.data.rl_prompt_file))
#         p_set = list(RLPromptDataset(p_path))
#         self.env = MathReasoningEnv(p_set, self.verifier, self.prompt_constructor,
#                                     self.agent_config.env.max_steps_per_episode, self.config)
#
#         ac_params = list(self.actor.parameters()) + list(self.critic.parameters())
#         self.ac_optimizer = AdamW(ac_params, lr=self.agent_config.algorithm.learning_rate)
#         if self.use_trm:
#             self.trm_optimizer = AdamW(self.token_reward_model.parameters(),
#                                        lr=self.agent_config.algorithm.trm_learning_rate)
#
#         self.ppo_trainer = OffPolicyPPO_Trainer(self.actor, self.old_actor, self.critic, self.ac_optimizer,
#                                                 self.agent_config.algorithm, self.tokenizer, self.rank, self.world_size)
#         self.timesteps, self.start_iteration = 0, 0
#         self.load_checkpoint()
#
#     def learn(self):
#         pbar = trange(self.start_iteration, self.agent_config.exploration.total_iterations,
#                       disable=not is_main_process(), desc="RL Training")
#         for iteration in pbar:
#             self._sync_local_actor()
#
#             self.local_actor.eval()
#             collected_trajs = self._collect_rollouts()
#
#             self.actor.train()
#
#             all_gathered_trajs = [None] * self.world_size
#             if is_main_process():
#                 dist.gather_object(collected_trajs, all_gathered_trajs, dst=0)
#             else:
#                 dist.gather_object(collected_trajs, None, dst=0)
#
#             if is_main_process() and self.replay_buffer:
#                 total_trajs_in_iter, correct_in_iter = 0, 0
#                 flat_trajs = [traj for rank_trajs in all_gathered_trajs for traj in rank_trajs]
#                 for traj in flat_trajs:
#                     if traj:
#                         if self.agent_config.exploration.get("use_lge", False):
#                             traj = self._compute_and_add_intrinsic_rewards(traj)
#                         self.replay_buffer.add_trajectory(traj)
#                         self.timesteps += len(traj)
#                         total_trajs_in_iter += 1
#                         if traj[-1]['external_reward'] == 1.0: correct_in_iter += 1
#
#                 avg_acc = (correct_in_iter / total_trajs_in_iter) if total_trajs_in_iter > 0 else 0
#                 log_data = {'rollout/correctness': avg_acc, 'rollout/total_trajectories': float(total_trajs_in_iter),
#                             'rollout/timesteps_total': float(self.timesteps), 'iteration': float(iteration)}
#                 self.logger.log(log_data, step=iteration)
#                 pbar.set_description(f"Iter {iteration} | Buffer: {len(self.replay_buffer)} | Acc: {avg_acc:.2f}")
#
#             dist.barrier()
#             buffer_size = broadcast_object(len(self.replay_buffer) if is_main_process() and self.replay_buffer else 0)
#
#             if buffer_size >= self.agent_config.exploration.learning_starts:
#                 self._update_models_distributed(iteration)
#
#             if is_main_process() and iteration > 0 and (iteration + 1) % self.agent_config.saving.save_interval == 0:
#                 self.save_checkpoint(iteration)
#
#     def _sync_local_actor(self):
#         cpu_fsdp_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
#
#         with FSDP.state_dict_type(self.actor, StateDictType.FULL_STATE_DICT, cpu_fsdp_state_dict_config):
#             full_state_dict = self.actor.state_dict()
#
#         if is_main_process():
#             self.local_actor.load_state_dict(full_state_dict)
#
#         dist.barrier()
#
#     def _collect_one_trajectory(self) -> Tuple[List[Dict], Dict]:
#         trajectory, final_info = [], {}
#         obs_text, env_info = self.env.reset()
#         for _ in range(self.agent_config.env.max_steps_per_episode):
#             state_tokens = self.tokenizer(obs_text, return_tensors="pt").to(self.device)
#             sampling_params = {'max_new_tokens': 2048, 'do_sample': True, 'temperature': 0.9, 'top_p': 0.95}
#
#             action_ids, behavior_log_prob = self.local_actor.generate(
#                 state_tokens['input_ids'],
#                 state_tokens['attention_mask'],
#                 sampling_params
#             )
#
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
#
#         return trajectory, final_info
#
#     def _collect_rollouts(self) -> List[List[Dict]]:
#         total_rollouts = self.agent_config.exploration.rollouts_per_iteration
#         rollouts_per_rank = total_rollouts // self.world_size
#         if self.rank < total_rollouts % self.world_size: rollouts_per_rank += 1
#         local_trajectories = [traj for traj, _ in (self._collect_one_trajectory() for _ in range(rollouts_per_rank)) if
#                               traj]
#         return local_trajectories
#
#     # ... The rest of the file is identical to the previous correct version ...
#     def _compute_and_add_intrinsic_rewards(self, trajectory: List[Dict]) -> List[Dict]:
#         if not self.encoder or not self.replay_buffer: return trajectory
#         all_states_text = [t['state_text'] for t in trajectory]
#         lge_config = self.agent_config.exploration.lge_config
#         with torch.no_grad():
#             new_latent_vectors = self.encoder.encode(all_states_text).to(torch.bfloat16)
#         intrinsic_rewards = []
#         for latent_vec in new_latent_vectors:
#             reward = 0.0
#             if self.replay_buffer.latent_archive_size > lge_config.k_nearest_neighbors:
#                 archive_device = self.replay_buffer.latent_state_archive[:self.replay_buffer.latent_archive_size].to(
#                     self.device)
#                 distances = torch.norm(archive_device - latent_vec, dim=1)
#                 knn_distances, _ = torch.topk(distances, lge_config.k_nearest_neighbors, largest=False)
#                 reward = knn_distances.mean().item()
#             intrinsic_rewards.append(reward)
#             self.replay_buffer.latent_state_archive[self.replay_buffer.next_latent_idx] = latent_vec.cpu()
#             self.replay_buffer.next_latent_idx = (self.replay_buffer.next_latent_idx + 1) % lge_config.archive_capacity
#             if self.replay_buffer.latent_archive_size < lge_config.archive_capacity:
#                 self.replay_buffer.latent_archive_size += 1
#         for i, transition in enumerate(trajectory):
#             transition['intrinsic_reward'] = intrinsic_rewards[i]
#         return trajectory
#
#     def _collate_local_rl_batch(self, trajectories: List[List[Dict]]) -> Dict[str, torch.Tensor]:
#         collated = defaultdict(list)
#         her_k = self.agent_config.buffer.her_k_relabel
#         use_her = self.agent_config.buffer.use_her
#         for traj in trajectories:
#             final_reward = traj[-1]['external_reward']
#             pass_rate = traj[0]['metadata'].get('pass_rate', 0.5)
#             self._process_single_traj(collated, traj, final_reward, pass_rate)
#             if use_her and her_k > 0 and len(traj) > 1:
#                 future_indices = np.random.randint(0, len(traj), size=her_k)
#                 for idx in future_indices:
#                     self._process_single_traj(collated, traj, 1.0, 0.5, achieved_state_idx=idx)
#         if not collated['full_text']: return {}
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
#
#     def _process_single_traj(self, collated_batch: defaultdict, trajectory: List[Dict], final_reward: float,
#                              pass_rate: float, achieved_state_idx: Optional[int] = None):
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
#             if self.agent_config.exploration.get("use_lge", False):
#                 intrinsic_rewards = torch.tensor([t.get('intrinsic_reward', 0.0) for t in current_traj],
#                                                  device=self.device)
#                 sequence_lengths = (tokenized.attention_mask == 1).sum(dim=1) - 1
#                 for i in range(len(current_traj)):
#                     rewards[i, sequence_lengths[i]] += self.agent_config.exploration.lge_config.bonus_coef * \
#                                                        intrinsic_rewards[i]
#             extrinsic_reward = 1.0 if achieved_state_idx is not None else (
#                 final_reward if final_reward == 1.0 else -pass_rate)
#             reward_token_pos = (tokenized.attention_mask[-1] == 1).long().sum() - 1
#             rewards[-1, reward_token_pos] += extrinsic_reward
#             advantages, returns = compute_gae(rewards, values, dones, tokenized.attention_mask,
#                                               self.agent_config.algorithm.gamma, self.agent_config.algorithm.tau_gae)
#         collated_batch['full_text'].extend(full_texts)
#         collated_batch['prompt_text'].extend([t['state_text'] for t in current_traj])
#         collated_batch['behavior_log_prob'].extend([t['behavior_log_prob'] for t in current_traj])
#         collated_batch['advantages'].append(advantages)
#         collated_batch['returns'].append(returns)
#         collated_batch['outcome_labels'].extend([final_reward] * len(current_traj))
#
#     def _sample_and_process_batch_distributed(self) -> Dict[str, Any]:
#         batch_size = self.agent_config.algorithm.batch_size
#         local_batch_size = batch_size // self.world_size
#         scatter_list = [None] * self.world_size
#         if is_main_process() and self.replay_buffer:
#             sampled_data = self.replay_buffer.sample_trajectories(batch_size)
#             if sampled_data:
#                 tree_indices, trajectories, is_weights = sampled_data
#                 trajs_chunks = [trajectories[i:i + local_batch_size] for i in
#                                 range(0, len(trajectories), local_batch_size)]
#                 if len(trajs_chunks) == self.world_size:
#                     scatter_list = [{'trajs': trajs_chunks[i], 'indices': tree_indices[i:i + local_batch_size],
#                                      'weights': is_weights[i:i + local_batch_size]} for i in range(self.world_size)]
#         local_data = broadcast_object(scatter_list)
#         if not local_data: return {}
#         local_data_chunk = local_data[self.rank]
#         if not local_data_chunk or not local_data_chunk['trajs']: return {}
#         local_batch = self._collate_local_rl_batch(local_data_chunk['trajs'])
#         if not local_batch: return {}
#         local_batch['tree_indices'] = torch.tensor(local_data_chunk['indices'], dtype=torch.long, device=self.device)
#         local_batch['weights'] = torch.tensor(local_data_chunk['weights'], dtype=torch.float32, device=self.device)
#         return local_batch
#
#     def _train_trm_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
#         self.trm_optimizer.zero_grad()
#         token_rewards = self.token_reward_model.module(batch['input_ids'], batch['attention_mask']).squeeze(-1)
#         masked_rewards = token_rewards * batch['attention_mask']
#         seq_logits = torch.sum(masked_rewards, dim=-1) / (torch.sum(batch['attention_mask'], dim=-1) + 1e-8)
#         loss = F.binary_cross_entropy_with_logits(seq_logits, batch['outcome_labels'].float())
#         loss.backward()
#         self.trm_optimizer.step()
#         return loss
#
#     def _update_models_distributed(self, iteration: int):
#         self.actor.train()
#         self.critic.train()
#         if self.use_trm: self.token_reward_model.train()
#         self.ppo_trainer.update_old_policy()
#         for epoch in range(self.agent_config.algorithm.ppo_epochs):
#             log_dict = defaultdict(list)
#             local_batch = self._sample_and_process_batch_distributed()
#             if not local_batch:
#                 if is_main_process(): print("Skipping update, no data sampled or processed for this rank.")
#                 continue
#             micro_rl_batch = {k: v.to(self.device) for k, v in local_batch.items()}
#             if self.use_trm: log_dict['loss/trm'].append(self._train_trm_step(micro_rl_batch).item())
#             ppo_log, ppo_loss = self.ppo_trainer.train_step(micro_rl_batch, return_loss=True)
#             for k, v in ppo_log.items(): log_dict[k].append(v)
#             total_loss = ppo_loss
#             self.ac_optimizer.zero_grad()
#             total_loss.backward()
#             self.ac_optimizer.step()
#             if self.replay_buffer and self.replay_buffer.use_per:
#                 with torch.no_grad():
#                     new_values = self.critic(micro_rl_batch['input_ids'], micro_rl_batch['attention_mask']).squeeze(-1)
#                     td_errors = micro_rl_batch['returns'].sum(dim=-1) - new_values
#                 all_td_errors = [torch.empty_like(td_errors) for _ in range(self.world_size)]
#                 all_indices = [torch.empty_like(micro_rl_batch['tree_indices']) for _ in range(self.world_size)]
#                 dist.all_gather(all_td_errors, td_errors)
#                 dist.all_gather(all_indices, micro_rl_batch['tree_indices'])
#                 if is_main_process():
#                     self.replay_buffer.update_priorities(torch.cat(all_indices), torch.cat(all_td_errors))
#             if is_main_process():
#                 avg_loss = ppo_loss.clone().detach()
#                 dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
#                 log_dict['loss/policy_avg'] = [avg_loss.item()]
#                 final_log = {k: float(np.mean(v)) for k, v in log_dict.items() if v}
#                 final_log['epoch'] = float(epoch)
#                 self.logger.log(final_log, step=iteration * self.agent_config.algorithm.ppo_epochs + epoch)
#
#     def save_checkpoint(self, iteration: int):
#         checkpoint_dir = os.path.join(self.agent_config.saving.checkpoint_dir, f"iter_{iteration}")
#         if is_main_process(): print(f"\nSaving distributed checkpoint to {checkpoint_dir}...")
#         state_dict = {"actor": self.actor, "critic": self.critic, "old_actor": self.old_actor,
#                       "ac_optimizer": self.ac_optimizer}
#         if self.use_trm: state_dict["trm"] = self.token_reward_model; state_dict["trm_optimizer"] = self.trm_optimizer
#         writer = FileSystemWriter(checkpoint_dir)
#         save(state_dict=state_dict, storage_writer=writer)
#         if is_main_process() and self.replay_buffer:
#             self.replay_buffer.save(os.path.join(checkpoint_dir, "replay_buffer.pkl"))
#             metadata = {'iteration': iteration, 'timesteps': self.timesteps}
#             with open(os.path.join(checkpoint_dir, "metadata.json"), 'w') as f: json.dump(metadata, f)
#         dist.barrier(device_ids=[self.rank])
#         if is_main_process(): print(f"✅ Checkpoint saved successfully.")
#
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
#         if is_main_process() and self.replay_buffer:
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


# # src/trainers/rl_agent.py (FINAL, LOAD-BALANCED, AND STABLE VERSION)

# import os
# import torch
# import torch.distributed as dist
# from torch.optim import AdamW
# from omegaconf import DictConfig
# from tqdm import trange
# from collections import defaultdict
# from typing import Dict, List, Tuple, Any, Optional
# import numpy as np
# import json
# from copy import deepcopy

# from transformers import AutoTokenizer, PreTrainedModel
# from peft import PeftModel

# import functools
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
# from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
# from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType, FullStateDictConfig
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
# from src.utils.distributed_utils import is_main_process, broadcast_object, get_rank, get_world_size
# import torch.nn.functional as F


# def _get_transformer_layer_class(model: PreTrainedModel) -> type:
#     if hasattr(model, '_orig_mod'):
#         model = model._orig_mod
#     if isinstance(model, PeftModel):
#         model = model.get_base_model()
#     for _, module in model.named_modules():
#         if "DecoderLayer" in module.__class__.__name__ or "TransformerBlock" in module.__class__.__name__:
#             return module.__class__
#     raise ValueError("Could not find a valid transformer layer class name.")


# class LCARE_Agent:
#     def __init__(self, config: DictConfig, rank: int, world_size: int, logger: SwanLabLogger):
#         self.config, self.agent_config, self.model_config = config, config.trainer, config.model
#         self.rank, self.world_size, self.device, self.logger = rank, world_size, torch.device(f"cuda:{rank}"), logger
#         self.verifier = Verifier(config)
#         self.use_lora = self.model_config.get("use_lora", False)
#         if self.agent_config.get("use_lora") is not None: self.use_lora = self.agent_config.use_lora

#         self.tokenizer = AutoTokenizer.from_pretrained(config.model.path, trust_remote_code=True, padding_side='left')
#         if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.prompt_constructor = PromptConstructor(config, self.tokenizer)

#         model_kwargs = {'trust_remote_code': True}
#         if config.get("use_flash_attention_2", True):
#             model_kwargs['attn_implementation'] = "flash_attention_2"

#         # Create models on CPU first to save VRAM
#         actor_cpu = LCARE_Actor(self.model_config, self.tokenizer, **model_kwargs)
#         critic_cpu = LCARE_Critic(config.model.critic, **model_kwargs)

#         if self.use_lora and os.path.isdir(self.agent_config.initial_policy_path):
#             if is_main_process(): print(f"Loading LoRA adapter from {self.agent_config.initial_policy_path}")
#             actor_cpu.model.load_adapter(self.agent_config.initial_policy_path, "default")

#         # [STABLE-FIX] Remove the problematic local_actor. old_actor will be used for generation.
#         old_actor_cpu = deepcopy(actor_cpu)
#         models_to_wrap = [actor_cpu, critic_cpu, old_actor_cpu]
#         fsdp_dtype = torch.bfloat16

#         fsdp_config = {
#             'device_id': self.device,
#             'use_orig_params': self.use_lora,
#             'mixed_precision': MixedPrecision(param_dtype=fsdp_dtype, reduce_dtype=fsdp_dtype, buffer_dtype=fsdp_dtype)
#         }

#         wrapped_models = []
#         for model_cpu in models_to_wrap:
#             model_gpu = model_cpu.to(self.device)
#             model_gpu.to(dtype=fsdp_dtype)
#             auto_wrap_policy = functools.partial(transformer_auto_wrap_policy,
#                                                  transformer_layer_cls={_get_transformer_layer_class(model_gpu.model)})
#             wrapped_model = FSDP(model_gpu, auto_wrap_policy=auto_wrap_policy, **fsdp_config)
#             wrapped_models.append(wrapped_model)

#         self.actor, self.critic, self.old_actor = wrapped_models[0], wrapped_models[1], wrapped_models[2]

#         # Sync old_actor initially. It will be updated by the PPO trainer later.
#         self.old_actor.load_state_dict(self.actor.state_dict())
#         self.old_actor.eval()

#         self.use_trm = self.agent_config.exploration.get("use_token_reward_model", False)
#         self.token_reward_model = None
#         if self.use_trm:
#             trm = LCARE_TokenRewardModel(config.model.token_reward_model, **model_kwargs).to(self.device)
#             trm.to(dtype=fsdp_dtype)
#             self.token_reward_model = torch.nn.parallel.DistributedDataParallel(trm, device_ids=[rank])
#             if self.use_lora:
#                 self.token_reward_model.module.model.load_state_dict(self.actor.module.state_dict(), strict=False)

#         hidden_dim = actor_cpu.model.config.hidden_size
#         self.replay_buffer = LCAREReplayBuffer(config, latent_dim=hidden_dim) if is_main_process() else None

#         self.encoder: Optional[LGE_Encoder] = None
#         if self.agent_config.exploration.get("use_lge", False) and is_main_process():
#             # Encoder is only needed on rank 0 where the buffer lives.
#             # It can be initialized from the CPU actor model.
#             base_model_for_encoder = actor_cpu.model.get_base_model() if self.use_lora else actor_cpu.model
#             self.encoder = LGE_Encoder(base_model_for_encoder, self.tokenizer, self.device)

#         p_path = str(os.path.join(config.data.processed_dir, config.data.rl_prompt_file))
#         p_set = list(RLPromptDataset(p_path))
#         self.env = MathReasoningEnv(p_set, self.verifier, self.prompt_constructor,
#                                     self.agent_config.env.max_steps_per_episode, self.config)

#         ac_params = list(self.actor.parameters()) + list(self.critic.parameters())
#         self.ac_optimizer = AdamW(ac_params, lr=self.agent_config.algorithm.learning_rate, fused=True)
#         self.trm_optimizer = None
#         if self.use_trm:
#             self.trm_optimizer = AdamW(self.token_reward_model.parameters(),
#                                        lr=self.agent_config.algorithm.trm_learning_rate, fused=True)

#         self.ppo_trainer = OffPolicyPPO_Trainer(self.actor, self.old_actor, self.critic, self.ac_optimizer,
#                                                 self.agent_config.algorithm, self.tokenizer, self.rank, self.world_size)

#         self.timesteps, self.start_iteration = 0, 0
#         self.load_checkpoint()

#     def learn(self):
#         pbar = trange(self.start_iteration, self.agent_config.exploration.total_iterations,
#                       disable=not is_main_process(), desc="RL Training")
#         for iteration in pbar:
#             self.old_actor.eval()  # Ensure old_actor is in eval mode for generation

#             # [LOAD-BALANCE FIX] All ranks collect data in parallel using the FSDP old_actor.
#             collected_trajs = self._collect_rollouts()

#             # Gather all collected trajectories to the main process for buffering.
#             all_gathered_trajs = [None] * self.world_size
#             dist.gather_object(collected_trajs, all_gathered_trajs if is_main_process() else None, dst=0)

#             if is_main_process() and self.replay_buffer:
#                 flat_trajs = [traj for rank_trajs in all_gathered_trajs for traj in rank_trajs]

#                 total_trajs_in_iter, correct_in_iter = 0, 0
#                 for traj in flat_trajs:
#                     if traj:
#                         if self.agent_config.exploration.get("use_lge", False):
#                             traj = self._compute_and_add_intrinsic_rewards(traj)
#                         self.replay_buffer.add_trajectory(traj)
#                         self.timesteps += len(traj)
#                         total_trajs_in_iter += 1
#                         if traj[-1]['external_reward'] == 1.0: correct_in_iter += 1

#                 avg_acc = (correct_in_iter / total_trajs_in_iter) if total_trajs_in_iter > 0 else 0
#                 log_data = {'rollout/correctness': avg_acc, 'rollout/total_trajectories': float(total_trajs_in_iter),
#                             'rollout/timesteps_total': float(self.timesteps), 'iteration': float(iteration)}
#                 self.logger.log(log_data, step=iteration)
#                 pbar.set_description(f"Iter {iteration} | Buffer: {len(self.replay_buffer)} | Acc: {avg_acc:.2f}")

#             dist.barrier()
#             buffer_size = broadcast_object(len(self.replay_buffer) if is_main_process() and self.replay_buffer else 0)

#             if buffer_size >= self.agent_config.exploration.learning_starts:
#                 self._update_models_distributed(iteration)

#             if is_main_process() and iteration > 0 and (iteration + 1) % self.agent_config.saving.save_interval == 0:
#                 self.save_checkpoint(iteration)

#     def _collect_rollouts(self) -> List[List[Dict]]:
#         """
#         [LOAD-BALANCE FIX] Each rank collects its share of rollouts in parallel.
#         """
#         total_rollouts = self.agent_config.exploration.rollouts_per_iteration
#         rollouts_per_rank = total_rollouts // self.world_size
#         if self.rank < total_rollouts % self.world_size:
#             rollouts_per_rank += 1

#         local_trajectories = []
#         pbar_desc = f"Collecting Rollouts (Rank {self.rank})"
#         # Disable progress bar for non-main processes to keep logs clean
#         for _ in trange(rollouts_per_rank, desc=pbar_desc, disable=not is_main_process()):
#             traj, _ = self._collect_one_trajectory()
#             if traj:
#                 local_trajectories.append(traj)
#         return local_trajectories

#     def _collect_one_trajectory(self) -> Tuple[List[Dict], Dict]:
#         """
#         [STABLE-FIX] Uses the FSDP-wrapped `self.old_actor` for generation.
#         """
#         trajectory, final_info = [], {}
#         obs_text, env_info = self.env.reset()
#         for _ in range(self.agent_config.env.max_steps_per_episode):
#             state_tokens = self.tokenizer(obs_text, return_tensors="pt").to(self.device)
#             sampling_params = {'max_new_tokens': 2048, 'do_sample': True, 'temperature': 0.9, 'top_p': 0.95}

#             with torch.no_grad():
#                 # Access the generate method via `.module` for FSDP-wrapped models
#                 action_ids, behavior_log_prob = self.old_actor.module.generate(
#                     state_tokens['input_ids'],
#                     state_tokens['attention_mask'],
#                     sampling_params
#                 )

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

#     def _compute_and_add_intrinsic_rewards(self, trajectory: List[Dict]) -> List[Dict]:
#         if not self.encoder or not self.replay_buffer: return trajectory
#         all_states_text = [t['state_text'] for t in trajectory]
#         lge_config = self.agent_config.exploration.lge_config
#         with torch.no_grad():
#             new_latent_vectors = self.encoder.encode(all_states_text).to(torch.bfloat16)

#         intrinsic_rewards = []
#         for latent_vec in new_latent_vectors:
#             reward = 0.0
#             if self.replay_buffer.latent_archive_size > lge_config.k_nearest_neighbors:
#                 archive_device = self.replay_buffer.latent_state_archive[:self.replay_buffer.latent_archive_size].to(
#                     self.device)
#                 distances = torch.norm(archive_device - latent_vec, dim=1)
#                 knn_distances, _ = torch.topk(distances, lge_config.k_nearest_neighbors, largest=False)
#                 reward = knn_distances.mean().item()
#             intrinsic_rewards.append(reward)

#             self.replay_buffer.latent_state_archive[self.replay_buffer.next_latent_idx] = latent_vec.cpu()
#             self.replay_buffer.next_latent_idx = (self.replay_buffer.next_latent_idx + 1) % lge_config.archive_capacity
#             if self.replay_buffer.latent_archive_size < lge_config.archive_capacity:
#                 self.replay_buffer.latent_archive_size += 1

#         for i, transition in enumerate(trajectory):
#             transition['intrinsic_reward'] = intrinsic_rewards[i]
#         return trajectory

#     def _process_and_collate_batch(self, trajectories: List[List[Dict]]) -> Dict[str, torch.Tensor]:
#         if not trajectories: return {}

#         collated = defaultdict(list)
#         her_k = self.agent_config.buffer.her_k_relabel
#         use_her = self.agent_config.buffer.use_her

#         for traj in trajectories:
#             final_reward = traj[-1]['external_reward']
#             pass_rate = traj[0]['metadata'].get('pass_rate', 0.5)
#             self._process_single_traj_for_batch(collated, traj, final_reward, pass_rate)
#             if use_her and her_k > 0 and len(traj) > 1:
#                 future_indices = np.random.randint(0, len(traj), size=her_k)
#                 for idx in future_indices:
#                     self._process_single_traj_for_batch(collated, traj, 1.0, 0.5, achieved_state_idx=idx)

#         if not collated['full_text']: return {}

#         padded = {}
#         tokenized = self.tokenizer(collated['full_text'], padding='longest', truncation=True, max_length=8192,
#                                    return_tensors='pt')
#         padded['input_ids'], padded['attention_mask'] = tokenized['input_ids'], tokenized['attention_mask']

#         prompt_lens = [len(ids) for ids in
#                        self.tokenizer(collated['prompt_text'], padding=False, truncation=True, max_length=4096)[
#                            'input_ids']]
#         labels = padded['input_ids'].clone()
#         for i, p_len in enumerate(prompt_lens): labels[i, :p_len] = -100
#         labels[padded['input_ids'] == self.tokenizer.pad_token_id] = -100
#         padded['labels'] = labels

#         for k in ['behavior_log_prob', 'outcome_labels']: padded[k] = torch.tensor(collated[k], dtype=torch.float32)
#         for k in ['advantages', 'returns']: padded[k] = torch.nn.utils.rnn.pad_sequence(collated[k], batch_first=True,
#                                                                                         padding_value=0)

#         return padded

#     def _process_single_traj_for_batch(self, collated_batch: defaultdict, trajectory: List[Dict], final_reward: float,
#                                        pass_rate: float, achieved_state_idx: Optional[int] = None):
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
#             if self.use_trm:
#                 rewards += self.token_reward_model.module(tokenized.input_ids, tokenized.attention_mask).squeeze(-1)

#             if self.agent_config.exploration.get("use_lge", False):
#                 intrinsic_rewards = torch.tensor([t.get('intrinsic_reward', 0.0) for t in current_traj],
#                                                  device=self.device)
#                 sequence_lengths = (tokenized.attention_mask == 1).sum(dim=1) - 1
#                 for i in range(len(current_traj)):
#                     rewards[i, sequence_lengths[i]] += self.agent_config.exploration.lge_config.bonus_coef * \
#                                                        intrinsic_rewards[i]

#             extrinsic_reward = 1.0 if achieved_state_idx is not None else (
#                 final_reward if final_reward == 1.0 else -pass_rate)
#             reward_token_pos = (tokenized.attention_mask[-1] == 1).long().sum() - 1
#             rewards[-1, reward_token_pos] += extrinsic_reward

#             advantages, returns = compute_gae(rewards, values, dones, tokenized.attention_mask,
#                                               self.agent_config.algorithm.gamma, self.agent_config.algorithm.tau_gae)

#         collated_batch['full_text'].extend(full_texts)
#         collated_batch['prompt_text'].extend([t['state_text'] for t in current_traj])
#         collated_batch['behavior_log_prob'].extend([t['behavior_log_prob'] for t in current_traj])
#         collated_batch['advantages'].append(advantages)
#         collated_batch['returns'].append(returns)
#         collated_batch['outcome_labels'].extend([final_reward] * len(current_traj))

#     def _update_models_distributed(self, iteration: int):
#         self.actor.train()
#         self.critic.train()
#         if self.use_trm: self.token_reward_model.train()

#         for epoch in range(self.agent_config.algorithm.ppo_epochs):
#             sampled_data = None
#             if is_main_process() and self.replay_buffer:
#                 sampled_data = self.replay_buffer.sample_trajectories(self.agent_config.algorithm.batch_size)

#             broadcasted_data = broadcast_object(sampled_data)

#             if not broadcasted_data:
#                 if is_main_process(): print("Skipping update, no data sampled from buffer.")
#                 continue

#             tree_indices, trajectories, is_weights = broadcasted_data

#             trajs_per_rank = len(trajectories) // self.world_size
#             start_idx = self.rank * trajs_per_rank
#             end_idx = (self.rank + 1) * trajs_per_rank if self.rank != self.world_size - 1 else len(trajectories)

#             local_trajs = trajectories[start_idx:end_idx]
#             local_indices = torch.tensor(tree_indices[start_idx:end_idx], dtype=torch.long, device=self.device)
#             local_weights = torch.tensor(is_weights[start_idx:end_idx], dtype=torch.float32, device=self.device)

#             if not local_trajs: continue
#             processed_batch = self._process_and_collate_batch(local_trajs)
#             if not processed_batch: continue

#             processed_batch['tree_indices'] = local_indices
#             processed_batch['weights'] = local_weights

#             log_dict = defaultdict(list)
#             micro_batch = {k: v.to(self.device) for k, v in processed_batch.items()}

#             if self.use_trm:
#                 trm_loss = self._train_trm_step(micro_batch)
#                 log_dict['loss/trm'].append(trm_loss.item())

#             ppo_log, ppo_loss = self.ppo_trainer.train_step(micro_batch, return_loss=True)
#             for k, v in ppo_log.items(): log_dict[k].append(v)

#             self.ac_optimizer.zero_grad()
#             ppo_loss.backward()
#             self.ac_optimizer.step()

#             if self.replay_buffer and self.replay_buffer.use_per:
#                 with torch.no_grad():
#                     new_values = self.critic(micro_batch['input_ids'], micro_batch['attention_mask']).squeeze(-1)
#                     td_errors = micro_batch['returns'].sum(dim=-1) - new_values

#                 all_td_errors = [torch.empty_like(td_errors) for _ in range(self.world_size)]
#                 all_indices = [torch.empty_like(micro_batch['tree_indices']) for _ in range(self.world_size)]
#                 dist.all_gather(all_td_errors, td_errors)
#                 dist.all_gather(all_indices, micro_batch['tree_indices'])

#                 if is_main_process():
#                     self.replay_buffer.update_priorities(torch.cat(all_indices), torch.cat(all_td_errors))

#             if is_main_process():
#                 avg_loss = ppo_loss.clone().detach()
#                 dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
#                 log_dict['loss/policy_avg'] = [avg_loss.item()]
#                 final_log = {k: float(np.mean(v)) for k, v in log_dict.items() if v}
#                 final_log['epoch'] = float(epoch)
#                 self.logger.log(final_log, step=iteration * self.agent_config.algorithm.ppo_epochs + epoch)

#         # The only sync point needed for PPO.
#         self.ppo_trainer.update_old_policy()

#     def _train_trm_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
#         self.trm_optimizer.zero_grad()
#         token_rewards = self.token_reward_model.module(batch['input_ids'], batch['attention_mask']).squeeze(-1)
#         masked_rewards = token_rewards * batch['attention_mask']
#         seq_logits = torch.sum(masked_rewards, dim=-1) / (torch.sum(batch['attention_mask'], dim=-1) + 1e-8)
#         loss = F.binary_cross_entropy_with_logits(seq_logits, batch['outcome_labels'].float())
#         loss.backward()
#         self.trm_optimizer.step()
#         return loss

#     def save_checkpoint(self, iteration: int):
#         checkpoint_dir = os.path.join(self.agent_config.saving.checkpoint_dir, f"iter_{iteration}")
#         if is_main_process(): print(f"\nSaving distributed checkpoint to {checkpoint_dir}...")

#         state_dict = {"actor": self.actor, "critic": self.critic, "old_actor": self.old_actor,
#                       "ac_optimizer": self.ac_optimizer}
#         if self.use_trm:
#             state_dict["trm"] = self.token_reward_model
#             state_dict["trm_optimizer"] = self.trm_optimizer

#         writer = FileSystemWriter(checkpoint_dir)
#         save(state_dict=state_dict, storage_writer=writer)

#         if is_main_process() and self.replay_buffer:
#             self.replay_buffer.save(os.path.join(checkpoint_dir, "replay_buffer.pkl"))
#             metadata = {'iteration': iteration, 'timesteps': self.timesteps}
#             with open(os.path.join(checkpoint_dir, "metadata.json"), 'w') as f: json.dump(metadata, f)

#         dist.barrier()
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
#         if self.use_trm:
#             state_dict["trm"] = self.token_reward_model
#             state_dict["trm_optimizer"] = self.trm_optimizer

#         reader = FileSystemReader(latest_ckpt_path)
#         load(state_dict=state_dict, storage_reader=reader)

#         if is_main_process() and self.replay_buffer:
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

# # src/trainers/rl_agent.py (最终的、完整的、经过架构重构的、绝对稳健的版本)

# import os
# import torch
# import torch.distributed as dist
# from torch.optim import AdamW
# from omegaconf import DictConfig
# from tqdm import trange
# from collections import defaultdict
# from typing import Dict, List, Tuple, Any, Optional
# import numpy as np
# import json
# from copy import deepcopy

# from transformers import AutoTokenizer, PreTrainedModel, AutoConfig
# from peft import PeftModel, get_peft_model_state_dict

# import functools
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
# from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
# from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType, FullStateDictConfig
# from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter, save_state_dict, load_state_dict

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

# def _get_transformer_layer_class(model: PreTrainedModel) -> type:
#     if isinstance(model, PeftModel): model = model.get_base_model()
#     for _, module in model.named_modules():
#         if "DecoderLayer" in module.__class__.__name__ or "TransformerBlock" in module.__class__.__name__:
#             return module.__class__
#     raise ValueError("无法在模型中找到一个有效的Transformer层类用于FSDP包装。")


# class LCARE_Agent:
#     def __init__(self, config: DictConfig, rank: int, world_size: int, logger: SwanLabLogger):
#         self.config, self.agent_config, self.model_config = config, config.trainer, config.model
#         self.rank, self.world_size, self.device, self.logger = rank, world_size, torch.device(f"cuda:{rank}"), logger
#         self.verifier = Verifier(config)
#         self.use_lora = self.model_config.get("use_lora", False)
#         self.use_trm = self.agent_config.exploration.get("use_token_reward_model", False)

#         if self.agent_config.exploration.rollouts_per_iteration % self.world_size != 0:
#             raise ValueError(f"CRITICAL ERROR: `rollouts_per_iteration` ({self.agent_config.exploration.rollouts_per_iteration}) "
#                              f"must be divisible by `world_size` ({self.world_size}) to ensure symmetric workload.")

#         if not self.use_lora: raise NotImplementedError("此Agent专为LoRA训练设计。")
#         self.tokenizer = AutoTokenizer.from_pretrained(config.model.path, trust_remote_code=True, padding_side='left')
#         if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.prompt_constructor = PromptConstructor(config, self.tokenizer)
        
#         actor_cpu_template = LCARE_Actor(self.model_config, self.tokenizer)
#         if os.path.isdir(self.agent_config.initial_policy_path):
#             if is_main_process(): print(f"Loading initial LoRA adapter from {self.agent_config.initial_policy_path}")
#             actor_cpu_template.model.load_adapter(self.agent_config.initial_policy_path, "default")
        
#         self.local_actor = deepcopy(actor_cpu_template).to(self.device).eval()
#         fsdp_dtype = torch.bfloat16
#         critic_cpu = LCARE_Critic(config.model.critic)
#         old_actor_cpu = deepcopy(actor_cpu_template)
#         models_to_fsdp = [actor_cpu_template, critic_cpu, old_actor_cpu]
#         for m in models_to_fsdp: m.to(device=self.device, dtype=fsdp_dtype)
        
#         fsdp_config = {'device_id': self.device, 'use_orig_params': self.use_lora, 'mixed_precision': MixedPrecision(param_dtype=fsdp_dtype, reduce_dtype=fsdp_dtype, buffer_dtype=fsdp_dtype)}
#         auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={_get_transformer_layer_class(actor_cpu_template.model)})
        
#         self.actor = FSDP(models_to_fsdp[0], auto_wrap_policy=auto_wrap_policy, **fsdp_config)
#         self.critic = FSDP(models_to_fsdp[1], auto_wrap_policy=auto_wrap_policy, **fsdp_config)
#         self.old_actor = FSDP(models_to_fsdp[2], auto_wrap_policy=auto_wrap_policy, **fsdp_config)
#         self.old_actor.load_state_dict(self.actor.state_dict())
        
#         self.token_reward_model, self.trm_optimizer = None, None
#         if self.use_trm:
#             trm_cpu = LCARE_TokenRewardModel(config.model.token_reward_model)
#             trm_cpu.to(device=self.device, dtype=fsdp_dtype)
#             self.token_reward_model = FSDP(trm_cpu, auto_wrap_policy=auto_wrap_policy, **fsdp_config)
#             self.token_reward_model.load_state_dict(self.actor.state_dict(), strict=False)
        
#         dist.barrier()
#         if is_main_process(): print("✅ 所有模型已成功使用FSDP包装。")
        
#         self.replay_buffer, self.encoder = None, None
#         if is_main_process():
#             actor_config = AutoConfig.from_pretrained(self.model_config.path, trust_remote_code=True)
#             self.replay_buffer = LCAREReplayBuffer(self.agent_config.buffer, self.agent_config.exploration, actor_config.hidden_size)
#             self.encoder = LGE_Encoder(actor_cpu_template.model.get_base_model(), self.tokenizer, self.device)
            
#         prompt_path = os.path.join(config.data.processed_dir, config.data.rl_prompt_file)
#         prompt_set = list(RLPromptDataset(prompt_path)) if os.path.exists(prompt_path) else []
#         self.env = MathReasoningEnv(prompt_set, self.verifier, self.prompt_constructor, self.agent_config.env.max_steps_per_episode, self.config)
        
#         params_to_optimize = list(self.actor.parameters()) + list(self.critic.parameters())
#         self.ac_optimizer = AdamW(params_to_optimize, lr=self.agent_config.algorithm.learning_rate, fused=torch.cuda.is_available())
#         if self.use_trm: self.trm_optimizer = AdamW(self.token_reward_model.parameters(), lr=self.agent_config.algorithm.trm_learning_rate, fused=torch.cuda.is_available())
        
#         self.ppo_trainer = OffPolicyPPO_Trainer(self.actor, self.old_actor, self.critic, self.ac_optimizer, self.agent_config.algorithm, self.tokenizer, self.rank, self.world_size)
#         self.timesteps, self.start_iteration = 0, 0
#         self.load_checkpoint()

#     def learn(self):
#         pbar = trange(self.start_iteration, self.agent_config.exploration.total_iterations, 
#                       disable=not is_main_process(), desc="RL训练")
#         for iteration in pbar:
#             self._sync_local_actor()

#             # 数据采集现在是一个单一、对称的操作
#             trajectories = self._collect_rollouts()

#             # 使用高效的张量通信而不是脆弱的对象通信
#             gathered_trajs = [None] * self.world_size
#             dist.all_gather_object(gathered_trajs, trajectories)
            
#             if is_main_process():
#                 flat_trajs = [traj for rank_trajs in gathered_trajs for traj in rank_trajs]
#                 self._process_and_log_rollouts(flat_trajs, iteration, pbar)

#             dist.barrier()
#             buffer_size = broadcast_object(len(self.replay_buffer) if is_main_process() else 0)
            
#             if buffer_size >= self.agent_config.exploration.learning_starts:
#                 self._update_models_distributed(iteration)
            
#             if (iteration + 1) % self.agent_config.saving.save_interval == 0:
#                 self.save_checkpoint(iteration)

#     def _sync_local_actor(self):
#         lora_state_dict = None
#         fsdp_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
#         with FSDP.state_dict_type(self.actor, StateDictType.FULL_STATE_DICT, fsdp_save_policy):
#             cpu_state = self.actor.module.model.state_dict()
#             if is_main_process():
#                 lora_state_dict = get_peft_model_state_dict(self.actor.module.model, state_dict=cpu_state)
#         lora_state_dict = broadcast_object(lora_state_dict, src_rank=0)
#         self.local_actor.model.load_state_dict(lora_state_dict, strict=False)
#         dist.barrier()

#     def _collect_rollouts(self) -> List[List[Dict]]:
#         rollouts_per_rank = self.agent_config.exploration.rollouts_per_iteration // self.world_size
#         return [self._collect_one_trajectory() for _ in range(rollouts_per_rank)]

#     def _collect_one_trajectory(self) -> List[Dict]:
#         trajectory, (obs_text, env_info) = [], self.env.reset()
#         for _ in range(self.agent_config.env.max_steps_per_episode):
#             state_tokens = self.tokenizer(obs_text, return_tensors="pt", truncation=True, max_length=4096).to(self.device)
#             sampling_params = {'max_new_tokens': 2048, 'do_sample': True, 'temperature': 0.9, 'top_p': 0.95}
#             action_ids, behavior_log_prob = self.local_actor.generate(state_tokens['input_ids'], state_tokens['attention_mask'], sampling_params)
#             action_text = self.tokenizer.decode(action_ids[0], skip_special_tokens=True)
#             next_obs_text, _, terminated, truncated, step_info = self.env.step(action_text)
#             done = terminated or truncated
#             trajectory.append({'state_text': obs_text, 'action_ids': action_ids[0].cpu(),'behavior_log_prob': behavior_log_prob[0].cpu(),'external_reward': 1.0 if step_info.get('is_correct') else 0.0,'metadata': env_info, 'done': done})
#             obs_text = next_obs_text
#             if done: break
#         return trajectory

#     def _process_and_log_rollouts(self, trajectories: List[List[Dict]], iteration: int, pbar: trange):
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
#         self.logger.log({'rollout/correctness': avg_acc, 'rollout/total_trajectories': float(total_trajs), 'rollout/timesteps_total': float(self.timesteps)}, step=iteration)
#         pbar.set_description(f"Iter {iteration} | Buffer: {len(self.replay_buffer)} | Acc: {avg_acc:.2f}")

#     def _update_models_distributed(self, iteration: int):
#         self.actor.train(); self.critic.train()
#         if self.use_trm: self.token_reward_model.train()
#         self.ppo_trainer.update_old_policy()
#         for epoch in range(self.agent_config.algorithm.ppo_epochs):
#             local_data_chunk = self._sample_and_scatter_batch()
#             if not local_data_chunk or not local_data_chunk.get('trajs'): continue
#             local_batch = self._collate_local_rl_batch(local_data_chunk['trajs'])
#             if not local_batch: continue
#             local_batch['tree_indices'] = torch.tensor(local_data_chunk['indices'], dtype=torch.long, device=self.device)
#             local_batch['weights'] = torch.tensor(local_data_chunk['weights'], dtype=torch.float32, device=self.device)
#             log_dict, _ = self._perform_train_step(local_batch)
#             self._update_per_priorities(local_batch)
#             if is_main_process():
#                 final_log = {k: float(np.mean(v)) for k, v in log_dict.items() if v}
#                 self.logger.log(final_log, step=iteration * self.agent_config.algorithm.ppo_epochs + epoch)

#     def _sample_and_scatter_batch(self) -> Optional[Dict[str, Any]]:
#         batch_size = self.agent_config.algorithm.batch_size
#         local_bs = batch_size // self.world_size
#         if self.rank == 0 and batch_size % self.world_size != 0:
#             print(f"Warning: batch_size {batch_size} is not divisible by world_size {self.world_size}. Some GPUs may have smaller batches.")
#         data_to_scatter = [None] * self.world_size
#         if is_main_process() and self.replay_buffer and len(self.replay_buffer) >= batch_size:
#             sampled_data = self.replay_buffer.sample_trajectories(batch_size)
#             if sampled_data:
#                 tree_indices, trajectories, is_weights = sampled_data
#                 for i in range(self.world_size):
#                     start, end = i * local_bs, (i + 1) * local_bs
#                     data_to_scatter[i] = {'trajs': trajectories[start:end], 'indices': tree_indices[start:end], 'weights': is_weights[start:end]}
#         scattered_data = broadcast_object(data_to_scatter)
#         return scattered_data[self.rank] if scattered_data else None

#     def _collate_local_rl_batch(self, trajectories: List[List[Dict]]) -> Dict[str, Any]:
#         collated = defaultdict(list)
#         use_her = self.agent_config.buffer.use_her
#         her_k = self.agent_config.buffer.her_k_relabel
#         for traj in trajectories:
#             if not traj: continue
#             final_reward = traj[-1]['external_reward']
#             pass_rate = traj[0]['metadata'].get('pass_rate', 0.5)
#             self._process_single_traj(collated, traj, final_reward, pass_rate)
#             if use_her and her_k > 0 and final_reward < 1.0 and len(traj) > 1:
#                 future_indices = np.random.randint(0, len(traj), size=her_k)
#                 for idx in future_indices: self._process_single_traj(collated, traj, 1.0, 0.5, achieved_state_idx=idx)
#         if not collated['full_text']: return {}
        
#         padded = {}
#         tokenized = self.tokenizer(collated['full_text'], padding='longest', truncation=True, max_length=1024, return_tensors='pt')
#         padded['input_ids'], padded['attention_mask'] = tokenized['input_ids'], tokenized['attention_mask']
        
#         prompt_lens = [len(self.tokenizer.encode(pt, add_special_tokens=False)) for pt in collated['prompt_text']]
#         labels = padded['input_ids'].clone()
#         for i, p_len in enumerate(prompt_lens): labels[i, :p_len] = -100
#         labels[padded['input_ids'] == self.tokenizer.pad_token_id] = -100
#         padded['labels'] = labels
        
#         padded['advantages'] = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(adv) for adv in collated['advantages']], batch_first=True, padding_value=0.0)
#         padded['returns'] = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(ret) for ret in collated['returns']], batch_first=True, padding_value=0.0)
#         padded['outcome_labels'] = torch.tensor(collated['outcome_labels'], dtype=torch.float32)
#         return padded

#     def _process_single_traj(self, collated_batch: defaultdict, trajectory: List[Dict], final_reward: float, pass_rate: float, achieved_state_idx: Optional[int] = None):
#         current_traj = trajectory[:achieved_state_idx + 1] if achieved_state_idx is not None else trajectory
#         if not current_traj: return
#         full_texts = [t['state_text'] + self.tokenizer.decode(t['action_ids'], skip_special_tokens=True) for t in current_traj]
#         dones = torch.tensor([float(t['done']) for t in current_traj], device=self.device)
#         if achieved_state_idx is not None: dones[-1] = 1.0
        
#         with torch.no_grad():
#             tokenized = self.tokenizer(full_texts, padding='longest', truncation=True, max_length=1024, return_tensors="pt").to(self.device)
#             values = self.critic(tokenized.input_ids, tokenized.attention_mask).squeeze(-1)
#             rewards = torch.zeros_like(values)
#             if self.use_trm and self.token_reward_model:
#                 token_rewards = self.token_reward_model(tokenized.input_ids, tokenized.attention_mask).squeeze(-1)
#                 sequence_lengths = torch.sum(tokenized.attention_mask, dim=1) - 1
#                 rewards = token_rewards[torch.arange(len(token_rewards)), sequence_lengths]
#             if self.agent_config.exploration.get("use_lge", False):
#                 intrinsic_rewards = torch.tensor([t.get('intrinsic_reward', 0.0) for t in current_traj], device=self.device)
#                 rewards += self.agent_config.exploration.lge_config.bonus_coef * intrinsic_rewards
#             extrinsic_reward = 1.0 if achieved_state_idx is not None else (final_reward if final_reward == 1.0 else -pass_rate)
#             rewards[-1] += extrinsic_reward
#             advantages, returns = compute_gae(rewards.cpu().numpy(), values.cpu().numpy(), dones.cpu().numpy(), tokenized.attention_mask.cpu().numpy(), self.agent_config.algorithm.gamma, self.agent_config.algorithm.tau_gae)
        
#         collated_batch['full_text'].extend(full_texts)
#         collated_batch['prompt_text'].extend([t['state_text'] for t in current_traj])
#         collated_batch['advantages'].append(advantages)
#         collated_batch['returns'].append(returns)
#         collated_batch['outcome_labels'].extend([final_reward] * len(current_traj))

#     def _perform_train_step(self, batch: Dict) -> Tuple[defaultdict, Optional[torch.Tensor]]:
#         log_dict = defaultdict(list)
#         total_loss = None
#         if self.use_trm and self.trm_optimizer and self.token_reward_model:
#             self.trm_optimizer.zero_grad()
#             trm_loss = self.token_reward_model(batch['input_ids'], batch['attention_mask'], batch['outcome_labels'])
#             trm_loss.backward()
#             self.trm_optimizer.step()
#             log_dict['loss/trm'].append(trm_loss.item())
        
#         self.ac_optimizer.zero_grad()
#         ppo_log, ppo_loss = self.ppo_trainer.train_step(batch, return_loss=True)
#         if ppo_loss is not None:
#             total_loss = ppo_loss
#             total_loss.backward()
#             self.ac_optimizer.step()
#         for k, v in ppo_log.items(): log_dict[k].append(v)
#         return log_dict, total_loss

#     def _update_per_priorities(self, batch: Dict):
#         if not (self.replay_buffer and self.replay_buffer.use_per): return
#         with torch.no_grad():
#             new_values = self.critic(batch['input_ids'], batch['attention_mask']).squeeze(-1)
#             td_errors = batch['returns'].sum(dim=-1) - new_values
#         all_td_errors = [torch.empty_like(td_errors) for _ in range(self.world_size)]
#         all_indices = [torch.empty_like(batch['tree_indices']) for _ in range(self.world_size)]
#         dist.all_gather(all_td_errors, td_errors)
#         dist.all_gather(all_indices, batch['tree_indices'])
#         if is_main_process(): self.replay_buffer.update_priorities(torch.cat(all_indices), torch.cat(all_td_errors))

#     def _compute_and_add_intrinsic_rewards(self, trajectory: List[Dict]) -> List[Dict]:
#         if not self.encoder or not self.replay_buffer: return trajectory
#         all_states_text = [t['state_text'] for t in trajectory]
#         lge_config = self.agent_config.exploration.lge_config
#         with torch.no_grad():
#             new_latent_vectors = self.encoder.encode(all_states_text).to(torch.bfloat16)
#         for i, latent_vec in enumerate(new_latent_vectors):
#             reward = 0.0
#             if self.replay_buffer.latent_archive_size > lge_config.k_nearest_neighbors:
#                 archive_tensor = self.replay_buffer.latent_state_archive[:self.replay_buffer.latent_archive_size].to(self.device)
#                 distances = torch.norm(archive_tensor - latent_vec, dim=1)
#                 knn_distances, _ = torch.topk(distances, lge_config.k_nearest_neighbors, largest=False)
#                 reward = knn_distances.mean().item()
#             trajectory[i]['intrinsic_reward'] = reward
#             self.replay_buffer.latent_state_archive[self.replay_buffer.next_latent_idx] = latent_vec.cpu()
#             self.replay_buffer.next_latent_idx = (self.replay_buffer.next_latent_idx + 1) % lge_config.archive_capacity
#             if self.replay_buffer.latent_archive_size < lge_config.archive_capacity: self.replay_buffer.latent_archive_size += 1
#         return trajectory

#     def save_checkpoint(self, iteration: int):
#         dist.barrier()
#         checkpoint_dir = os.path.join(self.agent_config.saving.checkpoint_dir, f"iter_{iteration}")
#         if is_main_process(): os.makedirs(checkpoint_dir, exist_ok=True)
        
#         writer = FileSystemWriter(checkpoint_dir)
        
#         state_dict_to_save = { "actor": self.actor, "critic": self.critic, "old_actor": self.old_actor, "ac_optimizer": self.ac_optimizer }
#         if self.use_trm and self.token_reward_model and self.trm_optimizer:
#             state_dict_to_save["trm"] = self.token_reward_model
#             state_dict_to_save["trm_optimizer"] = self.trm_optimizer
        
#         save_state_dict(state_dict=state_dict_to_save, storage_writer=writer)

#         if is_main_process():
#             print(f"\n主进程(Rank 0)开始保存非模型组件到 {checkpoint_dir}...")
#             if self.replay_buffer: self.replay_buffer.save(os.path.join(checkpoint_dir, "replay_buffer.pkl"))
#             metadata = {'iteration': iteration, 'timesteps': self.timesteps}
#             with open(os.path.join(checkpoint_dir, "metadata.json"), 'w') as f: json.dump(metadata, f)
#             print(f"✅ 检查点组件已在主进程上保存完毕。")

#         dist.barrier()
#         if is_main_process(): print(f"检查点保存流程完成。")

#     def load_checkpoint(self):
#         checkpoint_dir = self.agent_config.saving.checkpoint_dir
#         if not os.path.isdir(checkpoint_dir):
#             if is_main_process(): print("无检查点目录，从头开始训练。")
#             return
#         iter_dirs = [d for d in os.listdir(checkpoint_dir) if d.startswith("iter_") and os.path.isdir(os.path.join(checkpoint_dir, d))]
#         if not iter_dirs:
#             if is_main_process(): print("无检查点文件，从头开始训练。")
#             return
#         latest_iter = max(int(d.split('_')[1]) for d in iter_dirs)
#         load_path = os.path.join(checkpoint_dir, f"iter_{latest_iter}")
#         if is_main_process(): print(f"检测到最新检查点，将从 {load_path} 恢复训练...")

#         reader = FileSystemReader(load_path)
        
#         state_dict_to_load = { "actor": self.actor, "critic": self.critic, "old_actor": self.old_actor, "ac_optimizer": self.ac_optimizer }
#         if self.use_trm and self.token_reward_model and self.trm_optimizer:
#             state_dict_to_load["trm"] = self.token_reward_model
#             state_dict_to_load["trm_optimizer"] = self.trm_optimizer

#         load_state_dict(state_dict=state_dict_to_load, storage_reader=reader)

#         if is_main_process() and self.replay_buffer:
#             self.replay_buffer.load(os.path.join(load_path, "replay_buffer.pkl"))
#             with open(os.path.join(load_path, "metadata.json"), 'r') as f:
#                 metadata = json.load(f)
#                 self.start_iteration = metadata['iteration'] + 1
#                 self.timesteps = metadata['timesteps']
        
#         self.start_iteration = int(broadcast_object(self.start_iteration if is_main_process() else 0))
#         self.timesteps = int(broadcast_object(self.timesteps if is_main_process() else 0))
        
#         dist.barrier()
#         if is_main_process(): print(f"✅ 成功从迭代 {self.start_iteration - 1} 恢复。当前总步数: {self.timesteps}")

# src/trainers/rl_agent.py (最终的、完整的、釜底抽薪的稳健版本)

import os
import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import DictConfig
from tqdm import trange
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import json
from copy import deepcopy

from transformers import AutoTokenizer, AutoConfig


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


class LCARE_Agent:
    def __init__(self, config: DictConfig, rank: int, world_size: int, logger: SwanLabLogger):
        self.config, self.agent_config, self.model_config = config, config.trainer, config.model
        self.rank, self.world_size, self.device, self.logger = rank, world_size, torch.device(f"cuda:{rank}"), logger
        self.verifier = Verifier(config)
        self.use_lora = self.model_config.get("use_lora", False)
        self.use_trm = self.agent_config.exploration.get("use_token_reward_model", False)

        # 在初始化时就检查任务分配的对称性
        if self.agent_config.exploration.rollouts_per_iteration % self.world_size != 0:
            raise ValueError(
                f"CRITICAL ERROR: `rollouts_per_iteration` ({self.agent_config.exploration.rollouts_per_iteration}) "
                f"must be divisible by `world_size` ({self.world_size}) to ensure symmetric workload.")
        if not self.use_lora: raise NotImplementedError("此Agent专为LoRA训练设计。")

        self.tokenizer = AutoTokenizer.from_pretrained(config.model.path, trust_remote_code=True, padding_side='left')
        model_generation_config = AutoConfig.from_pretrained(config.model.path, trust_remote_code=True)
        self.eos_token_ids = getattr(model_generation_config, 'eos_token_id', self.tokenizer.eos_token_id)
        if not isinstance(self.eos_token_ids, list): self.eos_token_ids = [self.eos_token_ids]

        if self.tokenizer.pad_token_id is None:
            pad_token_id_from_config = getattr(model_generation_config, 'pad_token_id', self.eos_token_ids[0])
            if is_main_process(): print(
                f"⚠️ WARNING: Tokenizer's `pad_token_id` not set. Setting to `{pad_token_id_from_config}`.")
            self.tokenizer.pad_token_id = pad_token_id_from_config
            self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(self.tokenizer.pad_token_id)

        if is_main_process(): print(
            f"✅ EOS tokens: {self.eos_token_ids}, PAD token: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")

        self.prompt_constructor = PromptConstructor(config, self.tokenizer)

        actor_cpu_template = LCARE_Actor(self.model_config, self.tokenizer)
        if os.path.isdir(self.agent_config.initial_policy_path):
            if is_main_process(): print(f"Loading initial LoRA adapter from {self.agent_config.initial_policy_path}")
            actor_cpu_template.model.load_adapter(self.agent_config.initial_policy_path, "default")

        self.local_actor = deepcopy(actor_cpu_template).to(self.device).eval()
        actor_train = deepcopy(actor_cpu_template).to(self.device)
        critic_train = LCARE_Critic(config.model.critic).to(self.device)

        self.actor = DDP(actor_train, device_ids=[self.rank])
        self.critic = DDP(critic_train, device_ids=[self.rank])

        self.token_reward_model, self.trm_optimizer = None, None
        if self.use_trm:
            trm_train = LCARE_TokenRewardModel(config.model.token_reward_model).to(self.device)
            self.token_reward_model = DDP(trm_train, device_ids=[self.rank])
            self.token_reward_model.module.load_state_dict(self.actor.module.state_dict(), strict=False)

        dist.barrier()
        if is_main_process(): print("✅ 所有模型已成功使用DDP包装。")

        self.replay_buffer, self.encoder = None, None
        if is_main_process():
            self.replay_buffer = LCAREReplayBuffer(self.agent_config.buffer, self.agent_config.exploration,
                                                   actor_cpu_template.model.config.hidden_size)
            self.encoder = LGE_Encoder(actor_cpu_template.model.get_base_model(), self.tokenizer, self.device)

        prompt_path = os.path.join(config.data.processed_dir, config.data.rl_prompt_file)
        self.prompt_set = list(RLPromptDataset(prompt_path)) if os.path.exists(prompt_path) else []
        self.env = MathReasoningEnv(self.prompt_set, self.verifier, self.prompt_constructor,
                                    self.agent_config.env.max_steps_per_episode, self.config)

        params_to_optimize = list(self.actor.parameters()) + list(self.critic.parameters())
        self.ac_optimizer = AdamW(params_to_optimize, lr=self.agent_config.algorithm.learning_rate,
                                  fused=torch.cuda.is_available())
        if self.use_trm: self.trm_optimizer = AdamW(self.token_reward_model.parameters(),
                                                    lr=self.agent_config.algorithm.trm_learning_rate,
                                                    fused=torch.cuda.is_available())

        self.ppo_trainer = OffPolicyPPO_Trainer(self.actor, self.critic, self.ac_optimizer, self.agent_config.algorithm,
                                                self.tokenizer, self.rank, self.world_size)
        self.timesteps, self.start_iteration = 0, 0
        self.load_checkpoint()

    def learn(self):
        pbar = trange(self.start_iteration, self.agent_config.exploration.total_iterations,
                      disable=not is_main_process(), desc="RL训练")
        for iteration in pbar:
            self.local_actor.load_state_dict(self.actor.module.state_dict())

            # [终极架构] 数据采集、填充、通信的全新流程
            trajectories = self._collect_rollouts()

            gathered_trajs = [None] * self.world_size
            dist.all_gather_object(gathered_trajs, trajectories)

            if is_main_process():
                flat_trajs = [t for rank_trajs in gathered_trajs for t in rank_trajs]
                self._process_and_store_rollouts(flat_trajs, iteration, pbar)

            dist.barrier()
            buffer_size = broadcast_object(len(self.replay_buffer) if is_main_process() else 0)

            if buffer_size >= self.agent_config.exploration.learning_starts:
                self._update_models_distributed(iteration)

            if is_main_process() and (iteration + 1) % self.agent_config.saving.save_interval == 0:
                self.save_checkpoint(iteration)

    def _collect_rollouts(self) -> List[List[Dict]]:
        rollouts_per_rank = self.agent_config.exploration.rollouts_per_iteration // self.world_size
        return [self._collect_one_trajectory() for _ in range(rollouts_per_rank)]

    def _collect_one_trajectory(self) -> List[Dict]:
        trajectory, (obs_text, env_info) = [], self.env.reset()
        for _ in range(self.agent_config.env.max_steps_per_episode):
            state_tokens = self.tokenizer(obs_text, return_tensors="pt", truncation=True, max_length=2048).to(
                self.device)
            with torch.no_grad():
                sampling_params = {'max_new_tokens': 1024, 'do_sample': True, 'temperature': 0.9, 'top_p': 0.95,
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
            if not traj: continue
            if self.agent_config.exploration.get("use_lge", False):
                traj = self._compute_and_add_intrinsic_rewards(traj)
            self.replay_buffer.add_trajectory(traj)
            self.timesteps += len(traj)
            total_trajs += 1
            if traj[-1]['external_reward'] == 1.0: correct_trajs += 1
        avg_acc = (correct_trajs / total_trajs) if total_trajs > 0 else 0
        self.logger.log({'rollout/correctness': avg_acc, 'rollout/total_trajectories': float(total_trajs),
                         'rollout/timesteps_total': float(self.timesteps)}, step=iteration)
        pbar.set_description(f"Iter {iteration} | Buffer: {len(self.replay_buffer)} | Acc: {avg_acc:.2f}")

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
                self.logger.log(final_log, step=iteration * self.agent_config.algorithm.ppo_epochs + epoch)

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

    def _collate_and_compute_rewards(self, data_chunk: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        trajectories = data_chunk['trajs']
        if not trajectories: return None

        collated = defaultdict(list)
        use_her, her_k = self.agent_config.buffer.use_her, self.agent_config.buffer.her_k_relabel
        for traj in trajectories:
            self._process_single_traj_for_batching(collated, traj, traj[-1]['external_reward'],
                                                   traj[0]['metadata'].get('pass_rate', 0.5))
            if use_her and her_k > 0 and traj[-1]['external_reward'] < 1.0 and len(traj) > 1:
                for _ in range(her_k):
                    split_point = np.random.randint(1, len(traj))
                    self._process_single_traj_for_batching(collated, traj[:split_point], 1.0, 0.5, is_her=True)

        if not collated['prompts']: return None

        # 批量Tokenize
        prompt_tok = self.tokenizer(collated['prompts'], padding=True, truncation=True, max_length=1024,
                                    return_tensors='pt')
        action_tok = self.tokenizer.pad({'input_ids': collated['actions']}, padding=True, return_tensors='pt')

        input_ids = torch.cat([prompt_tok.input_ids, action_tok.input_ids], dim=1).to(self.device)
        attention_mask = torch.cat([prompt_tok.attention_mask, action_tok.attention_mask], dim=1).to(self.device)

        labels = input_ids.clone()
        labels[:, :prompt_tok.input_ids.shape[1]] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100

        # 批量计算Values和Rewards
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
            seq_len, final_reward, pass_rate, is_her = collated['seq_lens'][i], collated['final_rewards'][i], \
            collated['pass_rates'][i], collated['is_her'][i]
            end_idx = start_idx + seq_len

            seq_rewards = rewards[start_idx:end_idx].clone()
            extrinsic_reward = 1.0 if is_her else (final_reward if final_reward == 1.0 else -pass_rate)
            seq_rewards[-1] += extrinsic_reward

            seq_dones = torch.zeros_like(seq_rewards)
            seq_dones[-1] = 1.0

            response_mask_bool = torch.zeros_like(attention_mask[start_idx:end_idx], dtype=torch.bool)
            response_mask_bool[:, prompt_tok.input_ids.shape[1]:] = action_tok.attention_mask[start_idx:end_idx] > 0

            adv, ret = compute_gae(seq_rewards.cpu().numpy(), values[start_idx:end_idx].cpu().numpy(),
                                   seq_dones.cpu().numpy(),
                                   response_mask_bool.cpu().numpy(),
                                   self.agent_config.algorithm.gamma, self.agent_config.algorithm.tau_gae)

            advantages.append(torch.from_numpy(adv))
            returns.append(torch.from_numpy(ret))
            start_idx = end_idx

        return {
            'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels,
            'advantages': torch.cat(advantages).to(self.device),
            'returns': torch.cat(returns).to(self.device),
            'behavior_log_probs': torch.cat(collated['logprobs']).to(self.device),
            'tree_indices': torch.tensor(data_chunk['indices'], dtype=torch.long, device=self.device),
            'weights': torch.tensor(data_chunk['weights'], dtype=torch.float32, device=self.device),
            'outcome_labels': torch.tensor(collated['outcome_labels'], dtype=torch.float32).to(self.device)
        }

    def _process_single_traj_for_batching(self, collated: defaultdict, trajectory: List[Dict], final_reward: float,
                                          pass_rate: float, is_her: bool = False):
        collated['prompts'].extend([s['state_text'] for s in trajectory])
        collated['actions'].extend([s['action_ids'] for s in trajectory])
        collated['logprobs'].extend([s['behavior_log_prob'] for s in trajectory])
        collated['seq_lens'].append(len(trajectory))
        collated['final_rewards'].append(final_reward)
        collated['pass_rates'].append(pass_rate)
        collated['is_her'].append(is_her)
        collated['outcome_labels'].extend([final_reward] * len(trajectory))

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

    def save_checkpoint(self, iteration: int):
        if not is_main_process(): return
        checkpoint_dir = os.path.join(self.agent_config.saving.checkpoint_dir, f"iter_{iteration}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"\nSaving checkpoint on rank 0 to {checkpoint_dir}...")
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
        print(f"✅ Checkpoint saved.")

    def load_checkpoint(self):
        checkpoint_dir = self.agent_config.saving.checkpoint_dir
        if not os.path.isdir(checkpoint_dir):
            if is_main_process(): print("No checkpoint directory found, starting from scratch.")
            return
        iter_dirs = [d for d in os.listdir(checkpoint_dir) if
                     d.startswith("iter_") and os.path.isdir(os.path.join(checkpoint_dir, d))]
        if not iter_dirs:
            if is_main_process(): print("No checkpoints found, starting from scratch.")
            return
        latest_iter = max(int(d.split('_')[1]) for d in iter_dirs)
        load_path = os.path.join(checkpoint_dir, f"iter_{latest_iter}")
        if is_main_process(): print(f"Resuming from latest checkpoint: {load_path}")

        self.actor.module.model.load_adapter(load_path, "default")
        self.critic.module.load_state_dict(torch.load(os.path.join(load_path, 'critic.pt'), map_location='cpu'))
        if self.use_trm and self.token_reward_model and os.path.exists(os.path.join(load_path, 'trm.pt')):
            self.token_reward_model.module.load_state_dict(
                torch.load(os.path.join(load_path, 'trm.pt'), map_location='cpu'))

        ac_optim_state_dict = torch.load(os.path.join(load_path, 'ac_optimizer.pt'), map_location='cpu',
                                         weights_only=True)
        self.ac_optimizer.load_state_dict(ac_optim_state_dict)
        if self.trm_optimizer and os.path.exists(os.path.join(load_path, 'trm_optimizer.pt')):
            trm_optim_state_dict = torch.load(os.path.join(load_path, 'trm_optimizer.pt'), map_location='cpu',
                                              weights_only=True)
            self.trm_optimizer.load_state_dict(trm_optim_state_dict)

        if is_main_process() and self.replay_buffer:
            self.replay_buffer.load(os.path.join(load_path, "replay_buffer.pkl"))
            with open(os.path.join(load_path, "metadata.json"), 'r') as f:
                metadata = json.load(f)
                self.start_iteration = metadata['iteration'] + 1
                self.timesteps = metadata['timesteps']

        self.start_iteration = int(broadcast_object(self.start_iteration if is_main_process() else 0))
        self.timesteps = int(broadcast_object(self.timesteps if is_main_process() else 0))

        dist.barrier()
        if is_main_process(): print(
            f"✅ Successfully resumed from iteration {self.start_iteration - 1}. Current timesteps: {self.timesteps}")