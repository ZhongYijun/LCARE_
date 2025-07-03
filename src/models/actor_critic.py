# # src/models/actor_critic.py (FINAL & OPTIMIZED)

# from typing import Dict, Tuple, Optional
# import torch
# import torch.nn as nn
# from omegaconf import DictConfig, OmegaConf
# from peft import LoraConfig, get_peft_model
# from torch.distributions import Categorical
# from transformers import AutoModelForCausalLM, AutoTokenizer

# from src.utils.distributed_utils import get_rank


# class LCARE_Actor(nn.Module):
#     """
#     [FINAL] L-CARE Actor Network.
#     - Accepts **kwargs for advanced features like Flash Attention.
#     - Implements FSDP-compatible manual generation.
#     """

#     def __init__(self, model_config: DictConfig, tokenizer: AutoTokenizer, **kwargs):
#         super().__init__()
#         self.config = model_config
#         self.tokenizer = tokenizer

#         print(f"Initializing Actor from path: {self.config.path}")
#         # Pass **kwargs to from_pretrained to enable features like Flash Attention
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.config.path, torch_dtype=torch.bfloat16, **kwargs
#         )

#         if self.config.get("use_lora", False):
#             print("Applying LoRA to the Actor model...")
#             lora_config_dict = OmegaConf.to_container(self.config.lora_config, resolve=True)
#             lora_config = LoraConfig(**lora_config_dict)
#             self.model = get_peft_model(self.model, lora_config)
#             if get_rank() == 0:
#                 self.model.print_trainable_parameters()

#     def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
#                 labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
#         logits = outputs.logits
#         dist = Categorical(logits=logits)
#         gather_labels = labels.clone()
#         gather_labels[gather_labels == -100] = 0
#         log_probs = dist.log_prob(gather_labels)
#         entropy = dist.entropy()
#         return log_probs, entropy

#     def forward_sft(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
#         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
#         return outputs.loss

#     @torch.no_grad()
#     def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, sampling_params: Dict) -> Tuple[
#         torch.Tensor, torch.Tensor]:
#         """
#         [FSDP-COMPATIBLE] Manually implemented autoregressive generation loop.
#         """
#         self.model.eval()

#         max_new_tokens = sampling_params.get("max_new_tokens", 64)
#         temperature = sampling_params.get("temperature", 1.0)

#         generated_ids = []
#         log_probs = []

#         current_ids = input_ids
#         current_attention_mask = attention_mask

#         for _ in range(max_new_tokens):
#             outputs = self.model(input_ids=current_ids, attention_mask=current_attention_mask)
#             next_token_logits = outputs.logits[:, -1, :]

#             if temperature > 0:
#                 next_token_logits = next_token_logits / temperature

#             dist = Categorical(logits=next_token_logits)
#             next_token = dist.sample()
#             log_prob = dist.log_prob(next_token)

#             generated_ids.append(next_token.unsqueeze(-1))
#             log_probs.append(log_prob.unsqueeze(-1))

#             current_ids = torch.cat([current_ids, next_token.unsqueeze(-1)], dim=-1)
#             current_attention_mask = torch.cat(
#                 [current_attention_mask, torch.ones_like(next_token.unsqueeze(-1))], dim=-1
#             )

#             if next_token.item() == self.tokenizer.eos_token_id:
#                 break

#         action_ids = torch.cat(generated_ids, dim=1)
#         behavior_log_prob = torch.cat(log_probs, dim=1).sum(dim=-1)

#         return action_ids, behavior_log_prob


# class LCARE_Critic(nn.Module):
#     """
#     [FINAL] L-CARE Critic Network.
#     - Accepts **kwargs for advanced features like Flash Attention.
#     """
#     def __init__(self, model_config: DictConfig, **kwargs):
#         super().__init__()
#         self.config = model_config
#         print(f"Initializing Transformer-based Critic from path: {self.config.path}")
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.config.path, torch_dtype=torch.bfloat16, **kwargs
#         )
#         self.value_head = nn.Linear(self.model.config.hidden_size, 1, bias=False)

#     def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
#         outputs = self.model(
#             input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False
#         )
#         last_hidden_state = outputs.hidden_states[-1]
#         batch_size = input_ids.shape[0]
#         sequence_lengths = torch.sum(attention_mask, dim=1) - 1
#         last_token_hidden_states = last_hidden_state[
#             torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
#         values = self.value_head(last_token_hidden_states)
#         return values


# class LCARE_TokenRewardModel(nn.Module):
#     """
#     [FINAL] L-CARE Token Reward Model.
#     - Accepts **kwargs for advanced features like Flash Attention.
#     """
#     def __init__(self, model_config: DictConfig, **kwargs):
#         super().__init__()
#         self.config = model_config
#         print(f"Initializing Token-level Reward Model from path: {self.config.path}")
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.config.path, torch_dtype=torch.bfloat16, **kwargs
#         )
#         self.reward_head = nn.Linear(self.model.config.hidden_size, 1, bias=False)

#     def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
#         outputs = self.model(
#             input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False
#         )
#         last_hidden_state = outputs.hidden_states[-1]
#         token_rewards = self.reward_head(last_hidden_state)
#         return token_rewards

# # src/models/actor_critic.py (FINAL & DTYPE-SAFE)

# from typing import Dict, Tuple, Optional
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from omegaconf import DictConfig, OmegaConf
# from peft import LoraConfig, get_peft_model
# from torch.distributions import Categorical
# from transformers import AutoModelForCausalLM, AutoTokenizer

# from src.utils.distributed_utils import get_rank


# class LCARE_Actor(nn.Module):
#     def __init__(self, model_config: DictConfig, tokenizer: AutoTokenizer, **kwargs):
#         super().__init__()
#         self.config = model_config
#         self.tokenizer = tokenizer
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.config.path, torch_dtype=torch.bfloat16, **kwargs
#         )
#         if self.config.get("use_lora", False):
#             lora_config_dict = OmegaConf.to_container(self.config.lora_config, resolve=True)
#             lora_config = LoraConfig(**lora_config_dict)
#             self.model = get_peft_model(self.model, lora_config)
#             if get_rank() == 0: self.model.print_trainable_parameters()

#     def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
#                 labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
#         logits = outputs.logits
#         dist = Categorical(logits=logits)
#         gather_labels = labels.clone()
#         gather_labels[gather_labels == -100] = 0
#         log_probs = dist.log_prob(gather_labels)
#         entropy = dist.entropy()
#         return log_probs, entropy

#     @torch.no_grad()
#     def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, sampling_params: Dict) -> Tuple[
#         torch.Tensor, torch.Tensor]:
#         self.model.eval()
#         max_new_tokens = sampling_params.get("max_new_tokens", 64)
#         temperature = sampling_params.get("temperature", 1.0)
#         top_p = sampling_params.get("top_p", 0.95)
#         generated_ids, log_probs = [], []
#         current_ids = input_ids
#         past_key_values = None

#         for _ in range(max_new_tokens):
#             outputs = self.model(
#                 input_ids=current_ids,
#                 use_cache=True,
#                 past_key_values=past_key_values,
#                 attention_mask=attention_mask
#             )
#             next_token_logits = outputs.logits[:, -1, :]
#             past_key_values = outputs.past_key_values
#             if temperature > 0: next_token_logits = next_token_logits / temperature
#             if top_p < 1.0:
#                 sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
#                 cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
#                 sorted_indices_to_remove = cumulative_probs > top_p
#                 sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
#                 sorted_indices_to_remove[..., 0] = 0
#                 indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
#                 next_token_logits[indices_to_remove] = -float('Inf')
#             dist = Categorical(logits=next_token_logits)
#             next_token = dist.sample()
#             log_prob = dist.log_prob(next_token)
#             generated_ids.append(next_token.unsqueeze(-1))
#             log_probs.append(log_prob.unsqueeze(-1))
#             current_ids = next_token.unsqueeze(-1)
#             attention_mask = torch.cat([attention_mask, torch.ones_like(current_ids)], dim=-1)
#             if next_token.item() == self.tokenizer.eos_token_id: break

#         if not generated_ids:
#             action_ids = torch.empty((input_ids.shape[0], 0), dtype=torch.long, device=input_ids.device)
#             behavior_log_prob = torch.zeros(input_ids.shape[0], device=input_ids.device)
#         else:
#             action_ids = torch.cat(generated_ids, dim=1)
#             behavior_log_prob = torch.cat(log_probs, dim=1).sum(dim=-1)
#         return action_ids, behavior_log_prob


# class LCARE_Critic(nn.Module):
#     def __init__(self, model_config: DictConfig, **kwargs):
#         super().__init__()
#         self.config = model_config
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.config.path, torch_dtype=torch.bfloat16, **kwargs
#         )
#         # --- [CRITICAL FIX] ---
#         # 确保 value_head 的 dtype 与基础模型一致
#         self.value_head = nn.Linear(
#             self.model.config.hidden_size, 1, bias=False,
#             dtype=self.model.dtype # 显式指定dtype
#         )
#         # --- [END OF FIX] ---

#     def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
#         outputs = self.model(
#             input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False
#         )
#         last_hidden_state = outputs.hidden_states[-1]
#         batch_size = input_ids.shape[0]
#         sequence_lengths = torch.sum(attention_mask, dim=1) - 1
#         last_token_hidden_states = last_hidden_state[
#             torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
#         values = self.value_head(last_token_hidden_states)
#         return values


# class LCARE_TokenRewardModel(nn.Module):
#     def __init__(self, model_config: DictConfig, **kwargs):
#         super().__init__()
#         self.config = model_config
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.config.path, torch_dtype=torch.bfloat16, **kwargs
#         )
#         # --- [CRITICAL FIX] ---
#         # 确保 reward_head 的 dtype 与基础模型一致
#         self.reward_head = nn.Linear(
#             self.model.config.hidden_size, 1, bias=False,
#             dtype=self.model.dtype # 显式指定dtype
#         )
#         # --- [END OF FIX] ---

#     def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
#         outputs = self.model(
#             input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False
#         )
#         last_hidden_state = outputs.hidden_states[-1]
#         token_rewards = self.reward_head(last_hidden_state)
#         return token_rewards


# src/models/actor_critic.py (最终的、完整的、与新架构100%匹配的版本)

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from torch.distributions import Categorical
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from src.utils.distributed_utils import get_rank


class LCARE_Actor(nn.Module):
    """
    L-CARE Actor网络。
    包含了用于SFT的forward_sft、用于PPO训练的forward，以及用于数据采集的generate方法。
    """
    def __init__(self, model_config: DictConfig, tokenizer: AutoTokenizer, **kwargs):
        super().__init__()
        self.config = model_config
        self.tokenizer = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.path, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True, 
            **kwargs
        )
        if self.config.get("use_lora", False):
            lora_config_dict = OmegaConf.to_container(self.config.lora_config, resolve=True)
            lora_config = LoraConfig(**lora_config_dict)
            self.model = get_peft_model(self.model, lora_config)
            if get_rank() == 0: self.model.print_trainable_parameters()

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """用于PPO训练的核心forward方法，计算新策略的对数概率和熵。"""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits
        dist = Categorical(logits=logits)
        
        # 克隆labels以避免原地修改
        gather_labels = labels.clone()
        # 将-100替换为0，以便可以从dist中gather，这不会影响最终结果，因为这些位置的log_prob不会被使用
        gather_labels[gather_labels == -100] = 0 
        
        log_probs = dist.log_prob(gather_labels)
        entropy = dist.entropy()
        return log_probs, entropy
    
    def forward_sft(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """用于SFT训练的forward方法，直接返回损失。"""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
        return outputs.loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, sampling_params: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [最终健壮版] 用于数据采集的生成方法。
        使用Hugging Face官方的generate函数，并精确计算生成序列的对数概率。
        """
        self.model.eval()

        # 使用GenerationConfig来管理所有生成参数，这是最稳健的方式
        generation_config = GenerationConfig(
            max_new_tokens=sampling_params.get("max_new_tokens", 1024),
            do_sample=sampling_params.get("do_sample", True),
            temperature=sampling_params.get("temperature", 0.7),
            top_p=sampling_params.get("top_p", 0.9),
            # 正确传递可能是列表的eos_token_id
            eos_token_id=sampling_params.get("eos_token_id", self.tokenizer.eos_token_id),
            # 确保pad_token_id也被传递，以在生成时抑制警告
            pad_token_id=self.tokenizer.pad_token_id,
        )

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=True, # 必须为True才能获取scores
            output_scores=True,           # 必须为True才能计算logprobs
        )

        # `generate`返回的sequences包含了输入的prompt，我们需要把它去掉
        action_ids = outputs.sequences[:, input_ids.shape[-1]:]

        # `scores`是一个元组，包含了每一步生成的token的logits
        # 我们需要用它来计算实际生成token的对数概率
        log_probs_list = []
        # `scores`的长度等于生成的token数量
        for i, score in enumerate(outputs.scores):
            # 获取在第i步实际生成的token的ID
            # action_ids的形状是 (batch_size, sequence_length)
            action_id_at_step_i = action_ids[:, i].unsqueeze(-1)
            # score的形状是 (batch_size, vocab_size)
            # 计算log_softmax以得到对数概率
            log_prob_at_step_i = F.log_softmax(score, dim=-1).gather(1, action_id_at_step_i)
            log_probs_list.append(log_prob_at_step_i)

        if not log_probs_list:
            # 如果没有生成任何token（例如，输入就是EOS）
            return (
                torch.empty(input_ids.shape[0], 0, dtype=torch.long, device=input_ids.device), 
                torch.zeros(input_ids.shape[0], device=input_ids.device)
            )

        # 将每一步的log_prob连接起来 (batch_size, sequence_length)
        # 然后在序列维度上求和，得到每个样本的最终log_prob
        behavior_log_prob = torch.cat(log_probs_list, dim=-1).sum(dim=-1)

        return action_ids, behavior_log_prob


class LCARE_Critic(nn.Module):
    def __init__(self, model_config: DictConfig, **kwargs):
        super().__init__()
        self.config = model_config
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.path, torch_dtype=torch.bfloat16, **kwargs
        )
        self.value_head = nn.Linear(self.model.config.hidden_size, 1, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False
        )
        last_hidden_state = outputs.hidden_states[-1]
        batch_size = input_ids.shape[0]
        sequence_lengths = torch.sum(attention_mask, dim=1) - 1
        last_token_hidden_states = last_hidden_state[
            torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        values = self.value_head(last_token_hidden_states)
        return values


class LCARE_TokenRewardModel(nn.Module):
    def __init__(self, model_config: DictConfig, **kwargs):
        super().__init__()
        self.config = model_config
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.path, torch_dtype=torch.bfloat16, **kwargs
        )
        self.reward_head = nn.Linear(self.model.config.hidden_size, 1, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False
        )
        last_hidden_state = outputs.hidden_states[-1]
        token_rewards = self.reward_head(last_hidden_state)
        return token_rewards