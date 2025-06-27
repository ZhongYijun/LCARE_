# # src/models/actor_critic.py
#
# from typing import Dict, Tuple, Optional
# import torch
# import torch.nn as nn
# from omegaconf import DictConfig, OmegaConf
# from peft import LoraConfig, PeftModel, get_peft_model
# from torch.distributions import Categorical
# from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
#
#
# class LCARE_Actor(nn.Module):
#     """[V-FINAL] L-CARE的策略网络 (Actor)。"""
#
#     def __init__(self, model_config: DictConfig, **kwargs):
#         super().__init__()
#         self.config = model_config
#
#         print(f"Initializing Actor from path: {self.config.path}")
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.config.path, trust_remote_code=True, padding_side='left'
#         )
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.config.path, torch_dtype=torch.bfloat16, trust_remote_code=True
#         )
#
#         if self.config.get("use_lora", False):
#             print("Applying LoRA to the Actor model...")
#             lora_config_dict = OmegaConf.to_container(self.config.lora_config, resolve=True)
#             lora_config = LoraConfig(**lora_config_dict)
#             self.model = get_peft_model(self.model, lora_config)
#             self.model.print_trainable_parameters()
#
#     def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
#                 labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         """PPO Actor前向传播，计算log_prob和熵。"""
#         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
#         logits = outputs.logits
#         dist = Categorical(logits=logits)
#
#         gather_labels = labels.clone()
#         gather_labels[gather_labels == -100] = 0
#         log_probs = dist.log_prob(gather_labels)
#         entropy = dist.entropy()
#         return log_probs, entropy
#
#     def forward_sft(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
#         """[NEW] 为行为克隆损失计算SFT损失。"""
#         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
#         return outputs.loss
#
#     @torch.no_grad()
#     def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, sampling_params: Dict) -> Tuple[
#         torch.Tensor, torch.Tensor]:
#         """数据采集的生成函数。"""
#         self.model.eval()
#         generation_config = GenerationConfig(
#             pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id, **sampling_params,
#         )
#         outputs = self.model.generate(
#             input_ids=input_ids, attention_mask=attention_mask, generation_config=generation_config,
#             output_scores=True, return_dict_in_generate=True,
#         )
#         action_ids = outputs.sequences[:, input_ids.shape[1]:]
#
#         stacked_scores = torch.stack(outputs.scores, dim=1)
#         dist = Categorical(logits=stacked_scores)
#         action_ids_for_log_prob = action_ids.clone()
#         action_ids_for_log_prob[action_ids_for_log_prob == self.tokenizer.pad_token_id] = 0
#         log_probs_per_token = dist.log_prob(action_ids_for_log_prob)
#
#         action_mask = (action_ids != self.tokenizer.pad_token_id).long()
#         log_probs = (log_probs_per_token * action_mask).sum(dim=-1)
#         return action_ids, log_probs
#
#
# class LCARE_Critic(nn.Module):
#     """[V-FINAL] L-CARE的上下文感知价值网络 (Critic)。"""
#
#     def __init__(self, model_config: DictConfig):
#         super().__init__()
#         self.config = model_config
#         print(f"Initializing Transformer-based Critic from path: {self.config.path}")
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.config.path, torch_dtype=torch.bfloat16, trust_remote_code=True,
#         )
#         self.value_head = nn.Linear(self.model.config.hidden_size, 1, bias=False)
#
#     def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
#         """前向传播，计算给定状态序列的价值。"""
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
#
#
# class LCARE_TokenRewardModel(nn.Module):
#     """[V-FINAL] L-CARE的Token级奖励模型 (TRM)。"""
#
#     def __init__(self, model_config: DictConfig):
#         super().__init__()
#         self.config = model_config
#         print(f"Initializing Token-level Reward Model from path: {self.config.path}")
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.config.path, torch_dtype=torch.bfloat16, trust_remote_code=True,
#         )
#         self.reward_head = nn.Linear(self.model.config.hidden_size, 1, bias=False)
#
#     def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
#         """为输入序列的每个token计算奖励值。"""
#         outputs = self.model(
#             input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False
#         )
#         last_hidden_state = outputs.hidden_states[-1]
#         token_rewards = self.reward_head(last_hidden_state)
#         return token_rewards

# src/models/actor_critic.py

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from torch.distributions import Categorical
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from src.utils.distributed_utils import get_rank


class LCARE_Actor(nn.Module):
    """
    L-CARE的策略网络 (Actor)。
    通过依赖注入接收Tokenizer，实现了与PromptConstructor等组件的解耦。
    """

    def __init__(self, model_config: DictConfig, tokenizer: AutoTokenizer, **kwargs):
        super().__init__()
        self.config = model_config

        # 使用注入的Tokenizer实例
        self.tokenizer = tokenizer

        print(f"Initializing Actor from path: {self.config.path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )

        if self.config.get("use_lora", False):
            print("Applying LoRA to the Actor model...")
            lora_config_dict = OmegaConf.to_container(self.config.lora_config, resolve=True)
            lora_config = LoraConfig(**lora_config_dict)
            self.model = get_peft_model(self.model, lora_config)
            if get_rank() == 0:
                self.model.print_trainable_parameters()

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """PPO Actor前向传播，计算log_prob和熵。"""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits
        dist = Categorical(logits=logits)

        gather_labels = labels.clone()
        gather_labels[gather_labels == -100] = 0
        log_probs = dist.log_prob(gather_labels)
        entropy = dist.entropy()
        return log_probs, entropy

    def forward_sft(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """为行为克隆计算SFT损失。"""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
        return outputs.loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, sampling_params: Dict) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """数据采集的生成函数。"""
        self.model.eval()
        generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id, **sampling_params,
        )
        outputs = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask, generation_config=generation_config,
            output_scores=True, return_dict_in_generate=True,
        )
        action_ids = outputs.sequences[:, input_ids.shape[1]:]

        stacked_scores = torch.stack(outputs.scores, dim=1)
        dist = Categorical(logits=stacked_scores)
        action_ids_for_log_prob = action_ids.clone()
        action_ids_for_log_prob[action_ids_for_log_prob == self.tokenizer.pad_token_id] = 0
        log_probs_per_token = dist.log_prob(action_ids_for_log_prob)

        action_mask = (action_ids != self.tokenizer.pad_token_id).long()
        log_probs = (log_probs_per_token * action_mask).sum(dim=-1)
        return action_ids, log_probs


class LCARE_Critic(nn.Module):
    """[V-FINAL] L-CARE的上下文感知价值网络 (Critic)。"""

    def __init__(self, model_config: DictConfig):
        super().__init__()
        self.config = model_config
        print(f"Initializing Transformer-based Critic from path: {self.config.path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.path, torch_dtype=torch.bfloat16, trust_remote_code=True,
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
    """ --- L-CARE的Token级奖励模型 (TRM)。"""

    def __init__(self, model_config: DictConfig):
        super().__init__()
        self.config = model_config
        print(f"Initializing Token-level Reward Model from path: {self.config.path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
        self.reward_head = nn.Linear(self.model.config.hidden_size, 1, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False
        )
        last_hidden_state = outputs.hidden_states[-1]
        token_rewards = self.reward_head(last_hidden_state)
        return token_rewards

