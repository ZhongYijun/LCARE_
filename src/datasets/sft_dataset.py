# # src/datasets/sft_dataset.py

# import pandas as pd
# import torch
# from torch.utils.data import Dataset
# from transformers import AutoTokenizer
# from typing import List, Dict, Any


# class SFTDataset(Dataset):
#     """[V-FINAL] 用于SFT阶段的PyTorch Dataset类, 已优化为不提前padding。"""

#     def __init__(self, parquet_path: str, tokenizer: AutoTokenizer, max_length: int = 8192):
#         try:
#             self.df = pd.read_parquet(parquet_path)
#         except Exception as e:
#             raise FileNotFoundError(f"Could not read SFT parquet file at {parquet_path}.") from e

#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx: int) -> dict:
#         row = self.df.iloc[idx]
#         problem = row['problem']
#         solution = row['solution_cot']

#         conversation = [
#             {"role": "user", "content": problem},
#             {"role": "assistant", "content": solution}
#         ]

#         prompt_only_text = self.tokenizer.apply_chat_template(
#             [conversation[0]], tokenize=False, add_generation_prompt=True
#         )
#         full_text = self.tokenizer.apply_chat_template(
#             conversation, tokenize=False, add_generation_prompt=False
#         ) + self.tokenizer.eos_token

#         # 只进行tokenize，不进行padding
#         tokenized_prompt = self.tokenizer(prompt_only_text, truncation=True, max_length=self.max_length)
#         tokenized_full = self.tokenizer(full_text, truncation=True, max_length=self.max_length)

#         prompt_len = len(tokenized_prompt.input_ids)

#         # 将input_ids转换为torch.Tensor
#         input_ids = torch.LongTensor(tokenized_full.input_ids)
#         attention_mask = torch.LongTensor(tokenized_full.attention_mask)

#         labels = input_ids.clone()
#         labels[:prompt_len] = -100  # -100 会在计算损失时被忽略

#         return {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "labels": labels
#         }


# def collate_fn_sft(batch: List[Dict], tokenizer: AutoTokenizer) -> Dict[str, Any]:
#     """
#     [CORRECTED] 为SFT Dataloader实现动态padding。
#     它会将一个batch的所有样本填充到该batch中最长样本的长度。
#     """
#     # 确保tokenizer的padding_side是right，这对于Causal LM的训练是标准做法
#     tokenizer.padding_side = 'right'

#     # 从batch中分离出需要padding的张量
#     input_ids_list = [item['input_ids'] for item in batch]
#     attention_mask_list = [item['attention_mask'] for item in batch]
#     labels_list = [item['labels'] for item in batch]

#     # [CRITICAL FIX] 将所有需要padding的项放入一个字典中，并只调用一次tokenizer.pad()
#     to_pad = {
#         "input_ids": input_ids_list,
#         "attention_mask": attention_mask_list
#     }
    
#     # 一次性调用 padding，同时处理 input_ids 和 attention_mask
#     padded_batch = tokenizer.pad(
#         to_pad,
#         return_tensors="pt",
#         padding='longest',
#     )

#     # labels需要手动进行padding，因为它的padding值是-100，而不是tokenizer的pad_token_id
#     padded_labels = torch.nn.utils.rnn.pad_sequence(
#         labels_list, batch_first=True, padding_value=-100
#     )
    
#     # 将手动padding的labels添加到最终的batch中
#     padded_batch['labels'] = padded_labels

#     return padded_batch

# src/trainers/sft_trainer.py

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
    [V-FINAL-DEADLOCK-FIX] SFT训练器。
    - 采用最终的、最健壮的模型保存逻辑，以彻底解决FSDP死锁问题。
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
        if is_main_process(): print(f"Loading base model from {model_path} for SFT...")
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
        model.gradient_checkpointing_enable()
        if use_lora:
            if is_main_process(): print("Applying LoRA...")
            if not hasattr(self.model_config, 'lora_config'): raise ValueError(
                "`use_lora` is true, but `lora_config` is not defined.")
            lora_config_dict = OmegaConf.to_container(self.model_config.lora_config, resolve=True)
            lora_config = LoraConfig(**lora_config_dict)
            model = get_peft_model(model, lora_config)
            if is_main_process(): model.print_trainable_parameters()
        try:
            transformer_block_class = self._get_transformer_layer_class(model)
            if is_main_process(): print(
                f"Found transformer layer class to wrap with FSDP: {transformer_block_class.__name__}")
        except ValueError:
            if dist.is_initialized(): dist.destroy_process_group()
            exit(1)
        transformer_layer_cls_set = {transformer_block_class}
        auto_wrap_policy = functools.partial(transformer_auto_wrap_policy,
                                             transformer_layer_cls=transformer_layer_cls_set)
        fsdp_dtype = torch.bfloat16
        model = model.to(dtype=fsdp_dtype)
        mixed_precision_policy = torch.distributed.fsdp.MixedPrecision(param_dtype=fsdp_dtype, reduce_dtype=fsdp_dtype,
                                                                       buffer_dtype=fsdp_dtype)
        self.model = FSDP(model, auto_wrap_policy=auto_wrap_policy, device_id=self.device, use_orig_params=use_lora,
                          mixed_precision=mixed_precision_policy)
        dist.barrier(device_ids=[self.rank])
        if is_main_process(): print("✅ FSDP model wrapped successfully.")
        train_dataset = SFTDataset(self.sft_config.sft_data_path, self.tokenizer, max_length=self.sft_config.max_length)
        train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        collate_with_tokenizer = partial(collate_fn_sft, tokenizer=self.tokenizer)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.sft_config.batch_size_per_gpu,
                                           sampler=train_sampler, num_workers=self.sft_config.get("num_workers", 4),
                                           pin_memory=True, collate_fn=collate_with_tokenizer)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.sft_config.learning_rate)
        num_training_steps = self.sft_config.epochs * len(self.train_dataloader)
        self.lr_scheduler = get_scheduler(name="cosine", optimizer=self.optimizer,
                                          num_warmup_steps=int(0.03 * num_training_steps),
                                          num_training_steps=num_training_steps)

    @staticmethod
    def _get_transformer_layer_class(model: PreTrainedModel) -> type:
        if isinstance(model, PeftModel): model = model.get_base_model()
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
        """
        [FINAL DEADLOCK FIX] 使用最健壮的方式保存FSDP + LoRA 模型。
        """
        output_dir = self.sft_config.output_dir
        use_lora = self.sft_config.get("use_lora", self.model_config.get("use_lora", False))

        if is_main_process():
            print(f"Starting model saving process to {output_dir}...")
            os.makedirs(output_dir, exist_ok=True)

        # 确保所有进程都准备好开始保存
        dist.barrier(device_ids=[self.rank])

        # 1. 获取完整的、在CPU上的state_dict (仅rank 0拥有)
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state_dict = self.model.state_dict()

        # 2. 在所有进程都完成了state_dict的收集后，只有主进程进行文件写入
        if is_main_process():
            print("Model state dict collected on Rank 0. Now saving to disk...")

            if use_lora:
                # [NEW LOGIC] 手动过滤出LoRA权重并保存，而不是调用Peft的save_pretrained
                lora_weights = {k: v for k, v in cpu_state_dict.items() if "lora_" in k}
                if not lora_weights:
                    print("WARNING: `use_lora` is true, but no LoRA weights found in the state dict.")

                # 保存adapter权重和配置文件
                torch.save(lora_weights, os.path.join(output_dir, "adapter_model.bin"))

                # 从原始模型获取并保存adapter_config.json
                # self.model.module是获取FSDP包装下的原始模型
                if isinstance(self.model.module, PeftModel):
                    self.model.module.peft_config['default'].save_pretrained(output_dir)
                print("LoRA adapter weights and config saved manually.")

            else:
                # 如果不使用LoRA，则保存完整的模型
                torch.save(cpu_state_dict, os.path.join(output_dir, "pytorch_model.bin"))
                print("Full model state dict saved.")

            # 保存tokenizer
            self.tokenizer.save_pretrained(output_dir)
            print("Tokenizer saved.")
            print(f"✅ SFT model saved successfully to {output_dir}.")

        # 3. 最终屏障，确保rank 0写完文件后，所有进程才一起退出此函数
        # dist.barrier(device_ids=[self.rank])