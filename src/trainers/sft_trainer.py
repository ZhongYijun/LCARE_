# # src/trainers/sft_trainer.py
#
# import os
# import functools
# import torch
# import torch.distributed as dist
# from torch.utils.data import DataLoader, DistributedSampler
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
#
# from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
# # from torch.distributed.fsdp.state_dict_utils import FullStateDictConfig
# from torch.distributed.fsdp import FullStateDictConfig
# from peft import get_peft_model, LoraConfig, PeftModel
# from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, PreTrainedModel
# from omegaconf import DictConfig, OmegaConf
# from tqdm import tqdm
#
# from src.datasets.sft_dataset import SFTDataset
# from src.utils.logger import WandbLogger
# from src.utils.distributed_utils import is_main_process, get_rank
#
#
# class SFTTrainer:
#     """
#     一个用于对大语言模型进行监督微调的训练器，支持FSDP和LoRA。
#     """
#
#     def __init__(self, config: DictConfig, rank: int, world_size: int):
#         self.config = config
#         self.sft_config = config.trainer
#         self.rank = rank
#         self.world_size = world_size
#         self.device = torch.device(f"cuda:{rank}")
#
#         self.logger = WandbLogger(config, self.rank)
#
#         if is_main_process(): print(f"Loading tokenizer from {self.sft_config.model_path}")
#         self.tokenizer = AutoTokenizer.from_pretrained(self.sft_config.model_path, trust_remote_code=True)
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#
#         if is_main_process(): print(f"Loading model from {self.sft_config.model_path}")
#         model = AutoModelForCausalLM.from_pretrained(
#             self.sft_config.model_path,
#             torch_dtype=torch.bfloat16,
#             trust_remote_code=True
#         )
#
#         if self.sft_config.use_lora:
#             if is_main_process(): print("Applying LoRA...")
#             lora_config_dict = OmegaConf.to_container(self.sft_config.lora_config, resolve=True)
#             lora_config = LoraConfig(**lora_config_dict)
#             model = get_peft_model(model, lora_config)
#             if is_main_process():
#                 model.print_trainable_parameters()
#
#         # [修复] 将辅助方法标记为静态方法
#         transformer_layer_name = self._get_transformer_layer_name(model)
#         if is_main_process(): print(f"Found transformer layer to wrap with FSDP: {transformer_layer_name}")
#
#         auto_wrap_policy = functools.partial(
#             transformer_auto_wrap_policy,
#             transformer_layer_cls={
#                 layer for layer in model.modules() if layer.__class__.__name__ == transformer_layer_name
#             },
#         )
#
#         self.model = FSDP(
#             model,
#             auto_wrap_policy=auto_wrap_policy,
#             device_id=self.device,
#             use_orig_params=self.sft_config.use_lora,
#             mixed_precision=torch.distributed.fsdp.MixedPrecision(
#                 param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
#             )
#         )
#         dist.barrier()
#         if is_main_process(): print("✅ FSDP model wrapped.")
#
#         train_dataset = SFTDataset(self.sft_config.sft_data_path, self.tokenizer, max_length=self.sft_config.max_length)
#         train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
#         self.train_dataloader = DataLoader(
#             train_dataset, batch_size=self.sft_config.batch_size_per_gpu, sampler=train_sampler,
#             num_workers=self.sft_config.get("num_workers", 4), pin_memory=True
#         )
#
#         self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.sft_config.learning_rate)
#         num_training_steps = self.sft_config.epochs * len(self.train_dataloader)
#         self.lr_scheduler = get_scheduler(
#             name="cosine",
#             optimizer=self.optimizer,
#             num_warmup_steps=int(0.03 * num_training_steps),
#             num_training_steps=num_training_steps,
#         )
#
#     @staticmethod
#     def _get_transformer_layer_name(model: PreTrainedModel) -> str:
#         """动态查找模型中的transformer block/layer类名"""
#         for name, module in model.named_modules():
#             if "DecoderLayer" in module.__class__.__name__ or "TransformerBlock" in module.__class__.__name__:
#                 return module.__class__.__name__
#         raise ValueError("Could not find a valid transformer layer class name in the model.")
#
#     def train(self):
#         """主训练循环"""
#         self.model.train()
#         total_steps = 0
#         for epoch in range(self.sft_config.epochs):
#             assert isinstance(self.train_dataloader.sampler, DistributedSampler)
#             self.train_dataloader.sampler.set_epoch(epoch)
#
#             pbar = tqdm(self.train_dataloader, disable=(not is_main_process()),
#                         desc=f"SFT Epoch {epoch + 1}/{self.sft_config.epochs} | Rank {get_rank()}")
#
#             for batch in pbar:
#                 batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
#                 outputs = self.model(**batch)
#                 loss = outputs.loss
#                 loss.backward()
#                 self.optimizer.step()
#                 self.lr_scheduler.step()
#                 self.optimizer.zero_grad()
#                 total_steps += 1
#                 if is_main_process():
#                     pbar.set_postfix({"loss": loss.item()})
#                     self.logger.log({"sft/loss": loss.item(), "sft/lr": self.lr_scheduler.get_last_lr()[0]},
#                                     step=total_steps)
#             dist.barrier()
#
#         self.save_model()
#         if is_main_process(): self.logger.finish()
#
#     def save_model(self):
#         """使用FSDP推荐的方式保存模型和LoRA权重"""
#         if is_main_process():
#             print(f"Saving SFT model to {self.sft_config.output_dir}...")
#             os.makedirs(self.sft_config.output_dir, exist_ok=True)
#         dist.barrier()
#
#         save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
#         with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
#             cpu_state_dict = self.model.state_dict()
#             if is_main_process():
#                 unwrapped_model = self.model.module
#                 if self.sft_config.use_lora and isinstance(unwrapped_model, PeftModel):
#                     unwrapped_model.save_pretrained(self.sft_config.output_dir)
#                 else:
#                     torch.save(cpu_state_dict, os.path.join(self.sft_config.output_dir, "pytorch_model.bin"))
#                 self.tokenizer.save_pretrained(self.sft_config.output_dir)
#
#         dist.barrier()
#         if is_main_process(): print(f"✅ SFT model saved successfully.")

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

# [修复] 导入 SFTDataset 和我们为其专门设计的 collate_fn
from src.datasets.sft_dataset import SFTDataset, collate_fn_sft
from src.utils.logger import WandbLogger
from src.utils.distributed_utils import is_main_process, get_rank


class SFTTrainer:
    """
    [V-FINAL-EFFICIENT] 一个用于对大语言模型进行监督微调的训练器。
    - 支持FSDP和LoRA。
    - 使用动态填充（Dynamic Padding）以实现最高训练效率。
    """

    def __init__(self, config: DictConfig, rank: int, world_size: int):
        self.config = config
        self.sft_config = config.trainer
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")

        self.logger = WandbLogger(config, self.rank)

        if is_main_process(): print(f"Loading tokenizer from {self.sft_config.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.sft_config.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if is_main_process(): print(f"Loading model from {self.sft_config.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            self.sft_config.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        if self.sft_config.use_lora:
            if is_main_process(): print("Applying LoRA...")
            lora_config_dict = OmegaConf.to_container(self.sft_config.lora_config, resolve=True)
            lora_config = LoraConfig(**lora_config_dict)
            model = get_peft_model(model, lora_config)
            if is_main_process():
                model.print_trainable_parameters()

        transformer_layer_name = self._get_transformer_layer_name(model)
        if is_main_process(): print(f"Found transformer layer to wrap with FSDP: {transformer_layer_name}")

        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                layer for layer in model.modules() if layer.__class__.__name__ == transformer_layer_name
            },
        )

        self.model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            device_id=self.device,
            use_orig_params=self.sft_config.use_lora,
            mixed_precision=torch.distributed.fsdp.MixedPrecision(
                param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
            )
        )
        dist.barrier()
        if is_main_process(): print("✅ FSDP model wrapped.")

        train_dataset = SFTDataset(self.sft_config.sft_data_path, self.tokenizer, max_length=self.sft_config.max_length)
        train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)

        # [修复] 将tokenizer绑定到collate_fn，以实现动态padding
        collate_with_tokenizer = partial(collate_fn_sft, tokenizer=self.tokenizer)

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.sft_config.batch_size_per_gpu,
            sampler=train_sampler,
            num_workers=self.sft_config.get("num_workers", 4),
            pin_memory=True,
            collate_fn=collate_with_tokenizer  # <-- 使用动态padding
        )

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.sft_config.learning_rate)
        num_training_steps = self.sft_config.epochs * len(self.train_dataloader)
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=int(0.03 * num_training_steps),
            num_training_steps=num_training_steps,
        )

    @staticmethod
    def _get_transformer_layer_name(model: PreTrainedModel) -> str:
        """动态查找模型中的transformer block/layer类名"""
        for name, module in model.named_modules():
            if "DecoderLayer" in module.__class__.__name__ or "TransformerBlock" in module.__class__.__name__:
                return module.__class__.__name__
        raise ValueError("Could not find a valid transformer layer class name in the model.")

    def train(self):
        """主训练循环"""
        self.model.train()
        total_steps = 0
        for epoch in range(self.sft_config.epochs):
            assert isinstance(self.train_dataloader.sampler, DistributedSampler), "Sampler is not a DistributedSampler"
            self.train_dataloader.sampler.set_epoch(epoch)

            pbar = tqdm(self.train_dataloader, disable=(not is_main_process()),
                        desc=f"SFT Epoch {epoch + 1}/{self.sft_config.epochs} | Rank {get_rank()}")

            for batch in pbar:
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                total_steps += 1
                if is_main_process():
                    pbar.set_postfix({"loss": loss.item()})
                    self.logger.log({"sft/loss": loss.item(), "sft/lr": self.lr_scheduler.get_last_lr()[0]},
                                    step=total_steps)
            dist.barrier()

        self.save_model()
        if is_main_process(): self.logger.finish()

    def save_model(self):
        """使用FSDP推荐的方式保存模型和LoRA权重"""
        if is_main_process():
            print(f"Saving SFT model to {self.sft_config.output_dir}...")
            os.makedirs(self.sft_config.output_dir, exist_ok=True)
        dist.barrier()

        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state_dict = self.model.state_dict()
            if is_main_process():
                unwrapped_model = self.model.module
                if self.sft_config.use_lora and isinstance(unwrapped_model, PeftModel):
                    unwrapped_model.save_pretrained(self.sft_config.output_dir)
                else:
                    torch.save(cpu_state_dict, os.path.join(self.sft_config.output_dir, "pytorch_model.bin"))
                self.tokenizer.save_pretrained(self.sft_config.output_dir)

        dist.barrier()
        if is_main_process(): print(f"✅ SFT model saved successfully.")