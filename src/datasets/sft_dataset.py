# # src/datasets/sft_dataset.py
# import pandas as pd
#
# from torch.utils.data import Dataset
# from transformers import AutoTokenizer
#
#
# class SFTDataset(Dataset):
#     """用于SFT阶段的PyTorch Dataset类"""
#
#     def __init__(self, parquet_path: str, tokenizer: AutoTokenizer, max_length: int = 2048):
#         try:
#             self.df = pd.read_parquet(parquet_path)
#         except Exception as e:
#             raise FileNotFoundError(
#                 f"Could not read SFT parquet file at {parquet_path}. Please run data processing first.") from e
#
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         # 确保pad_token已设置
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, idx: int) -> dict:
#         row = self.df.iloc[idx]
#         problem = row['problem']
#         solution = row['solution_cot']
#
#         # 使用apply_chat_template来构建对话格式，更具通用性
#         conversation = [
#             {"role": "user", "content": problem},
#             {"role": "assistant", "content": solution}
#         ]
#         full_text = self.tokenizer.apply_chat_template(
#             conversation,
#             tokenize=False,
#             add_generation_prompt=False  # 不添加 assistant 后缀
#         ) + self.tokenizer.eos_token
#
#         # Tokenize
#         tokenized = self.tokenizer(
#             full_text,
#             truncation=True,
#             max_length=self.max_length,
#             padding="max_length",  # 填充到最大长度
#             return_tensors="pt"
#         )
#
#         # 为了计算损失，我们需要一个labels张量
#         # labels是input_ids的副本，但prompt部分被-100掩码
#         # 1. Tokenize prompt to find its length
#         prompt_only_text = self.tokenizer.apply_chat_template(
#             [conversation[0]],  # 只有user部分
#             tokenize=False,
#             add_generation_prompt=True  # 添加assistant后缀，以准确计算长度
#         )
#         prompt_tokens = self.tokenizer(prompt_only_text, return_tensors="pt").input_ids
#         prompt_len = prompt_tokens.shape[1]
#
#         labels = tokenized['input_ids'].clone()
#         labels[:, :prompt_len] = -100
#
#         # 将padding部分的label也设为-100
#         labels[tokenized['input_ids'] == self.tokenizer.pad_token_id] = -100
#
#         return {
#             "input_ids": tokenized['input_ids'].squeeze(0),
#             "attention_mask": tokenized['attention_mask'].squeeze(0),
#             "labels": labels.squeeze(0)
#         }
#
#
# src/datasets/sft_dataset.py
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Any


class SFTDataset(Dataset):
    """[V-FINAL] 用于SFT阶段的PyTorch Dataset类, 已优化为不提前padding。"""

    def __init__(self, parquet_path: str, tokenizer: AutoTokenizer, max_length: int = 8192):
        try:
            self.df = pd.read_parquet(parquet_path)
        except Exception as e:
            raise FileNotFoundError(f"Could not read SFT parquet file at {parquet_path}.") from e

        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        problem = row['problem']
        solution = row['solution_cot']

        conversation = [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution}
        ]

        prompt_only_text = self.tokenizer.apply_chat_template(
            [conversation[0]], tokenize=False, add_generation_prompt=True
        )
        full_text = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        ) + self.tokenizer.eos_token

        # 只进行tokenize，不进行padding
        tokenized_prompt = self.tokenizer(prompt_only_text, truncation=True, max_length=self.max_length)
        tokenized_full = self.tokenizer(full_text, truncation=True, max_length=self.max_length)

        prompt_len = len(tokenized_prompt.input_ids)

        # 将input_ids转换为torch.Tensor
        input_ids = torch.LongTensor(tokenized_full.input_ids)
        attention_mask = torch.LongTensor(tokenized_full.attention_mask)

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # -100 会在计算损失时被忽略

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def collate_fn_sft(batch: List[Dict], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """
    为SFT Dataloader实现动态padding。
    它会将一个batch的所有样本填充到该batch中最长样本的长度。
    """
    # [修复] 确保tokenizer的padding_side是right，这对于Causal LM的训练是标准做法
    tokenizer.padding_side = 'right'

    # 从batch中分离出需要padding的张量
    input_ids_list = [item['input_ids'] for item in batch]
    attention_mask_list = [item['attention_mask'] for item in batch]
    labels_list = [item['labels'] for item in batch]

    # 使用tokenizer的pad方法进行高效padding
    padded_inputs = tokenizer.pad(
        {"input_ids": input_ids_list},
        return_tensors="pt",
        padding='longest',
    )
    padded_masks = tokenizer.pad(
        {"attention_mask": attention_mask_list},
        return_tensors="pt",
        padding='longest',
    )
    # labels需要手动padding
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels_list, batch_first=True, padding_value=-100
    )

    return {
        'input_ids': padded_inputs['input_ids'],
        'attention_mask': padded_masks['attention_mask'],
        'labels': padded_labels
    }