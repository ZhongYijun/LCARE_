# src/datasets/sft_dataset.py

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Any


class SFTDataset(Dataset):
    """
    [V-FINAL-STABLE] 用于SFT阶段的PyTorch Dataset类。
    - 解决了OOM问题，将分词操作放回__getitem__。
    - 修复了所有已知的错误。
    """

    def __init__(self, parquet_path: str, tokenizer: AutoTokenizer, max_length: int = 2048): # 建议使用更小的max_length
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
        """
        [STABLE VERSION] 对单个样本进行分词，以降低collate_fn的内存峰值。
        """
        row = self.df.iloc[idx]
        problem = row['problem']
        solution = row['solution_cot']

        conversation = [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution}
        ]

        # 准备用于计算prompt长度的文本
        prompt_only_text = self.tokenizer.apply_chat_template(
            [conversation[0]], tokenize=False, add_generation_prompt=True
        )
        # 准备完整的训练文本
        full_text = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        ) + self.tokenizer.eos_token

        # 对单个样本进行分词，不填充
        tokenized_prompt = self.tokenizer(prompt_only_text, truncation=True, max_length=self.max_length)
        tokenized_full = self.tokenizer(full_text, truncation=True, max_length=self.max_length)

        prompt_len = len(tokenized_prompt['input_ids'])

        input_ids = torch.LongTensor(tokenized_full['input_ids'])
        attention_mask = torch.LongTensor(tokenized_full['attention_mask'])

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def collate_fn_sft(batch: List[Dict], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """
    [STABLE & CORRECTED] 实现了动态padding，并修复了tokenizer.pad()的用法。
    """
    tokenizer.padding_side = 'right'

    # 从批次中收集已经处理好的张量列表
    input_ids_list = [item['input_ids'] for item in batch]
    attention_mask_list = [item['attention_mask'] for item in batch]
    labels_list = [item['labels'] for item in batch]

    # [CRITICAL FIX] 使用torch.nn.utils.rnn.pad_sequence来手动填充所有张量
    # 这是最灵活且不会出错的方式。
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask_list, batch_first=True, padding_value=0
    )
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels_list, batch_first=True, padding_value=-100
    )

    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask,
        'labels': padded_labels
    }