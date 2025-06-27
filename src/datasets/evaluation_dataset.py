# src/datasets/evaluation_dataset.py

import os
from torch.utils.data import Dataset
from datasets import load_dataset, Dataset as HuggingFaceDataset
from transformers import AutoTokenizer
from typing import List, Dict, Any

from src.utils.prompt_constructor import PromptConstructor


class EvaluationDataset(Dataset):
    """
    一个用于评估阶段的PyTorch Dataset。
    它负责加载指定的数据集，并将其转换为模型需要的prompt格式。
    """

    def __init__(self, dataset_config: Dict, prompt_constructor: PromptConstructor, tokenizer: AutoTokenizer):
        self.config = dataset_config
        self.prompt_constructor = prompt_constructor
        self.tokenizer = tokenizer

        self.data = self._load_data()

    def _load_data(self) -> List[Dict]:
        """加载并预处理数据"""
        name = self.config['name']
        path = self.config['path']
        print(f"Loading evaluation dataset: {name} from {path}")

        dataset: HuggingFaceDataset
        try:
            # 优先尝试从HuggingFace Hub加载
            dataset = load_dataset(path, split=self.config.get('split', 'test'), trust_remote_code=True)
        # [修复] 捕获更具体的异常
        except FileNotFoundError:
            # 如果Hub上找不到，则尝试作为本地文件加载
            if not os.path.exists(path):
                raise FileNotFoundError(f"Dataset file for '{name}' not found at local path: {path}")

            try:
                if path.endswith((".jsonl", ".json")):
                    dataset = load_dataset('json', data_files=path, split='train')
                elif path.endswith(".parquet"):
                    dataset = load_dataset('parquet', data_files=path, split='train')
                else:
                    raise ValueError(
                        f"Unsupported local file format for dataset '{name}': {path}. Please use .json, .jsonl, or .parquet.")
            except Exception as e:
                raise IOError(f"Failed to load local dataset file '{path}'. Error: {e}")
        except Exception as e:
            # 捕获所有其他来自HuggingFace的加载错误
            raise IOError(f"Failed to load dataset '{name}' from HuggingFace Hub or local path '{path}'. Error: {e}")

        processed_data = []
        for item in dataset:
            # 适配不同数据集的字段名
            problem = item.get('problem', item.get('Question', item.get('question', '')))
            # 答案可能是列表或字符串
            answer_raw = item.get('answer', item.get('Correct Answer', item.get('final_answer', '')))
            ground_truth = str(answer_raw[0] if isinstance(answer_raw, list) else answer_raw)

            if not problem or not ground_truth:
                print(f"Warning: Skipping item in '{name}' with missing 'problem' or 'ground_truth': {item}")
                continue

            processed_data.append({
                'problem_text': str(problem),
                'ground_truth': ground_truth,
            })

        print(f"Successfully loaded {len(processed_data)} samples for dataset '{name}'.")
        return processed_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]

        prompt_text = self.prompt_constructor.get_evaluation_prompt(item['problem_text'])

        tokenized_prompt = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=2048
        )

        return {
            'input_ids': tokenized_prompt['input_ids'].squeeze(0),
            'attention_mask': tokenized_prompt['attention_mask'].squeeze(0),
            'problem_text': item['problem_text'],
            'ground_truth': item['ground_truth']
        }


def collate_fn_eval(batch: List[Dict], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """
    为评估Dataloader动态进行左padding。
    """
    input_ids_list = [item['input_ids'] for item in batch]
    attention_masks_list = [item['attention_mask'] for item in batch]

    # 左padding对于自回归生成（解码）是标准做法
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    padded = tokenizer.pad(
        {"input_ids": input_ids_list, "attention_mask": attention_masks_list},
        return_tensors="pt",
        padding='longest',
    )

    return {
        'input_ids': padded['input_ids'],
        'attention_mask': padded['attention_mask'],
        'problems_text': [item['problem_text'] for item in batch],
        'ground_truths': [item['ground_truth'] for item in batch]
    }