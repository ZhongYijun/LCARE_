# src/datasets/rl_prompt_dataset.py
import json
from torch.utils.data import Dataset
from typing import List, Dict

class RLPromptDataset(Dataset):
    """为RL Agent提供prompt的Dataset类"""
    def __init__(self, jsonl_path: str):
        self.prompts: List[Dict] = []
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.prompts.append(json.loads(line))
        except FileNotFoundError:
            raise FileNotFoundError(f"RL prompt pool file not found at {jsonl_path}. Please run data processing first.")

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        """返回一个字典，包含问题文本和标准答案"""
        item = self.prompts[idx]
        return {
            'problem_text': item['problem'],
            'ground_truth_answer': item['final_answer']
        }