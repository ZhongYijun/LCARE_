# # src/datasets/rl_prompt_dataset.py
# import json
# from torch.utils.data import Dataset
# from typing import List, Dict

# class RLPromptDataset(Dataset):
#     """为RL Agent提供prompt的Dataset类"""
#     def __init__(self, jsonl_path: str):
#         self.prompts: List[Dict] = []
#         try:
#             with open(jsonl_path, 'r', encoding='utf-8') as f:
#                 for line in f:
#                     self.prompts.append(json.loads(line))
#         except FileNotFoundError:
#             raise FileNotFoundError(f"RL prompt pool file not found at {jsonl_path}. Please run data processing first.")

#     def __len__(self) -> int:
#         return len(self.prompts)

#     def __getitem__(self, idx: int) -> Dict[str, str]:
#         """返回一个字典，包含问题文本和标准答案"""
#         item = self.prompts[idx]
#         return {
#             'problem_text': item['problem'],
#             'ground_truth_answer': item['final_answer']
#         }

# src/datasets/rl_prompt_dataset.py (修复后)

import json
from torch.utils.data import Dataset
from typing import List, Dict, Any


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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        [CORRECTED] 返回一个包含所有RL环境所需信息的字典。
        - 使用一致的键名。
        - 传递 pass_rate。
        """
        item = self.prompts[idx]
        return {
            'problem_text': item['problem'],
            'ground_truth_answer': item['final_answer'],
            'pass_rate': item.get('pass_rate', 0.5) # 确保传递pass_rate，并提供默认值
        }