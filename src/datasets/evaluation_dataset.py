# # src/datasets/evaluation_dataset.py (ULTIMATE ROBUST VERSION)
#
#
# import re
# from torch.utils.data import Dataset
# from datasets import load_dataset, concatenate_datasets
# from transformers import AutoTokenizer
# from typing import List, Dict, Any
#
# from src.utils.prompt_constructor import PromptConstructor
#
#
# class EvaluationDataset(Dataset):
#     """
#     [ULTIMATE ROBUST VERSION]
#     - 能够根据配置文件，处理需要指定`split`和单个`subset`或多个`subsets`列表的数据集。
#     - 使用从配置文件中传入的字段名进行解析。
#     """
#
#     def __init__(self, dataset_config: Dict, prompt_constructor: PromptConstructor, tokenizer: AutoTokenizer):
#         self.config = dataset_config
#         self.prompt_constructor = prompt_constructor
#         self.tokenizer = tokenizer
#
#         self.question_key = self.config['question_key']
#         self.answer_key = self.config['answer_key']
#
#         self.data = self._load_data()
#
#     def _load_data(self) -> List[Dict]:
#         """
#         [NEW] 加载并预处理数据。现在可以处理单个subset或一个subsets列表。
#         """
#         name = self.config['name']
#         path = self.config['path']
#         split = self.config.get('split', 'test')
#
#         print("-" * 50)
#         print(f"Loading evaluation dataset: {name}")
#         print(f"  - Path: {path}")
#         print(f"  - Split: {split}")
#
#         all_datasets = []
#
#         # 【核心修复】区分处理单个subset和多个subsets
#         if 'subsets' in self.config and self.config['subsets'] is not None:
#             # --- 处理子集列表 (例如 OlympiadBench) ---
#             print(f"  - Loading and concatenating multiple subsets...")
#             for subset_name in self.config['subsets']:
#                 try:
#                     print(f"    - Loading subset: {subset_name}")
#                     ds_part = load_dataset(path, name=subset_name, split=split, trust_remote_code=True)
#                     all_datasets.append(ds_part)
#                 except Exception as e:
#                     print(f"    - WARNING: Could not load subset '{subset_name}'. Skipping. Error: {e}")
#             if not all_datasets:
#                 raise IOError(f"Failed to load any subset for dataset '{name}'.")
#             # 合并所有加载的子集
#             dataset = concatenate_datasets(all_datasets)
#             print(f"  - Successfully concatenated {len(all_datasets)} subsets.")
#
#         else:
#             # --- 处理单个子集或无子集 ---
#             subset_name = self.config.get('subset')
#             if subset_name:
#                 print(f"  - Subset: {subset_name}")
#             try:
#                 dataset = load_dataset(path, name=subset_name, split=split, trust_remote_code=True)
#             except Exception as e:
#                 raise IOError(
#                     f"Failed to load dataset '{name}' from HuggingFace Hub (Path: {path}, Subset: {subset_name}, Split: {split}). Error: {e}")
#
#         # --- 后续处理流程保持不变 ---
#         processed_data = []
#         print(f"  - Processing loaded data with keys: question='{self.question_key}', answer='{self.answer_key}'")
#         for item in dataset:
#             problem = item.get(self.question_key)
#             answer_raw = item.get(self.answer_key)
#
#             if problem is None or answer_raw is None:
#                 continue
#
#             ground_truth = str(answer_raw[0] if isinstance(answer_raw, list) else answer_raw)
#             if self.answer_key == 'solution' and '\\boxed' in ground_truth:
#                 boxed_match = re.search(r"\\boxed\{(.*?)\}", ground_truth)
#                 if boxed_match:
#                     ground_truth = boxed_match.group(1).strip()
#
#             processed_data.append({
#                 'problem_text': str(problem),
#                 'ground_truth': ground_truth,
#             })
#
#         print(f"Successfully processed {len(processed_data)} samples for dataset '{name}'.")
#         return processed_data
#
#     def __len__(self) -> int:
#         return len(self.data)
#
#     def __getitem__(self, idx: int) -> Dict:
#         item = self.data[idx]
#         prompt_text = self.prompt_constructor.get_evaluation_prompt(item['problem_text'])
#
#         tokenized_prompt = self.tokenizer(prompt_text, return_tensors="pt", padding=False, truncation=True,
#                                           max_length=4096)
#
#         return {
#             'input_ids': tokenized_prompt['input_ids'].squeeze(0),
#             'attention_mask': tokenized_prompt['attention_mask'].squeeze(0),
#             'problem_text': item['problem_text'],
#             'ground_truth': item['ground_truth']
#         }
#
#
# def collate_fn_eval(batch: List[Dict], tokenizer: AutoTokenizer) -> Dict[str, Any]:
#     input_ids_list = [item['input_ids'] for item in batch]
#
#     tokenizer.padding_side = 'left'
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#
#     padded = tokenizer.pad({"input_ids": input_ids_list}, return_tensors="pt", padding='longest')
#
#     return {
#         'input_ids': padded['input_ids'],
#         'attention_mask': padded['attention_mask'],
#         'problems_text': [item['problem_text'] for item in batch],
#         'ground_truths': [item['ground_truth'] for item in batch]
#     }

# # src/datasets/evaluation_dataset.py (ULTIMATE ROBUST & EFFICIENT VERSION)
#
# import re
# from torch.utils.data import Dataset
# from datasets import load_dataset, concatenate_datasets, Dataset as HuggingFaceDataset
# from transformers import AutoTokenizer
# from typing import List, Dict, Any
#
# from src.utils.prompt_constructor import PromptConstructor
#
#
# class EvaluationDataset(Dataset):
#     def __init__(self, dataset_config: Dict, prompt_constructor: PromptConstructor, tokenizer: AutoTokenizer):
#         self.config = dataset_config
#         self.tokenizer = tokenizer  # Tokenizer is still needed for answer cleaning
#         # 【核心修复】不再需要 prompt_constructor，因为它将在collate_fn中使用
#         self.question_key = self.config['question_key']
#         self.answer_key = self.config['answer_key']
#         self.data = self._load_data()
#
#     def _load_data(self) -> List[Dict]:
#         name = self.config['name']
#         path = self.config['path']
#         split = self.config.get('split', 'test')
#
#         all_datasets = []
#         if 'subsets' in self.config and self.config['subsets'] is not None:
#             for subset_name in self.config['subsets']:
#                 try:
#                     ds_part = load_dataset(path, name=subset_name, split=split, trust_remote_code=True)
#                     all_datasets.append(ds_part)
#                 except Exception as e:
#                     print(f"WARNING: Could not load subset '{subset_name}'. Skipping. Error: {e}")
#             if not all_datasets: raise IOError(f"Failed to load any subset for dataset '{name}'.")
#             dataset = concatenate_datasets(all_datasets)
#         else:
#             subset_name = self.config.get('subset')
#             try:
#                 dataset = load_dataset(path, name=subset_name, split=split, trust_remote_code=True)
#             except Exception as e:
#                 raise IOError(f"Failed to load dataset '{name}'. Error: {e}")
#
#         processed_data = []
#         for item in dataset:
#             problem = item.get(self.question_key)
#             answer_raw = item.get(self.answer_key)
#             if problem is None or answer_raw is None: continue
#
#             ground_truth = str(answer_raw[0] if isinstance(answer_raw, list) else answer_raw)
#             if self.answer_key == 'solution' and '\\boxed' in ground_truth:
#                 boxed_match = re.search(r"\\boxed\{(.*?)\}", ground_truth)
#                 if boxed_match: ground_truth = boxed_match.group(1).strip()
#
#             processed_data.append({
#                 'id': item.get('id', -1),
#                 'problem_text': str(problem),
#                 'ground_truth': ground_truth,
#             })
#         return processed_data
#
#     def __len__(self) -> int:
#         return len(self.data)
#
#     def __getitem__(self, idx: int) -> Dict:
#         # 【核心修复】__getitem__现在只返回纯文本，不做任何tokenize
#         return self.data[idx]
#
#
# def collate_fn_eval(batch: List[Dict], tokenizer: AutoTokenizer, prompt_constructor: PromptConstructor) -> Dict[
#     str, Any]:
#     """
#     [NEW & EFFICIENT] 在collate_fn中进行批处理tokenize，以获得最佳性能。
#     """
#     problems_text = [item['problem_text'] for item in batch]
#     ground_truths = [item['ground_truth'] for item in batch]
#     ids = [item['id'] for item in batch]
#
#     # 1. 批量构建所有prompt
#     prompts_for_model = [prompt_constructor.get_evaluation_prompt(p) for p in problems_text]
#
#     # 2. 对整个批次的prompts进行一次tokenize和padding操作
#     tokenizer.padding_side = 'left'
#     if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
#
#     inputs = tokenizer(
#         prompts_for_model,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=4096,
#     )
#
#     return {
#         'input_ids': inputs['input_ids'],
#         'attention_mask': inputs['attention_mask'],
#         'problems_text': problems_text,
#         'ground_truths': ground_truths,
#         'ids': ids,
#         'prompts_for_model': prompts_for_model  # 将构建好的prompt也传出去，方便后续处理
#     }

# # src/datasets/evaluation_dataset.py (V3 - FINAL OPTIMIZED)
#
# import os
# from torch.utils.data import Dataset
# from datasets import load_dataset, concatenate_datasets
# from transformers import AutoTokenizer
# from typing import List, Dict, Any
# from src.utils.prompt_constructor import PromptConstructor
# import logging
#
# logger = logging.getLogger(__name__)
#
#
# class EvaluationDataset(Dataset):
#     """
#     [V3 - FINAL] 评估专用数据集。
#     - 能够处理需要特定配置名的数据集 (例如HuggingFace上的多子集数据集)。
#     - 仅加载和准备原始文本数据，将分词和填充的重任完全交给DataLoader的collate_fn，以实现最高效率。
#     """
#
#     def __init__(self, dataset_config: Dict, prompt_constructor: PromptConstructor):
#         self.config = dataset_config
#         self.prompt_constructor = prompt_constructor
#
#         self.question_key = self.config.get('question_key', 'problem')
#         self.answer_key = self.config.get('answer_key', 'solution')
#
#         self.data = self._load_data()
#
#     def _load_data(self) -> List[Dict]:
#         name = self.config['name']
#         path = self.config['path']
#         split = self.config.get('split', 'test')
#         subsets = self.config.get('subsets')
#
#         logger.info(f"Attempting to load dataset: {name} from path: {path}")
#         logger.info(f"Using keys -> question: '{self.question_key}', answer: '{self.answer_key}'")
#
#         final_dataset = None
#
#         try:
#             if subsets:
#                 subset_list = subsets if isinstance(subsets, list) else [subsets]
#                 logger.info(f"Loading specified subsets: {subset_list}")
#
#                 all_sub_datasets = []
#                 for subset_name in subset_list:
#                     try:
#                         ds = load_dataset(path, name=subset_name, split=split, trust_remote_code=True)
#                         all_sub_datasets.append(ds)
#                     except Exception as e:
#                         logger.warning(
#                             f"Could not load subset '{subset_name}' for dataset '{name}'. Skipping. Error: {e}")
#
#                 if all_sub_datasets:
#                     final_dataset = concatenate_datasets(all_sub_datasets)
#             else:
#                 final_dataset = load_dataset(path, split=split, trust_remote_code=True)
#
#         except Exception as e:
#             logger.error(f"Failed to load dataset '{name}' from HuggingFace Hub or local path '{path}'. Error: {e}",
#                          exc_info=True)
#             return []
#
#         if final_dataset is None:
#             logger.warning(f"No data loaded for dataset '{name}'. It will be skipped.")
#             return []
#
#         processed_data = []
#         for item in final_dataset:
#             problem = item.get(self.question_key)
#             answer_raw = item.get(self.answer_key)
#
#             if problem is None or answer_raw is None:
#                 logger.warning(
#                     f"Skipping item in '{name}' due to missing key '{self.question_key}' or '{self.answer_key}': {item}")
#                 continue
#
#             ground_truth = str(answer_raw[0] if isinstance(answer_raw, list) else answer_raw)
#             item_id = item.get('id', item.get('problem_id', -1))
#
#             processed_data.append({
#                 'id': item_id,
#                 'problem_text': str(problem),
#                 'ground_truth': ground_truth,
#             })
#
#         logger.info(f"Successfully loaded {len(processed_data)} samples for dataset '{name}'.")
#         return processed_data
#
#     def __len__(self) -> int:
#         return len(self.data)
#
#     def __getitem__(self, idx: int) -> Dict:
#         """
#         [OPTIMIZED] 仅返回原始文本和元数据。
#         分词被推迟到DataLoader的collate_fn中，以实现批量处理的最高效率。
#         """
#         item = self.data[idx]
#
#         # 构造将要送入模型的完整prompt文本
#         prompt_for_model = self.prompt_constructor.get_evaluation_prompt(item['problem_text'])
#
#         return {
#             'prompt_for_model': prompt_for_model,
#             'id': item['id'],
#             'problem_text': item['problem_text'],  # 保留原始问题用于日志记录
#             'ground_truth': item['ground_truth']
#         }

# src/datasets/evaluation_dataset.py
# [L-CARE V3 - FINAL, SUBSET-AWARE]

from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from typing import List, Dict, Any
import logging

from src.utils.prompt_constructor import PromptConstructor

# 使用模块级的logger
logger = logging.getLogger(__name__)


class EvaluationDataset(Dataset):
    """
    [V3 - FINAL] An enhanced PyTorch Dataset for evaluation.
    - Handles datasets requiring a specific configuration name via the 'subsets' key.
    - Supports loading and merging multiple subsets from a single Hugging Face path.
    - Dynamically uses field names from the configuration for maximum flexibility.
    """

    def __init__(self, dataset_config: Dict, prompt_constructor: PromptConstructor, tokenizer: AutoTokenizer):
        self.config = dataset_config
        self.prompt_constructor = prompt_constructor
        self.tokenizer = tokenizer

        # Safely get keys, defaulting to common names
        self.question_key = self.config.get('question_key', 'problem')
        self.answer_key = self.config.get('answer_key', 'solution')

        self.data = self._load_data()

    def _load_data(self) -> List[Dict]:
        """Loads and preprocesses data, handling various dataset structures."""
        name = self.config['name']
        path = self.config['path']
        split = self.config.get('split', 'test')
        subsets = self.config.get('subsets')

        logger.info(f"Attempting to load dataset: '{name}' from path: '{path}'")
        logger.info(f"Using keys -> question: '{self.question_key}', answer: '{self.answer_key}'")

        final_dataset = None
        try:
            # [CRITICAL FIX] Correctly handle single subset string or list of subset strings
            if subsets:
                subset_list = subsets if isinstance(subsets, list) else [subsets]
                logger.info(f"Attempting to load specified subsets: {subset_list}")

                all_sub_datasets = []
                for subset_name in subset_list:
                    try:
                        # The subset/config name is passed as the 'name' parameter to load_dataset
                        ds = load_dataset(path, name=subset_name, split=split, trust_remote_code=True)
                        all_sub_datasets.append(ds)
                        logger.info(f"  Successfully loaded subset '{subset_name}'.")
                    except Exception as e:
                        logger.warning(
                            f"  Could not load subset '{subset_name}' for dataset '{name}'. Skipping. Error: {e}")

                if all_sub_datasets:
                    final_dataset = concatenate_datasets(all_sub_datasets)
            else:
                # Load as a simple dataset if no subsets are specified
                logger.info("No subsets specified, loading directly.")
                final_dataset = load_dataset(path, split=split, trust_remote_code=True)

        except Exception as e:
            logger.error(f"Fatal error loading dataset '{name}' from HuggingFace Hub '{path}'. Error: {e}",
                         exc_info=True)
            return []

        if final_dataset is None:
            logger.warning(f"No data could be loaded for dataset '{name}'. It will be skipped.")
            return []

        processed_data = []
        for i, item in enumerate(final_dataset):
            problem = item.get(self.question_key)
            answer_raw = item.get(self.answer_key)

            if problem is None or answer_raw is None:
                logger.warning(
                    f"Skipping item in '{name}' due to missing key '{self.question_key}' or '{self.answer_key}': {item}")
                continue

            ground_truth = str(answer_raw[0] if isinstance(answer_raw, list) else answer_raw)
            item_id = item.get('id', item.get('problem_id', i))  # Use index as fallback ID

            processed_data.append({
                'id': item_id,
                'problem_text': str(problem),
                'ground_truth': ground_truth,
            })

        logger.info(f"Successfully processed {len(processed_data)} samples for dataset '{name}'.")
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
            max_length=4096  # A safe max length for prompts
        )

        return {
            'input_ids': tokenized_prompt['input_ids'].squeeze(0),
            'attention_mask': tokenized_prompt['attention_mask'].squeeze(0),
            'id': item['id'],
            'problem_text': item['problem_text'],
            'ground_truth': item['ground_truth']
        }


def collate_fn_eval(batch: List[Dict], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    input_ids_list = [item['input_ids'] for item in batch]
    attention_mask_list = [item['attention_mask'] for item in batch]

    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    padded = tokenizer.pad(
        {"input_ids": input_ids_list, "attention_mask": attention_mask_list},
        return_tensors="pt",
        padding='longest',
    )

    return {
        'input_ids': padded['input_ids'],
        'attention_mask': padded['attention_mask'],
        'ids': [item['id'] for item in batch],
        'problems_text': [item['problem_text'] for item in batch],
        'ground_truths': [item['ground_truth'] for item in batch]
    }