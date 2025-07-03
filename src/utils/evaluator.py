# src/utils/evaluator.py
# [NEW MODULE] A dedicated, reusable evaluator for consistency.

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
import logging
from typing import Dict, Any

from transformers import AutoTokenizer, GenerationConfig

# Import necessary components from the project
from src.models.actor_critic import LCARE_Actor
from src.datasets.evaluation_dataset import EvaluationDataset, collate_fn_eval
from src.utils.verifier import Verifier
from src.utils.prompt_constructor import PromptConstructor

logger = logging.getLogger(__name__)


class DynamicEvaluator:
    """
    A self-contained evaluator class to run evaluations consistently
    across different parts of the project (e.g., main eval script and RL agent).
    """

    def __init__(self, model_actor: LCARE_Actor, tokenizer: AutoTokenizer, verifier: Verifier,
                 prompt_constructor: PromptConstructor, device: torch.device):
        self.actor = model_actor
        self.tokenizer = tokenizer
        self.verifier = verifier
        self.prompt_constructor = prompt_constructor
        self.device = device

    def evaluate(self, dataset_config: Dict[str, Any], eval_batch_size: int) -> Dict[str, Any]:
        """
        Runs evaluation on a single dataset configuration.
        Returns a dictionary with results or an empty dict on failure.
        """
        name = dataset_config.get('name', 'Unknown Dataset')
        self.actor.eval()  # Ensure model is in evaluation mode

        try:
            dataset = EvaluationDataset(dataset_config, self.prompt_constructor, self.tokenizer)
            if not dataset:
                logger.warning(f"Dataset '{name}' is empty or failed to load. Skipping.")
                return {}
        except Exception as e:
            logger.error(f"Could not load or process dataset {name}. Skipping. Error: {e}", exc_info=True)
            return {}

        # [CRITICAL FIX] Correctly wrap collate_fn with functools.partial
        collate_fn_with_tokenizer = partial(collate_fn_eval, tokenizer=self.tokenizer)

        dataloader = DataLoader(
            dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn_with_tokenizer,
            num_workers=0  # Use 0 for stability in nested calls
        )

        correct_count = 0
        total_count = 0

        # [BUGFIX] Use GenerationConfig to avoid warnings
        generation_config = GenerationConfig(
            max_new_tokens=2048,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )

        for batch in tqdm(dataloader, desc=f"Evaluating {name}", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with torch.no_grad():
                # Pass the generation_config object
                action_ids, _ = self.actor.generate(input_ids, attention_mask, generation_config)

            full_solution_texts = self.tokenizer.batch_decode(action_ids, skip_special_tokens=True)

            for i in range(len(batch['problems_text'])):
                problem_text = batch['problems_text'][i]
                prompt_for_model = self.prompt_constructor.get_evaluation_prompt(problem_text)

                full_text = full_solution_texts[i]
                solution_only = full_text[len(prompt_for_model):].strip() if full_text.startswith(
                    prompt_for_model) else full_text.strip()

                is_correct = self.verifier.verify(
                    solution_text=solution_only,
                    ground_truth=batch['ground_truths'][i],
                    question=problem_text
                )
                if is_correct:
                    correct_count += 1
                total_count += 1

        self.actor.train()  # Revert model to training mode after evaluation

        if total_count > 0:
            pass_at_1 = correct_count / total_count
            return {"pass@1": pass_at_1, "correct": correct_count, "total": total_count}

        return {}