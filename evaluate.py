# evaluate.py

import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from functools import partial
from transformers import AutoTokenizer

from src.models.actor_critic import LCARE_Actor
from src.utils.verifier import Verifier
from src.utils.prompt_constructor import PromptConstructor
from src.datasets.evaluation_dataset import EvaluationDataset, collate_fn_eval


def run_evaluation(config: DictConfig):
    """
      主评估函数。
    - 可加载SFT LoRA权重或RL checkpoint。
    - 严格按照配置文件中的数据集列表进行评估。
    - 为技能增强生成详细的失败样本报告。
    """
    eval_config = config.evaluation
    print("--- Starting Final Evaluation (Skill-Enhancement Ready) ---")
    print(f"Evaluating model checkpoint from: {eval_config.model_path}")

    output_dir = os.path.join(eval_config.output_dir, config.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Evaluation results will be saved to: {output_dir}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.path, trust_remote_code=True, padding_side='left'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Initializing model architecture from '{config.model.path}'...")
    actor = LCARE_Actor(config.model, tokenizer).to(device)

    model_path = eval_config.model_path
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Provided model_path for evaluation '{model_path}' is not a valid directory.")

    try:
        # 检查目录中是否存在适配器权重文件
        if any(file.endswith(('.safetensors', '.bin', '.pt')) for file in os.listdir(model_path)):
            actor.model.load_adapter(model_path, adapter_name="default")
            print("Successfully loaded LoRA adapter for evaluation.")
        else:
            raise ValueError(f"No LoRA weights (.safetensors, .bin, .pt) found in '{model_path}'.")
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to load LoRA adapter from '{model_path}'. Ensure the directory contains adapter weights "
            f"and the base model in your config ('{config.model.path}') is correct. Error: {e}")

    actor.eval()

    verifier = Verifier(config)
    prompt_constructor = PromptConstructor(config, tokenizer)

    full_summary = {}
    for ds_conf in eval_config.datasets:
        name = ds_conf['name']
        print(f"\n--- Evaluating on dataset: {name} from path: {ds_conf['path']} ---")

        try:
            dataset = EvaluationDataset(ds_conf, prompt_constructor, tokenizer)
        except Exception as e:
            print(f"WARNING: Could not load or process dataset {name}. Skipping. Error: {e}")
            continue

        collate_with_tokenizer = partial(collate_fn_eval, tokenizer=tokenizer)
        dataloader = DataLoader(
            dataset, batch_size=eval_config.batch_size, shuffle=False, collate_fn=collate_with_tokenizer
        )

        correct_count = 0
        total_count = 0
        detailed_results = []

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {name}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                sampling_params = {"max_new_tokens": 2048, "do_sample": False}
                action_ids, _ = actor.generate(input_ids, attention_mask, sampling_params)

            full_solution_texts = tokenizer.batch_decode(action_ids, skip_special_tokens=True)
            prompts_text = batch['problems_text']

            for i in range(len(prompts_text)):
                prompt_text = prompts_text[i]
                prompt_for_model = prompt_constructor.get_evaluation_prompt(prompt_text)

                full_text = full_solution_texts[i]
                if full_text.startswith(prompt_for_model):
                    solution_only = full_text[len(prompt_for_model):].strip()
                else:
                    solution_only = full_text.strip()

                is_correct = verifier.verify(
                    solution_text=solution_only,
                    ground_truth=batch['ground_truths'][i],
                    question=prompt_text,
                    use_llm_judger=False
                )

                if is_correct: correct_count += 1
                total_count += 1

                problem_id = -1
                if isinstance(dataset.data, list) and batch_idx * eval_config.batch_size + i < len(dataset.data):
                    if isinstance(dataset.data[0], dict):
                        problem_id = dataset.data[batch_idx * eval_config.batch_size + i].get('id', -1)

                detailed_results.append({
                    "problem_idx_in_dataset": problem_id,
                    "problem": prompt_text,
                    "generated_solution": solution_only,
                    "ground_truth_answer": batch['ground_truths'][i],
                    "is_correct": is_correct
                })

        pass_at_1 = (correct_count / total_count) if total_count > 0 else 0
        full_summary[name] = {"pass@1": pass_at_1, "correct": correct_count, "total": total_count}
        print(f"Result for {name}: pass@1 = {pass_at_1:.4f} ({correct_count}/{total_count})")

        detailed_results_path = os.path.join(output_dir, f"detailed_results_{name}.jsonl")
        with open(detailed_results_path, 'w', encoding='utf-8') as f:
            for entry in detailed_results:
                f.write(json.dumps(entry) + '\n')
        print(f"Detailed results for {name} saved to {detailed_results_path}")

    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(full_summary, f, indent=4)
    print(f"\n✅ Evaluation finished. Summary saved to {summary_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="lcare_config")
def hydra_main(config: DictConfig) -> None:
    run_evaluation(config)


if __name__ == '__main__':
    hydra_main()