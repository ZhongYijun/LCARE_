# src/data_processing/generate_rft_data.py

# [1] PYTHON PATH CORRECTION
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import os
import hydra
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# [2] PROJECT-SPECIFIC IMPORTS
from src.utils.verifier import Verifier
from src.utils.prompt_constructor import PromptConstructor

def generate_rft_data(config: DictConfig):
    """
    [ULTIMATE ROBUST VERSION]
    - 使用DataParallel和DataLoader加速。
    - 在合并数据集前，通过移除多余列来强制统一数据结构，彻底解决collate_fn错误。
    """
    print("--- Starting RFT Data Generation (High-Performance & Robust) ---")
    
    # ... [3] INITIALIZATION & PARALLELIZATION (保持不变) ...
    rft_config = config.trainer.rft_generation
    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda:0" if num_gpus > 0 else "cpu")
    print(f"Detected {num_gpus} GPUs.")

    tokenizer = AutoTokenizer.from_pretrained(rft_config.model_path, trust_remote_code=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        rft_config.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    if num_gpus > 1:
        print(f"Using torch.nn.DataParallel for {num_gpus} GPUs.")
        model = torch.nn.DataParallel(model)
    model.to(device)
    model.eval()
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    verifier = Verifier(config)
    prompt_constructor = PromptConstructor(config, tokenizer)

    # --------------------------------------------------------------------------- #
    # [4] DATA LOADING & SCHEMA UNIFICATION
    # --------------------------------------------------------------------------- #
    all_problem_sets = []
    for source_conf in config.data.rft_generation_sources:
        print(f"Loading and cleaning RFT source: {source_conf.path}")
        ds = load_dataset(source_conf.path, split=source_conf.split)
        
        # 重命名列
        if source_conf.question_key != "problem": ds = ds.rename_column(source_conf.question_key, "problem")
        if source_conf.answer_key != "ground_truth": ds = ds.rename_column(source_conf.answer_key, "ground_truth")

        # [CRITICAL FIX] 强制统一schema：只保留我们需要的列，丢弃其他所有列。
        columns_to_keep = ["problem", "ground_truth"]
        columns_to_remove = [col for col in ds.column_names if col not in columns_to_keep]
        if columns_to_remove:
            print(f"  Removing extra columns: {columns_to_remove}")
            ds = ds.remove_columns(columns_to_remove)
            
        all_problem_sets.append(ds)
    
    combined_dataset = concatenate_datasets(all_problem_sets).shuffle(seed=config.seed)
    combined_dataset = combined_dataset.filter(lambda x: x['ground_truth'] is not None and len(str(x['ground_truth'])) > 0)
    
    if len(combined_dataset) > rft_config.max_problems_to_process:
        print(f"Subsampling dataset from {len(combined_dataset)} to {rft_config.max_problems_to_process} problems.")
        combined_dataset = combined_dataset.select(range(rft_config.max_problems_to_process))

    batch_size = rft_config.batch_size_per_gpu * max(1, num_gpus)
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False)
    print(f"Starting generation with Total Batch Size = {batch_size} ({rft_config.batch_size_per_gpu} per GPU)")

    # ... [5] & [6] CORE LOOP AND SAVING (保持不变) ...
    accepted_samples = []
    pbar = tqdm(dataloader, desc="Generating & Verifying RFT samples in Batches")
    
    for batch in pbar:
        problems_text = batch['problem']
        ground_truths_text = batch['ground_truth']
        prompts = [prompt_constructor.get_evaluation_prompt(p) for p in problems_text]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        
        generation_config = GenerationConfig(
            max_new_tokens=2048, do_sample=True, temperature=0.7, top_p=0.9,
            num_return_sequences=rft_config.num_samples_per_problem,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id
        )

        with torch.no_grad():
            generated_sequences = model.module.generate(**inputs, generation_config=generation_config) if num_gpus > 1 else model.generate(**inputs, generation_config=generation_config)

        decoded_texts = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
        
        for i in range(len(problems_text)):
            problem_text = problems_text[i]
            ground_truth = ground_truths_text[i]
            original_prompt = prompts[i]
            start_idx = i * rft_config.num_samples_per_problem
            end_idx = start_idx + rft_config.num_samples_per_problem
            candidates = decoded_texts[start_idx:end_idx]
            
            for text in candidates:
                solution_only = text[len(original_prompt):].strip()
                if verifier.verify(solution_text=solution_only, ground_truth=ground_truth, question=problem_text):
                    accepted_samples.append({"problem": problem_text, "solution_cot": solution_only})
                    break 

        pbar.set_postfix({"Accepted": len(accepted_samples)})
        if len(accepted_samples) >= rft_config.target_accepted_samples:
            print(f"\nTarget of {rft_config.target_accepted_samples} accepted samples reached. Stopping generation.")
            break

    if not accepted_samples: print("WARNING: No correct samples were generated.")
    df = pd.DataFrame(accepted_samples)
    output_filename = config.data.rft_sft_file
    output_path = os.path.join(config.data.processed_dir, output_filename)
    os.makedirs(config.data.processed_dir, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\n✅ RFT Data Generation Finished. {len(df)} accepted samples saved to {output_path}")

@hydra.main(version_base=None, config_path="../../configs", config_name="lcare_config")
def main(config: DictConfig) -> None:
    generate_rft_data(config)

if __name__ == "__main__":
    main()