# src/data_processing/build_math_datasets.py

import os
import json
import pandas as pd
from datasets import load_dataset

from omegaconf import DictConfig


def build_datasets(config: DictConfig):
    """
    [V-FINAL] 只处理在 config.data.rl_training_sources 中定义的RL训练数据源。
    """
    data_cfg = config.data
    os.makedirs(data_cfg.processed_dir, exist_ok=True)

    print("--- Starting Data Preparation for SFT and RL ---")

    all_processed_data = []
    for ds_conf in data_cfg.rl_training_sources:
        print(f"Loading source dataset: {ds_conf.name} from path: {ds_conf.path}...")
        dataset = load_dataset(ds_conf.path, split=ds_conf.split, trust_remote_code=True)
        all_processed_data.append(dataset.to_pandas())

    if not all_processed_data:
        raise RuntimeError("No training data sources were processed.")

    df_combined = pd.concat(all_processed_data, ignore_index=True)
    print(f"Loaded a total of {len(df_combined)} samples for SFT and RL.")

    # --- 创建SFT冷启动数据 (仅使用成功轨迹) ---
    df_positive = df_combined[df_combined['pass_rate'] > 0].copy()
    print(f"Found {len(df_positive)} positive samples for SFT.")

    sft_data = [{'problem': str(row['question']), 'solution_cot': str(row['answer'])} for _, row in
                df_positive.iterrows()]
    df_sft = pd.DataFrame(sft_data)

    df_sft_shuffled = df_sft.sample(frac=1, random_state=config.seed).reset_index(drop=True)

    test_size = min(500, int(0.05 * len(df_sft_shuffled)))
    if test_size == 0 and len(df_sft_shuffled) > 0: test_size = 1

    sft_test_df = df_sft_shuffled.iloc[:test_size]
    sft_train_df = df_sft_shuffled.iloc[test_size:]

    sft_train_path = os.path.join(data_cfg.processed_dir, data_cfg.sft_train_file)
    sft_test_path = os.path.join(data_cfg.processed_dir, data_cfg.sft_test_file)
    sft_train_df.to_parquet(sft_train_path, index=False)
    sft_test_df.to_parquet(sft_test_path, index=False)
    print(f"✅ Cold-Start SFT train set saved to {sft_train_path} ({len(sft_train_df)} samples)")
    print(f"✅ Cold-Start SFT test set saved to {sft_test_path} ({len(sft_test_df)} samples)")

    # --- 创建RL Prompt Pool (使用所有问题及元数据) ---
    rl_prompt_pool = [
        {'problem': str(row['question']), 'final_answer': str(row['gold_answer']), 'pass_rate': float(row['pass_rate'])}
        for _, row in df_combined.iterrows()]
    rl_prompt_path = os.path.join(data_cfg.processed_dir, data_cfg.rl_prompt_file)
    with open(rl_prompt_path, 'w', encoding='utf-8') as f:
        unique_prompts = {item['problem']: item for item in rl_prompt_pool}.values()
        for item in unique_prompts:
            f.write(json.dumps(item) + '\n')
    print(f"✅ RL prompt pool saved to {rl_prompt_path} ({len(unique_prompts)} unique prompts)")
