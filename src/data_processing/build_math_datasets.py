# src/data_processing/build_math_datasets.py

import os
import json
import pandas as pd
from datasets import load_dataset
from omegaconf import DictConfig


def build_datasets(config: DictConfig):
    """
    [FINAL-RFT-AWARE VERSION]
    - 主要为RL阶段创建 prompt_pool (需要 pass_rate 和 gold_answer)。
    - 同时，基于配置中的 rl_training_sources，为SFT冷启动创建一个数据集。
      (在RFT流程中，这个SFT数据集可作为备用)。
    - 动态地使用配置文件中为每个数据源指定的 question_key 和 answer_key。
    """
    data_cfg = config.data
    os.makedirs(data_cfg.processed_dir, exist_ok=True)

    print("--- Starting Data Preparation for SFT and RL (Robust Version) ---")

    all_sft_data = []
    all_rl_prompts = []

    # 仅使用在 config.data.rl_training_sources 中定义的源
    for ds_conf in data_cfg.rl_training_sources:
        print(f"Loading source dataset for RL/SFT prep: {ds_conf.name} from path: {ds_conf.path}...")
        dataset = load_dataset(ds_conf.path, split=ds_conf.split, trust_remote_code=True)

        # 动态获取字段名
        question_key = ds_conf.question_key
        answer_key = ds_conf.answer_key
        print(f"Using keys -> question: '{question_key}', answer: '{answer_key}'")

        df = dataset.to_pandas()

        # --- 为SFT冷启动数据筛选正样本 ---
        # 确保 'pass_rate' 列存在
        if 'pass_rate' in df.columns:
            df_positive = df[df['pass_rate'] > 0].copy()
            print(f"Found {len(df_positive)} positive samples for SFT from {ds_conf.name}.")
            
            for _, row in df_positive.iterrows():
                # [FIX] 使用配置中定义的键来提取数据
                all_sft_data.append({
                    'problem': str(row[question_key]),
                    'solution_cot': str(row[answer_key])
                })
        else:
            print(f"Warning: 'pass_rate' column not found in {ds_conf.name}. Skipping SFT data generation from this source.")


        # --- 为RL Prompt Pool准备所有问题 (需要特定元数据) ---
        if 'question' in df.columns and 'gold_answer' in df.columns and 'pass_rate' in df.columns:
            for _, row in df.iterrows():
                all_rl_prompts.append({
                    'problem': str(row['question']),
                    'final_answer': str(row['gold_answer']),
                    'pass_rate': float(row['pass_rate'])
                })
        else:
            print(f"Warning: Required columns for RL Prompt Pool not found in {ds_conf.name}. Skipping.")


    if not all_sft_data:
        print("WARNING: No positive samples found for SFT training across all sources.")

    if not all_rl_prompts:
        raise RuntimeError("No suitable data source found to create the RL prompt pool.")

    # --- 保存SFT数据 ---
    df_sft = pd.DataFrame(all_sft_data)
    df_sft_shuffled = df_sft.sample(frac=1, random_state=config.seed).reset_index(drop=True)

    test_size = min(500, int(0.05 * len(df_sft_shuffled)))
    if test_size == 0 and len(df_sft_shuffled) > 0: test_size = 1
    
    sft_test_df = df_sft_shuffled.iloc[:test_size]
    sft_train_df = df_sft_shuffled.iloc[test_size:]

    sft_train_path = os.path.join(data_cfg.processed_dir, data_cfg.sft_train_file)
    sft_test_path = os.path.join(data_cfg.processed_dir, data_cfg.sft_test_file)
    sft_train_df.to_parquet(sft_train_path, index=False)
    sft_test_df.to_parquet(sft_test_path, index=False)
    print(f"✅ Backup SFT train set saved to {sft_train_path} ({len(sft_train_df)} samples)")
    print(f"✅ Backup SFT test set saved to {sft_test_path} ({len(sft_test_df)} samples)")

    # --- 保存RL Prompt Pool ---
    rl_prompt_path = os.path.join(data_cfg.processed_dir, data_cfg.rl_prompt_file)
    with open(rl_prompt_path, 'w', encoding='utf-8') as f:
        # 去重
        unique_prompts = {item['problem']: item for item in all_rl_prompts}.values()
        for item in unique_prompts:
            f.write(json.dumps(item) + '\n')
    print(f"✅ RL prompt pool saved to {rl_prompt_path} ({len(unique_prompts)} unique prompts)")