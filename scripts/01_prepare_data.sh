#!/bin/bash

# scripts/01_prepare_data.sh
# 职责: 准备所有训练阶段所需的数据。

echo "--- Stage 0: Preparing SFT and RL data ---"
echo "This script will download internlm/OREAL-RL-Prompts and process it into:"
echo "1. sft_train.parquet & sft_test.parquet (from positive samples for SFT Cold Start)"
echo "2. rl_prompt_pool.jsonl (from all questions for RL Training)"
echo "-----------------------------------------------------"

# 使用我们新的主配置文件来运行数据处理任务
# Hydra会根据 'main.task' 自动调用正确的功能
python src/main.py --config-name=lcare_config_final main.task=process_data

echo "✅ Data preparation finished."