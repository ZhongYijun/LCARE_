#!/bin/bash
#### ✅ `scripts/05_create_enhancement_data.sh` (已修复)

set -e
echo "--- Optional Stage: Creating Skill Enhancement Data ---"

# 此脚本需要一个已完成的评估实验名作为输入
EVAL_EXPERIMENT_NAME=${1}

if [ -z "$EVAL_EXPERIMENT_NAME" ]; then
    echo "Usage: $0 <evaluation_experiment_name>"
    echo "Example: $0 eval_iter_80"
    exit 1
fi

echo "Analyzing failures from experiment: $EVAL_EXPERIMENT_NAME"
echo "--------------------------------------------------------"

python src/main.py \
    --config-name=lcare_config_final \
    main.task=create_enhancement_data \
    experiment_name=$EVAL_EXPERIMENT_NAME # 传递正确的实验名

echo "✅ Skill enhancement data created in data/processed/"