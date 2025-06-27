#!/bin/bash

# scripts/06_run_enhancement_sft.sh
# 职责: (课程学习循环) 使用增强数据，对一个已有的RL模型进行SFT微调。

echo "--- Optional Stage: Running Enhancement SFT ---"

# 需要两个参数：
# 1. 待增强的RL checkpoint路径
# 2. 为这次增强训练命名的实验名
RL_CHECKPOINT_PATH=${1}
ENHANCE_EXPERIMENT_NAME=${2:-"$(basename $RL_CHECKPOINT_PATH)_enhanced"}

if [ -z "$RL_CHECKPOINT_PATH" ]; then
    echo "Usage: $0 <path_to_rl_checkpoint> [enhancement_experiment_name]"
    echo "Example: $0 outputs/lcare-final-run-1/checkpoints/iter_80"
    exit 1
fi

echo "Enhancing model from: $RL_CHECKPOINT_PATH"
echo "Output will be saved under: outputs/$ENHANCE_EXPERIMENT_NAME"
echo "--------------------------------------------------------"

WORLD_SIZE=2

torchrun --nproc_per_node=$WORLD_SIZE src/main.py \
    --config-name=lcare_config_final \
    main.task=train_sft \
    trainer=sft_enhance \
    trainer.model_path=$RL_CHECKPOINT_PATH \
    experiment_name=$ENHANCE_EXPERIMENT_NAME

echo "✅ Enhancement SFT finished. The improved model is ready for another round of RL."