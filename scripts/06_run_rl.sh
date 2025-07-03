#!/bin/bash
set -e
echo "--- Stage 2: Running L-CARE RL Training (Fair Comparison) ---"
WORLD_SIZE=2
EXPERIMENT_NAME="L-CARE_Run_1" # 为这次实验命名
# export NCCL_CONNECT_TIMEOUT=120000
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export DEEPSEEK_API_KEY="sk-6c947070444143cd85853da2f2c6551d"
torchrun --nproc_per_node=$WORLD_SIZE main.py \
    --config-name=lcare_config \
    main.task=train_rl \
    trainer=rl_agent_ablation \
    experiment_name=$EXPERIMENT_NAME

echo "✅ RL Training finished for experiment: $EXPERIMENT_NAME"