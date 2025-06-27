#!/bin/bash
set -e
echo "--- Stage 2: Running L-CARE RL Training (Fair Comparison) ---"
WORLD_SIZE=2
EXPERIMENT_NAME="L-CARE_Fair_Comparison_Run_1" # 为这次实验命名

torchrun --nproc_per_node=$WORLD_SIZE src/main.py \
    --config-name=lcare_config_final \
    main.task=train_rl \
    trainer=rl_agent_fair_comparison \
    experiment_name=$EXPERIMENT_NAME

echo "✅ RL Training finished for experiment: $EXPERIMENT_NAME"