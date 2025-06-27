#!/bin/bash
set -e
echo "--- Stage 1: Running Cold-Start SFT to generate Initial Policy ---"
WORLD_SIZE=8 # 根据您的8x H200配置
torchrun --nproc_per_node=$WORLD_SIZE src/main.py \
    --config-name=lcare_config_final \
    main.task=train_sft \
    trainer=sft

echo "✅ Cold-Start SFT finished."