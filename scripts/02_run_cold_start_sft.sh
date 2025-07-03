# #!/bin/bash
# set -e
# echo "--- Stage 1: Running Cold-Start SFT to generate Initial Policy ---"
# WORLD_SIZE=2 # 根据您的8x H200配置
# torchrun --nproc_per_node=$WORLD_SIZE main.py \
#     --config-name=lcare_config \
#     main.task=train_sft \
#     trainer=sft

# echo "✅ Cold-Start SFT finished."

#!/bin/bash
# scripts/02_run_cold_start_sft.sh

set -e
echo "--- Stage 2: Running Cold-Start SFT to generate Initial Policy (2x A100) ---"

# 根据您的 2x A100 配置设置
WORLD_SIZE=2 

# 使用 torchrun 启动，它会为每个进程设置好环境变量
# --standalone: 自动设置MASTER_ADDR和MASTER_PORT，非常方便
# --nnodes=1: 指定为单节点训练
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$WORLD_SIZE \
    main.py \
    --config-name=lcare_config \
    main.task=train_sft \
    trainer=sft

echo "✅ Cold-Start SFT finished."