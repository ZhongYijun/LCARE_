#!/bin/bash

# scripts/00a_download_models.sh
# -----------------------------------------------------------------------------
# [NEW] 这个脚本负责从魔搭社区下载本项目所需的所有模型到本地的 'models/' 目录。
# 这可以避免在训练过程中因网络问题导致的中断，并确保实验的可复现性。
# -----------------------------------------------------------------------------

echo "======================================================================================"
echo "                         STEP 0a: DOWNLOADING ALL MODELS                              "
echo "======================================================================================"

# 激活conda环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lcare

# 检查Conda环境是否激活
if [ $? -ne 0 ]; then
    echo "Conda environment 'lcare' could not be activated. Please check your setup."
    exit 1
fi

# 创建用于存放模型的根目录
mkdir -p models

# 1. 下载 Qwen2.5-7B-Instruct (用于Actor和RFT生成)
echo "Downloading Qwen/Qwen2.5-7B-Instruct..."
modelscope download Qwen/Qwen2.5-7B-Instruct --local_dir models/Qwen2.5-7B-Instruct

# 2. 下载 Qwen/Qwen2-1.5B-Instruct (一个较小的模型，用作Critic和Token Reward Model)
# 注意：我们选择 1.5B 而非 1.8B 是因为 Qwen2 系列中 1.5B 是最接近的开源尺寸。
# 这与原论文的 "同源" 思想保持一致。
echo "Downloading Qwen/Qwen2.5-3B-Instruct..."
modelscope download Qwen/Qwen2.5-3B-Instruct --local_dir models/Qwen2.5-3B-Instruct


echo "✅ All models downloaded successfully to the 'models/' directory."