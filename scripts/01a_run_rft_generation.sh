#!/bin/bash

# scripts/01a_run_rft_generation.sh
# 这个新脚本负责执行RFT的数据生成流程。

echo "======================================================================================"
echo "                            STEP 1a: GENERATING RFT DATA                              "
echo "======================================================================================"

# 激活conda环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lcare

# 检查Conda环境是否激活
if [ $? -ne 0 ]; then
    echo "Conda environment 'lcare' could not be activated. Please check your setup."
    exit 1
fi

# 设置Hugging Face镜像（如果需要）
export HF_ENDPOINT=https://hf-mirror.com
export DEEPSEEK_API_KEY="sk-6c947070444143cd85853da2f2c6551d"
# 运行RFT数据生成脚本
# 它会使用 configs/trainer/sft.yaml 中的 rft_generation 部分的配置
# 注意：这个过程可能会非常耗时和消耗计算资源！
python src/data_processing/generate_rft_data.py \
    --config-name=lcare_config.yaml \
    trainer=sft

echo "✅ RFT data generation script finished."