#!/bin/bash
# [V-FINAL-ROBUST] 评估一个指定的模型checkpoint
# 职责:
# 1. 自动查找指定实验的最新checkpoint。
# 2. 对该checkpoint在所有评估数据集上运行评估。
# 3. 结果保存在一个以评估模型命名的独立目录中。

set -e

# --- 参数配置 ---
# 第一个参数是您想评估的RL实验名。
# 如果不提供，默认评估我们用于公平对比的实验。
DEFAULT_EXPERIMENT_NAME="L-CARE_Fair_Comparison_Run_1"
EXP_TO_EVAL=${1:-$DEFAULT_EXPERIMENT_NAME}

# 定义checkpoint的根目录
CKPT_ROOT_DIR="outputs/$EXP_TO_EVAL"

echo "--- Stage 3: Evaluating a Trained Model from Experiment: $EXP_TO_EVAL ---"

# --- 检查路径和查找最新checkpoint ---
if [ ! -d "$CKPT_ROOT_DIR" ]; then
    echo "❌ Error: Experiment directory not found at '$CKPT_ROOT_DIR'"
    echo "Please make sure you have run the RL training for this experiment first."
    exit 1
fi

# 查找最新的迭代目录 (例如: iter_80, iter_90, iter_100)
# `ls -d` 只列出目录, `sort -V` 按版本号正确排序, `tail -n 1` 取最后一个
LATEST_ITER_DIR=$(ls -d "$CKPT_ROOT_DIR"/iter_* 2>/dev/null | sort -V | tail -n 1)

if [ -z "$LATEST_ITER_DIR" ]; then
    echo "❌ Error: No iteration checkpoints (e.g., 'iter_*') found in '$CKPT_ROOT_DIR'"
    exit 1
fi

# --- 执行评估 ---
MODEL_TO_EVAL=$LATEST_ITER_DIR
# 为这次评估创建一个独立的实验名，例如 "eval_iter_100"
EVAL_EXPERIMENT_NAME="eval_$(basename "$MODEL_TO_EVAL")_from_${EXP_TO_EVAL}"

echo "Model to Evaluate: $MODEL_TO_EVAL"
echo "Evaluation Output Sub-directory: $EVAL_EXPERIMENT_NAME"
echo "--------------------------------------------------------"

# 评估是在单个GPU上完成的，以保证结果的可复现性
# - 'evaluation.model_path' 指定要加载权重的目录
# - 'experiment_name' 用于创建存放结果的子目录
python src/main.py \
    --config-name=lcare_config_final \
    main.task=evaluate \
    experiment_name=$EVAL_EXPERIMENT_NAME \
    evaluation.model_path=$MODEL_TO_EVAL

echo "✅ Evaluation finished. Results are in outputs/eval_results/$EVAL_EXPERIMENT_NAME"