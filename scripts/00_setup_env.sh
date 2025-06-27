#!/bin/bash
# [V-FINAL-SECURE] L-CARE项目的完整、安全的环境准备脚本
set -e

# --- 1. 检查Conda ---
if ! command -v conda &> /dev/null; then
    echo "❌ Conda could not be found. Please install Miniconda or Anaconda."
    exit 1
fi
echo "✅ Conda found."

# --- 2. 创建Conda环境 ---
ENV_NAME="lcare"
if conda info --envs | grep -q "^$ENV_NAME\s"; then
    echo "✅ Conda environment '$ENV_NAME' already exists. Skipping creation."
else
    echo "--------------------------------------------------"
    echo "Creating Conda environment named '$ENV_NAME' with Python 3.10..."
    conda create -n $ENV_NAME python=3.10 -y
    echo "✅ Conda environment '$ENV_NAME' created."
fi

# --- 3. 激活环境并安装依赖 ---
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME
echo "--------------------------------------------------"
echo "Environment '$ENV_NAME' activated. Installing dependencies..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
# faiss-gpu can be tricky, conda is the most reliable way
conda install -c pytorch faiss-gpu -y
pip install -r requirements.txt
echo "✅ All dependencies installed."

# --- 4. 配置API Keys ---
configure_api_key() {
    local key_name="$1"
    local env_var_name="$2"
    local auth_url="$3"

    echo "--------------------------------------------------"
    echo "🔧 Configuring $key_name..."

    # 检查环境变量是否已设置
    if [ -n "${!env_var_name}" ]; then
        echo "✅ $env_var_name is already set in your environment."
    else
        echo "Your $env_var_name is not set."
        echo "You can find your API key at: $auth_url"
        read -p "Please enter your $key_name API Key (or press Enter to skip): " user_key
        if [ -n "$user_key" ]; then
            export "$env_var_name"="$user_key"
            echo "✅ $env_var_name has been set for this session."
            echo "💡 To make it permanent, add this to your ~/.bashrc or ~/.zshrc:"
            echo "   export $env_var_name=\"$user_key\""
        else
            echo "⚠️ Skipping $key_name configuration."
            return
        fi
    fi
}

configure_api_key "WandB" "WANDB_API_KEY" "https://wandb.ai/authorize"
configure_api_key "DeepSeek" "DEEPSEEK_API_KEY" "https://platform.deepseek.com/api_keys"

if [ -n "$WANDB_API_KEY" ]; then
    wandb login
fi

echo "--------------------------------------------------"
echo "🎉 Environment setup complete!"
echo "To activate this environment, run: conda activate $ENV_NAME"
echo "--------------------------------------------------"