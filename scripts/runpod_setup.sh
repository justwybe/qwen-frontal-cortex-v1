#!/bin/bash
set -e

# One-shot setup script for a fresh RunPod pod with RTX A6000.
# Run this after SSH-ing into your pod.
#
# Usage:
#   export HF_TOKEN=hf_your_token
#   bash scripts/runpod_setup.sh

echo "============================================"
echo "  RunPod Setup - Qwen 2.5 Omni"
echo "============================================"

# Check for network volume
VOLUME="/workspace"
if [ ! -d "${VOLUME}" ]; then
    echo "ERROR: No volume found at ${VOLUME}."
    exit 1
fi

echo "Network volume found at ${VOLUME}"

# Install system deps
apt-get update && apt-get install -y ffmpeg

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch (if not pre-installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install vLLM dependencies
pip install setuptools_scm torchdiffeq resampy x_transformers accelerate
pip install "qwen-omni-utils[decord]"

# Install vLLM from Qwen's custom fork
if [ ! -d "/workspace/vllm" ]; then
    git clone -b qwen2_omni_public https://github.com/fyabc/vllm.git "/workspace/vllm"
    cd "/workspace/vllm"
    git checkout 729feed3ec2beefe63fda30a345ef363d08062f8
else
    cd "/workspace/vllm"
fi

pip install -r requirements/cuda.txt
pip install .

pip install git+https://github.com/huggingface/transformers
pip install huggingface_hub[cli]

echo ""
echo "Setup complete. Next steps:"
echo "  1. Download the model:  bash scripts/download_model.sh"
echo "  2. Start the server:    bash scripts/start.sh"
