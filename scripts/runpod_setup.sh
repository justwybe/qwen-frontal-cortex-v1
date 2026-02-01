#!/bin/bash
set -e

# One-shot setup script for a fresh RunPod pod with RTX 4090.
#
# Usage:
#   export HF_TOKEN=hf_your_token
#   bash scripts/runpod_setup.sh

echo "============================================"
echo "  RunPod Setup - Qwen 2.5 Omni 3B"
echo "============================================"

# Check for volume
VOLUME="/workspace"
if [ ! -d "${VOLUME}" ]; then
    echo "ERROR: No volume found at ${VOLUME}."
    exit 1
fi

echo "Volume found at ${VOLUME}"

# Install system deps
apt-get update && apt-get install -y ffmpeg

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install vLLM via pip (pre-built, no compilation needed)
pip install "vllm>=0.8.5.post1"

# Install Qwen Omni dependencies
pip install transformers accelerate
pip install "qwen-omni-utils[decord]"
pip install huggingface_hub[cli]

echo ""
echo "Setup complete. Next steps:"
echo "  1. Download the model:  bash scripts/download_model.sh"
echo "  2. Start the server:    bash scripts/start.sh"
