#!/bin/bash
set -e

# One-shot setup script for a fresh RunPod pod with RTX 4090.
# Builds vLLM-Omni from Qwen's custom fork for full audio output support.
#
# Usage:
#   export HF_TOKEN=hf_your_token
#   bash scripts/runpod_setup.sh

echo "============================================"
echo "  RunPod Setup - Qwen 2.5 Omni 3B"
echo "  (vLLM-Omni with Thinker+Talker)"
echo "============================================"

# Check for volume
VOLUME="/workspace"
if [ ! -d "${VOLUME}" ]; then
    echo "ERROR: No volume found at ${VOLUME}."
    exit 1
fi

echo "Volume found at ${VOLUME}"

# Install system deps
apt-get update && apt-get install -y ffmpeg libsndfile1

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install build dependencies for vLLM-Omni
pip install setuptools_scm torchdiffeq resampy x_transformers accelerate

# Install Qwen Omni dependencies
pip install "qwen-omni-utils[decord]"

# Build vLLM-Omni from Qwen's custom fork (supports --omni flag for speech output)
echo "Building vLLM-Omni from source (this requires ~16GB+ RAM)..."
if [ -d "/workspace/vllm" ]; then
    echo "Removing existing vLLM source directory..."
    rm -rf /workspace/vllm
fi

git clone -b qwen2_omni_public https://github.com/fyabc/vllm.git /workspace/vllm
cd /workspace/vllm
git checkout 729feed3ec2beefe63fda30a345ef363d08062f8
pip install -r requirements/cuda.txt
pip install .
cd /workspace

# Install latest transformers from source (needed for Qwen2.5-Omni support)
pip install git+https://github.com/huggingface/transformers

# Install Gradio web UI dependencies
pip install gradio openai soundfile

# Install huggingface-cli for model downloads
pip install huggingface_hub[cli]

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Download the model:  bash scripts/download_model.sh"
echo "  2. Start vLLM server:   bash scripts/start.sh"
echo "  3. Start web UI:        bash scripts/start_webui.sh"
echo ""
echo "The web UI will be available at port 7860."
echo "Make sure to expose port 7860 in your RunPod pod settings."
