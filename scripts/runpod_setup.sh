#!/bin/bash
set -e

# One-shot setup script for a fresh RunPod pod.
# Installs transformers + Gradio for direct Qwen 2.5 Omni inference.
# No vLLM, no compilation — just pip pre-built wheels.
#
# Usage:
#   bash scripts/runpod_setup.sh

echo "============================================"
echo "  RunPod Setup - Qwen 2.5 Omni 3B"
echo "  (Direct transformers inference)"
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

# Core inference stack
pip install transformers accelerate

# Qwen Omni multimodal utilities
pip install "qwen-omni-utils[decord]"

# Gradio web UI + audio handling
pip install gradio soundfile numpy

# Flash Attention (optional, speeds up inference — skip if it fails)
pip install flash-attn --no-build-isolation || echo "WARNING: flash-attn install failed (optional, continuing without it)"

# Model download CLI
pip install "huggingface_hub[cli]"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Download the model:  bash scripts/download_model.sh"
echo "  2. Launch everything:   bash scripts/start.sh"
echo ""
echo "The Gradio UI will be available at port 7860."
echo "Make sure to expose port 7860 in your RunPod pod settings."
