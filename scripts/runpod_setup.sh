#!/bin/bash
set -e

# One-shot setup script for a fresh RunPod pod.
# Installs transformers + Gradio for direct Qwen 2.5 Omni inference.
# No vLLM, no compilation — just pip pre-built wheels.
#
# Removes vLLM if present to prevent pip from rebuilding it (OOM).
# Limits parallel compilation jobs as a safety net.
#
# Usage:
#   bash scripts/runpod_setup.sh

echo "============================================"
echo "  RunPod Setup - Qwen 2.5 Omni 3B"
echo "  (Direct transformers inference)"
echo "============================================"

# ── Aggressively remove vLLM ──────────────────────────────────────────
# RunPod images often ship vLLM from source. If pip's resolver sees it,
# it tries to rebuild CUDA kernels → OOM / disk full. Remove every trace.
echo "Removing vLLM (if present)..."
pip uninstall vllm -y 2>/dev/null || true
# Remove source trees wherever they may be
rm -rf /workspace/vllm /root/vllm /opt/vllm
# Remove any leftover metadata from site-packages so pip forgets vLLM
find / -type d -name "vllm*.dist-info" -exec rm -rf {} + 2>/dev/null || true
find / -type d -name "vllm*.egg-info" -exec rm -rf {} + 2>/dev/null || true
find / -type f -name "vllm.egg-link" -delete 2>/dev/null || true
# Purge pip cache
pip cache purge 2>/dev/null || true
echo "vLLM cleanup done."

# Limit parallel compilation jobs as a safety net
export MAX_JOBS=2

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
pip install --no-cache-dir --upgrade pip setuptools wheel

# Core inference stack
pip install --no-cache-dir transformers accelerate

# Qwen Omni multimodal utilities
pip install --no-cache-dir "qwen-omni-utils[decord]"

# Gradio web UI + audio handling
pip install --no-cache-dir gradio soundfile numpy

# Flash Attention (optional, speeds up inference — skip if already installed or fails)
if python -c "import flash_attn" 2>/dev/null; then
    echo "flash-attn already installed, skipping"
else
    MAX_JOBS=2 pip install --no-cache-dir flash-attn --no-build-isolation \
        || echo "WARNING: flash-attn install failed (optional, continuing without it)"
fi

# huggingface_hub CLI is included in the base package (no [cli] extra needed)

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
