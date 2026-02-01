#!/bin/bash
set -e

# Run this on your RunPod to pre-download the model to network volume.
# This avoids re-downloading every time the pod restarts.
#
# Usage:
#   export HF_TOKEN=hf_your_token
#   bash scripts/download_model.sh

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Omni-3B}"
MODEL_PATH="${MODEL_PATH:-/workspace/models/Qwen2.5-Omni-3B}"

echo "Downloading ${MODEL_NAME} to ${MODEL_PATH}..."

pip install -q huggingface_hub[cli]

if [ -n "${HF_TOKEN}" ]; then
    huggingface-cli login --token "${HF_TOKEN}"
fi

mkdir -p "${MODEL_PATH}"

huggingface-cli download "${MODEL_NAME}" \
    --local-dir "${MODEL_PATH}" \
    --local-dir-use-symlinks False

echo "Done. Model saved to ${MODEL_PATH}"
echo "Total size:"
du -sh "${MODEL_PATH}"
