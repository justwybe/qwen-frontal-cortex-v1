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

mkdir -p "${MODEL_PATH}"

python -c "
from huggingface_hub import snapshot_download, login
import os

token = os.environ.get('HF_TOKEN')
if token:
    login(token=token)

snapshot_download(
    '${MODEL_NAME}',
    local_dir='${MODEL_PATH}',
)
"

echo "Done. Model saved to ${MODEL_PATH}"
echo "Total size:"
du -sh "${MODEL_PATH}"
