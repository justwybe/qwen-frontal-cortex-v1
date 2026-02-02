#!/bin/bash
set -e

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Omni-3B}"
MODEL_PATH="${MODEL_PATH:-/workspace/models/Qwen2.5-Omni-3B}"
WEBUI_PORT="${WEBUI_PORT:-7860}"

echo "============================================"
echo "  Qwen 2.5 Omni - Frontal Cortex v1"
echo "  (Direct transformers inference)"
echo "============================================"
echo "Model: ${MODEL_NAME}"
echo "Path:  ${MODEL_PATH}"
echo "Port:  ${WEBUI_PORT}"
echo "============================================"

# Download model if not already present
if [ ! -d "${MODEL_PATH}" ] || [ -z "$(ls -A ${MODEL_PATH} 2>/dev/null)" ]; then
    echo "Model not found at ${MODEL_PATH}. Downloading..."
    mkdir -p "${MODEL_PATH}"

    if [ -z "${HF_TOKEN}" ]; then
        echo "WARNING: HF_TOKEN not set. Download may fail for gated models."
    else
        python -m huggingface_hub.cli login --token "${HF_TOKEN}"
    fi

    python -m huggingface_hub.cli download "${MODEL_NAME}" \
        --local-dir "${MODEL_PATH}" \
        --local-dir-use-symlinks False

    echo "Download complete."
else
    echo "Model found at ${MODEL_PATH}. Skipping download."
fi

# Resolve the webui directory relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBUI_DIR="$(dirname "${SCRIPT_DIR}")/webui"

echo "Starting Gradio app..."
exec python "${WEBUI_DIR}/app.py" \
    --model-path "${MODEL_PATH}" \
    --port "${WEBUI_PORT}"
