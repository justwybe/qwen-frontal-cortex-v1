#!/bin/bash
set -e

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Omni-7B}"
MODEL_PATH="${MODEL_PATH:-/runpod-volume/models/Qwen2.5-Omni-7B}"
DTYPE="${DTYPE:-bfloat16}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

echo "============================================"
echo "  Qwen 2.5 Omni - Frontal Cortex v1"
echo "============================================"
echo "Model: ${MODEL_NAME}"
echo "Path:  ${MODEL_PATH}"
echo "Dtype: ${DTYPE}"
echo "Port:  ${PORT}"
echo "GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION}"
echo "Max Model Length: ${MAX_MODEL_LEN}"
echo "============================================"

# Download model if not already present
if [ ! -d "${MODEL_PATH}" ] || [ -z "$(ls -A ${MODEL_PATH} 2>/dev/null)" ]; then
    echo "Model not found at ${MODEL_PATH}. Downloading..."
    mkdir -p "${MODEL_PATH}"

    if [ -z "${HF_TOKEN}" ]; then
        echo "WARNING: HF_TOKEN not set. Download may fail for gated models."
    else
        huggingface-cli login --token "${HF_TOKEN}"
    fi

    huggingface-cli download "${MODEL_NAME}" \
        --local-dir "${MODEL_PATH}" \
        --local-dir-use-symlinks False

    echo "Download complete."
else
    echo "Model found at ${MODEL_PATH}. Skipping download."
fi

echo "Starting vLLM server..."

exec vllm serve "${MODEL_PATH}" \
    --port "${PORT}" \
    --host "${HOST}" \
    --dtype "${DTYPE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --trust-remote-code
