#!/bin/bash
set -e

# Launches the Gradio web UI for real-time multimodal conversation.
#
# Usage:
#   bash scripts/start_webui.sh
#
# Prerequisites:
#   - vLLM-Omni server running on port 8000 (bash scripts/start.sh)
#   - pip install gradio openai soundfile

WEBUI_PORT="${WEBUI_PORT:-7860}"
VLLM_HOST="${VLLM_HOST:-localhost}"
VLLM_PORT="${PORT:-8000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBUI_DIR="$(dirname "${SCRIPT_DIR}")/webui"

echo "============================================"
echo "  Qwen 2.5 Omni - Web UI"
echo "============================================"
echo "Web UI port:  ${WEBUI_PORT}"
echo "vLLM server:  ${VLLM_HOST}:${VLLM_PORT}"
echo "============================================"

# Wait for vLLM server to be ready
echo "Waiting for vLLM server at ${VLLM_HOST}:${VLLM_PORT}..."
MAX_RETRIES=60
RETRY=0
while ! curl -s "http://${VLLM_HOST}:${VLLM_PORT}/health" > /dev/null 2>&1; do
    RETRY=$((RETRY + 1))
    if [ ${RETRY} -ge ${MAX_RETRIES} ]; then
        echo "ERROR: vLLM server did not start. Start it first with: bash scripts/start.sh &"
        exit 1
    fi
    echo "  Retry ${RETRY}/${MAX_RETRIES}..."
    sleep 5
done
echo "vLLM server is ready!"

echo "Starting Gradio web UI..."
cd "${WEBUI_DIR}"
python app.py \
    --port "${WEBUI_PORT}" \
    --vllm-host "${VLLM_HOST}" \
    --vllm-port "${VLLM_PORT}"
