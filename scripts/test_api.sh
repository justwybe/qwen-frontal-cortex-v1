#!/bin/bash

# Test the vLLM OpenAI-compatible API endpoint.
#
# Usage:
#   bash scripts/test_api.sh [host:port]
#
# Example:
#   bash scripts/test_api.sh localhost:8000

ENDPOINT="${1:-localhost:8000}"

echo "Testing vLLM endpoint at ${ENDPOINT}..."
echo ""

# Health check
echo "--- Health Check ---"
curl -s "http://${ENDPOINT}/health" && echo " OK" || echo " FAILED"
echo ""

# List models
echo "--- Available Models ---"
curl -s "http://${ENDPOINT}/v1/models" | python3 -m json.tool 2>/dev/null || echo "Failed to list models"
echo ""

# Chat completion
echo "--- Chat Completion Test ---"
curl -s "http://${ENDPOINT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen2.5-Omni-7B",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! What model are you?"}
        ],
        "max_tokens": 128,
        "temperature": 0.7
    }' | python3 -m json.tool 2>/dev/null || echo "Failed to get completion"
