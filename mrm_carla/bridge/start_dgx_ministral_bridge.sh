#!/usr/bin/env bash

# Expose an image-triggered driver-monitor endpoint while vLLM stays local.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ID="${MODEL_ID:-mistralai/Ministral-3-8B-Instruct-2512}"
VLLM_API_KEY="${VLLM_API_KEY:-ministral-carla-local}"
VLLM_HOST_PORT="${VLLM_HOST_PORT:-8001}"
BRIDGE_PORT="${BRIDGE_PORT:-8000}"

if [[ -z "${LLM_BRIDGE_TOKEN:-}" ]]; then
  echo "ERROR: set LLM_BRIDGE_TOKEN to the same secret on DGX and CARLA PC." >&2
  echo "Generate one with: python3 -c 'import secrets; print(secrets.token_urlsafe(32))'" >&2
  exit 1
fi

exec python3 "${SCRIPT_DIR}/dgx_llm_control_server.py" \
  --host 0.0.0.0 \
  --port "${BRIDGE_PORT}" \
  --backend openai \
  --model "${MODEL_ID}" \
  --llm-url "http://127.0.0.1:${VLLM_HOST_PORT}/v1/chat/completions" \
  --llm-api-key "${VLLM_API_KEY}" \
  --token "${LLM_BRIDGE_TOKEN}" \
  --llm-timeout 10.0 \
  --telemetry-max-age 5.0
