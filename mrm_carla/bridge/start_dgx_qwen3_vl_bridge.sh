#!/usr/bin/env bash

# Run the driver-monitoring bridge against the local Qwen3-VL vLLM server.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}"
VLLM_API_KEY="${VLLM_API_KEY:-qwen-carla-local}"
VLLM_HOST_PORT="${VLLM_HOST_PORT:-8001}"
BRIDGE_PORT="${BRIDGE_PORT:-8000}"
LLM_TIMEOUT="${LLM_TIMEOUT:-120}"
TELEMETRY_MAX_AGE="${TELEMETRY_MAX_AGE:-2}"

BRIDGE_TOKEN_ARGS=()
if [[ -n "${LLM_BRIDGE_TOKEN:-}" ]]; then
  BRIDGE_TOKEN_ARGS=(--token "${LLM_BRIDGE_TOKEN}")
fi

exec python3 "${SCRIPT_DIR}/dgx_llm_control_server.py" \
  --host 127.0.0.1 \
  --port "${BRIDGE_PORT}" \
  --backend openai \
  --model "${MODEL_ID}" \
  --llm-url "http://127.0.0.1:${VLLM_HOST_PORT}/v1/chat/completions" \
  --llm-api-key "${VLLM_API_KEY}" \
  --llm-timeout "${LLM_TIMEOUT}" \
  --telemetry-max-age "${TELEMETRY_MAX_AGE}" \
  "${BRIDGE_TOKEN_ARGS[@]}"
