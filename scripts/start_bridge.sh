#!/usr/bin/env bash
# Start the HTTP bridge (127.0.0.1:8000) that builds the driver-state prompt,
# calls Qwen3-VL through vLLM and returns the three-state decision.
#
#   ./scripts/start_bridge.sh fusion     # for run_tcn_vlm_yolo.sh (TCN context in prompt)
#   ./scripts/start_bridge.sh vlm-only   # for run_vlm_only.sh
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
MODE="${1:-fusion}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}"
VLLM_API_KEY="${VLLM_API_KEY:-local-vllm-key}"
VLLM_PORT="${VLLM_PORT:-8001}"
BRIDGE_PORT="${BRIDGE_PORT:-8000}"
LLM_TIMEOUT="${LLM_TIMEOUT:-300}"

case "${MODE}" in
  fusion)   SERVER="${REPO_ROOT}/src/bridge/vlm_bridge_server.py" ;;
  vlm-only) SERVER="${REPO_ROOT}/src/bridge/vlm_bridge_server_vlm_only.py" ;;
  *) echo "usage: $0 [fusion|vlm-only]" >&2; exit 2 ;;
esac

# Shared secret between runner and bridge; the runners use the same default.
LLM_BRIDGE_TOKEN="${LLM_BRIDGE_TOKEN:-local-bridge-token}"

exec "${PYTHON}" "${SERVER}" \
  --host 127.0.0.1 \
  --port "${BRIDGE_PORT}" \
  --backend openai \
  --model "${MODEL_ID}" \
  --llm-url "http://127.0.0.1:${VLLM_PORT}/v1/chat/completions" \
  --llm-api-key "${VLLM_API_KEY}" \
  --llm-timeout "${LLM_TIMEOUT}" \
  --telemetry-max-age 86400 \
  --token "${LLM_BRIDGE_TOKEN}"
