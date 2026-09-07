#!/usr/bin/env bash

# Run Qwen3-VL-2B-Instruct as a one-image-at-a-time OpenAI-compatible server.
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}"
VLLM_IMAGE="${VLLM_IMAGE:-nvcr.io/nvidia/vllm:26.07-py3}"
VLLM_API_KEY="${VLLM_API_KEY:-qwen-carla-local}"
VLLM_HOST_PORT="${VLLM_HOST_PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen3-vl-vllm}"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is not installed." >&2
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  echo "ERROR: container '${CONTAINER_NAME}' already exists." >&2
  echo "Stop it first with: docker stop ${CONTAINER_NAME}" >&2
  exit 1
fi

mkdir -p "${HOME}/.cache/huggingface"

echo "Starting multimodal model: ${MODEL_ID}"
echo "API endpoint: http://127.0.0.1:${VLLM_HOST_PORT}/v1"
echo "The first run downloads the Docker image and model weights."

DOCKER_ENV_ARGS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  DOCKER_ENV_ARGS=(-e "HF_TOKEN=${HF_TOKEN}")
fi

exec docker run --rm \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p "127.0.0.1:${VLLM_HOST_PORT}:8000" \
  "${DOCKER_ENV_ARGS[@]}" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  "${VLLM_IMAGE}" \
  vllm serve "${MODEL_ID}" \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name "${MODEL_ID}" \
    --api-key "${VLLM_API_KEY}" \
    --dtype auto \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --limit-mm-per-prompt '{"image": 1}' \
    --max-num-seqs 1 \
    --enable-prefix-caching
