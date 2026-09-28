#!/usr/bin/env bash
# Serve Qwen3-VL-2B-Instruct as an OpenAI-compatible endpoint on 127.0.0.1:8001.
# The first run downloads the container image and the model weights.
#
# Uses the NVIDIA vLLM container by default. To use a native vLLM install
# instead, run:  VLLM_MODE=native ./scripts/start_vllm.sh
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}"
VLLM_MODE="${VLLM_MODE:-docker}"
VLLM_IMAGE="${VLLM_IMAGE:-nvcr.io/nvidia/vllm:26.07-py3}"
VLLM_API_KEY="${VLLM_API_KEY:-local-vllm-key}"
VLLM_PORT="${VLLM_PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
# 0.1 of the 128 GB unified memory on the DGX Spark (GB10) used in the paper.
# On a discrete GPU, raise this so that ~8 GB is available (e.g. 0.5 on 16 GB).
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.1}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen3-vl-vllm}"

SERVE_ARGS=(
  --served-model-name "${MODEL_ID}"
  --api-key "${VLLM_API_KEY}"
  --dtype auto
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --limit-mm-per-prompt '{"image": 1, "video": {"count": 1, "num_frames": 16}}'
  --max-num-seqs 1
  --enable-prefix-caching
)

echo "Model   : ${MODEL_ID}"
echo "Endpoint: http://127.0.0.1:${VLLM_PORT}/v1  (API key: VLLM_API_KEY)"

if [[ "${VLLM_MODE}" == "native" ]]; then
  exec vllm serve "${MODEL_ID}" --host 127.0.0.1 --port "${VLLM_PORT}" "${SERVE_ARGS[@]}"
fi

mkdir -p "${HOME}/.cache/huggingface"
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
  -p "127.0.0.1:${VLLM_PORT}:8000" \
  "${DOCKER_ENV_ARGS[@]}" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  "${VLLM_IMAGE}" \
  vllm serve "${MODEL_ID}" --host 0.0.0.0 --port 8000 "${SERVE_ARGS[@]}"
