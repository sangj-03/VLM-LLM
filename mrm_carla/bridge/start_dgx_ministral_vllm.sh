#!/usr/bin/env bash

# Run multimodal Ministral 3 8B in NVIDIA's DGX Spark vLLM container.
set -euo pipefail

MODEL_ID="${MODEL_ID:-mistralai/Ministral-3-8B-Instruct-2512}"
VLLM_IMAGE="${VLLM_IMAGE:-nvcr.io/nvidia/vllm:26.05.post1-py3}"
VLLM_API_KEY="${VLLM_API_KEY:-ministral-carla-local}"
VLLM_HOST_PORT="${VLLM_HOST_PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.30}"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is not installed." >&2
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: set HF_TOKEN after accepting the model license on Hugging Face." >&2
  echo "Example: export HF_TOKEN='hf_...'" >&2
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -Fxq ministral-vllm; then
  echo "ERROR: container 'ministral-vllm' already exists." >&2
  echo "Inspect it with: docker logs ministral-vllm" >&2
  echo "Remove it with:  docker rm -f ministral-vllm" >&2
  exit 1
fi

mkdir -p "${HOME}/.cache/huggingface"

echo "Starting ${MODEL_ID} on host http://127.0.0.1:${VLLM_HOST_PORT}"
echo "Image: ${VLLM_IMAGE}"
echo "The first start downloads the model and can take several minutes."

exec docker run --rm \
  --name ministral-vllm \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p "127.0.0.1:${VLLM_HOST_PORT}:8000" \
  -e "HF_TOKEN=${HF_TOKEN}" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  "${VLLM_IMAGE}" \
  vllm serve "${MODEL_ID}" \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name "${MODEL_ID}" \
    --api-key "${VLLM_API_KEY}" \
    --tokenizer-mode mistral \
    --config-format mistral \
    --load-format mistral \
    --dtype auto \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-num-seqs 1 \
    --limit-mm-per-prompt '{"image":1}' \
    --enable-prefix-caching
