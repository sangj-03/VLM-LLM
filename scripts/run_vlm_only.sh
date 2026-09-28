#!/usr/bin/env bash
# Baseline (paper): Qwen3-VL-2B-Instruct classifies every consecutive
# 1.5 s YOLO driver clip, with no TCN candidate selection.
#
# Requires the VLM bridge in VLM-only mode: ./scripts/start_bridge.sh vlm-only
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

OUT="${OUT:-${OUTPUT_ROOT}/vlm_only}"
mkdir -p "${OUT}"

exec "${PYTHON}" src/pipeline/run_vlm_only.py \
  --video-path "${VIDEO}" \
  --video-start-sec 0 \
  --bridge-url "${BRIDGE_URL}" \
  --bridge-timeout 300 \
  --vlm-only \
  --vlm-clip-seconds 1.5 \
  --yolo-driver-crop \
  --driver-yolo-model "${WEIGHTS}/yolo26n.pt" \
  --driver-tracker bytetrack.yaml \
  --driver-yolo-confidence 0.15 \
  --driver-yolo-interval 1 \
  --driver-min-x-ratio 0.45 \
  --driver-anchor-x 1260 \
  --driver-anchor-y 550 \
  --driver-box-hold-sec 2.0 \
  --vlm-fixed-driver-box-padding 0.20 \
  --vlm-driver-seat-left-ratio 0.57 \
  --event-output-dir "${OUT}" \
  --ground-truth-file "${GROUND_TRUTH}" \
  --accuracy-plot-file "${OUT}/vlm_only_accuracy.png" \
  "${PREVIEW_ARGS[@]}" \
  "$@"
