#!/usr/bin/env bash
# Proposed system (paper): YOLO26n+ByteTrack driver tracking -> 32-D features
# -> 5-model causal TCN ensemble (1.5 s @ 10 Hz) selects candidates
# -> Qwen3-VL-2B-Instruct verifies only the candidate 1.5 s driver clips.
#
# Requires the VLM bridge in fusion mode: ./scripts/start_bridge.sh fusion
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

OUT="${OUT:-${OUTPUT_ROOT}/tcn_vlm_yolo}"
mkdir -p "${OUT}"

exec "${PYTHON}" src/pipeline/run_tcn_vlm_yolo.py \
  --video-path "${VIDEO}" \
  --video-start-sec 0 \
  --bridge-url "${BRIDGE_URL}" \
  --bridge-timeout 300 \
  --no-inline-telemetry \
  --tcn-config configs/tcn/tcn_1p5s_32d.json \
  --tcn-ensemble-config configs/ensembles/observable_state_tcn_v38.json \
  --fall-onset-tcn-config configs/tcn/fall_onset_1s_32d.json \
  --fall-onset-tcn-ensemble-config configs/ensembles/fall_onset_tcn_v32_1s.json \
  --fast-onset-risk \
  --tcn-interval-sec 0.1 \
  --fall-onset-trigger-persist-sec 0.1 \
  --driver-yolo-model "${WEIGHTS}/yolo26n.pt" \
  --driver-yolo-interval 1 \
  --event-threshold 0.50 \
  --eye-state-model "${WEIGHTS}/eye_state_mrl_v2_pretrained.pt" \
  --eye-state-smoothing-sec 1.0 \
  --eye-closed-probability-threshold 0.70 \
  --eye-closed-persist-sec 0.6 \
  --eye-min-visible-fraction 0.60 \
  --eye-risk-weight 0.45 \
  --combined-risk-threshold 0.50 \
  --qwen-recheck-sec 1.0 \
  --eye-uncertain-tcn-ratio 0.59 \
  --eye-uncertain-watchdog-sec 1.0 \
  --event-output-dir "${OUT}" \
  --vlm-driver-seat-left-ratio 0.57 \
  --vlm-match-tcn-crop \
  --candidate-seconds 1.5 \
  --event-media-format video \
  --event-full-source-fps \
  --ground-truth-file "${GROUND_TRUTH}" \
  --accuracy-plot-file "${OUT}/tcn_vlm_yolo_accuracy.png" \
  "${PREVIEW_ARGS[@]}" \
  "$@"
