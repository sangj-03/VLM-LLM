#!/usr/bin/env bash
# Baseline (paper): the direct three-state TCN (v35, 1.5 s, 5 seeds) decides
# the driver state by itself. No VLM or bridge is needed.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/_common.sh"

OUT="${OUT:-${OUTPUT_ROOT}/tcn_only}"
mkdir -p "${OUT}"

exec "${PYTHON}" src/pipeline/run_tcn_only.py \
  --video-path "${VIDEO}" \
  --tcn-config configs/tcn/tcn_1p5s_32d.json \
  --tcn-ensemble-config configs/ensembles/multitask_state_tcn_v35.json \
  --driver-yolo-model "${WEIGHTS}/yolo26n.pt" \
  --event-output-dir "${OUT}" \
  --tcn-ground-truth-file "${GROUND_TRUTH}" \
  --tcn-accuracy-plot-file "${OUT}/tcn_only_accuracy.png" \
  --tcn-matched-file "${OUT}/tcn_only_matched.csv" \
  "${PREVIEW_ARGS[@]}" \
  "$@"
