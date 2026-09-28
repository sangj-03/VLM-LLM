#!/usr/bin/env bash
# Score TCN-only, VLM-only and TCN–VLM–YOLO on the same 1.5 s windows.
#
#   ./scripts/evaluate.sh            # saved paper runs -> results/paper_20260917/evaluation
#   ./scripts/evaluate.sh --latest   # newest runs under outputs/ -> outputs/evaluation
#
# Extra arguments are passed to src/evaluation/compare_systems.py.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
cd "${REPO_ROOT}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

if [[ "${1:-}" == "--latest" ]]; then
  shift
  newest() { ls -t $1 2>/dev/null | head -n 1 || true; }
  TCN_JSONL="$(newest "${OUTPUT_ROOT}/tcn_only/*/tcn_driver_states_*.jsonl")"
  VLM_JSONL="$(newest "${OUTPUT_ROOT}/vlm_only/*/vlm_clips.jsonl")"
  FUSION_CSV="${OUTPUT_ROOT}/tcn_vlm_yolo/tcn_vlm_yolo_accuracy_matched.csv"
  for f in "${TCN_JSONL}" "${VLM_JSONL}" "${FUSION_CSV}"; do
    if [[ -z "${f}" || ! -f "${f}" ]]; then
      echo "ERROR: run all three scripts first (missing: ${f:-TCN/VLM output})" >&2
      exit 1
    fi
  done
  set -- --tcn-jsonl "${TCN_JSONL}" --vlm-jsonl "${VLM_JSONL}" \
         --fusion-csv "${FUSION_CSV}" --output-dir "${OUTPUT_ROOT}/evaluation" "$@"
fi

exec "${PYTHON}" src/evaluation/compare_systems.py "$@"
