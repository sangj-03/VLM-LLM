#!/usr/bin/env bash
# List every file the three runners need and report which are missing.
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
VIDEO="${VIDEO:-assets/videos/1000013115_0-287s.mp4}"
PYTHON="${PYTHON:-python3}"

missing=0
check() {  # check <path> <description>
  if [[ -e "$1" ]]; then
    printf '  [ok]      %-80s %s\n' "$1" "$2"
  else
    printf '  [missing] %-80s %s\n' "$1" "$2"
    missing=$((missing + 1))
  fi
}

list_tcn_checkpoints() {
  "${PYTHON}" -c '
import json
for name, label in [
    ("observable_state_tcn_v38.json", "main TCN (TCN-VLM-YOLO)"),
    ("control_engagement_tcn_1p5s.json", "control TCN (TCN-VLM-YOLO)"),
    ("fall_onset_tcn_v32_1s.json", "fall-onset TCN (TCN-VLM-YOLO)"),
    ("multitask_state_tcn_v35.json", "state TCN (TCN-only)"),
]:
    for path in json.load(open("configs/ensembles/" + name))["model_paths"]:
        print(path + "\t" + label)
'
}

echo "Public weights (./scripts/download_assets.sh):"
check assets/weights/yolo26n.pt           "YOLO26n driver detector"
check assets/weights/face_landmarker.task "MediaPipe Face Landmarker"

echo "Project checkpoints (not distributed):"
check assets/weights/eye_state_mrl_v2_pretrained.pt "eye open/closed classifier"
while IFS=$'\t' read -r path label; do
  check "${path}" "${label}"
done < <(list_tcn_checkpoints)

echo "Input video:"
check "${VIDEO}" "in-cabin video (not distributed)"

echo
if (( missing > 0 )); then
  echo "${missing} file(s) missing. See README (준비물)."
  exit 1
fi
echo "All assets present."
