#!/usr/bin/env bash
# Download the public third-party weights into assets/weights and verify them.
# Project-trained TCN / eye-state checkpoints are not public; see README.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS="${REPO_ROOT}/assets/weights"
mkdir -p "${WEIGHTS}"

fetch() {  # fetch <url> <file> <sha256>
  local url="$1" out="${WEIGHTS}/$2" sha="$3"
  if [[ -f "${out}" ]] && echo "${sha}  ${out}" | sha256sum --check --status; then
    echo "ok       $2"
    return
  fi
  echo "download $2"
  curl -fL --retry 3 -o "${out}.part" "${url}"
  echo "${sha}  ${out}.part" | sha256sum --check --status \
    || { echo "ERROR: checksum mismatch for $2" >&2; rm -f "${out}.part"; exit 1; }
  mv "${out}.part" "${out}"
}

# Driver detector used with ByteTrack (Ultralytics YOLO26n, AGPL-3.0).
fetch https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt \
  yolo26n.pt 9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef
# MediaPipe Face Landmarker (eye blendshape features, Apache-2.0).
fetch https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
  face_landmarker.task 64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff

"${REPO_ROOT}/scripts/check_assets.sh"
