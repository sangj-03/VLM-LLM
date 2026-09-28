# Shared defaults for the run scripts. Source this file; do not execute it.
# Every value can be overridden from the environment, e.g.
#   PYTHON=~/miniconda3/envs/driver_env/bin/python VIDEO=/data/cabin.mp4 ./scripts/run_tcn_vlm_yolo.sh

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
VIDEO="${VIDEO:-${REPO_ROOT}/assets/videos/1000013115_0-287s.mp4}"
GROUND_TRUTH="${GROUND_TRUTH:-${REPO_ROOT}/data/labels/1000013115_ground_truth_nonoverlap_4s.csv}"
BRIDGE_URL="${BRIDGE_URL:-http://127.0.0.1:8000}"
# Shared secret between runner and bridge; must match scripts/start_bridge.sh.
export LLM_BRIDGE_TOKEN="${LLM_BRIDGE_TOKEN:-local-bridge-token}"
WEIGHTS="${REPO_ROOT}/assets/weights"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}"

cd "${REPO_ROOT}"

if [[ ! -f "${VIDEO}" ]]; then
  echo "ERROR: input video not found: ${VIDEO}" >&2
  echo "       Put it under assets/videos/ or set VIDEO=/path/to/video.mp4" >&2
  exit 1
fi

# Without a display, OpenCV's preview window cannot open (Qt xcb error).
# Real-time pacing is kept so VLM call timing matches the paper runs.
PREVIEW_ARGS=()
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  PREVIEW_ARGS=(--no-video-preview)
fi
