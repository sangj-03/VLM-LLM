#!/usr/bin/env bash

set -euo pipefail

CARLA_PROJECT_DIR="${CARLA_PROJECT_DIR:-/home/tmo/carla}"
UE4_EDITOR_BIN="${UE4_EDITOR_BIN:-/home/tmo/UnrealEngine_4.26/Engine/Binaries/Linux/UE4Editor}"
CARLA_PORT="${CARLA_PORT:-2000}"

mapfile -t LISTENER_PIDS < <(
  lsof -t -iTCP:"${CARLA_PORT}" -sTCP:LISTEN 2>/dev/null || true
)

if (( ${#LISTENER_PIDS[@]} > 0 )); then
  EXISTING_PID="${LISTENER_PIDS[0]}"
  EXISTING_COMMAND="$(ps -p "${EXISTING_PID}" -o args= 2>/dev/null || true)"
  if [[ "${EXISTING_COMMAND}" == *CarlaUE4* \
      && "${EXISTING_COMMAND}" == *Town04_Opt* ]]; then
    echo "Town04_Opt CARLA is already running on port ${CARLA_PORT}."
    echo "Reusing PID ${EXISTING_PID}; Ctrl+C stops the reused backend."
    stop_reused_carla() {
      kill -TERM "${EXISTING_PID}" 2>/dev/null || true
      exit 0
    }
    trap stop_reused_carla INT TERM
    while kill -0 "${EXISTING_PID}" 2>/dev/null; do
      sleep 2
    done
    exit 0
  fi

  echo "Port ${CARLA_PORT} is occupied by a non-Town04_Opt process:" >&2
  echo "  PID ${EXISTING_PID}: ${EXISTING_COMMAND}" >&2
  exit 1
fi

exec "${UE4_EDITOR_BIN}" \
  "${CARLA_PROJECT_DIR}/Unreal/CarlaUE4/CarlaUE4.uproject" \
  /Game/Carla/Maps/Town04_Opt \
  -game \
  -carla-server \
  -carla-rpc-port="${CARLA_PORT}" \
  -windowed \
  -ResX=1280 \
  -ResY=720 \
  -quality-level=Low \
  -nosound
