#!/usr/bin/env bash

set -euo pipefail

TOOL_DIR="/home/tmo/carla_tools/town04_autopilot"
BRIDGE_DIR="/home/tmo/carla_tools/llm_control_bridge"
PYTHON="/usr/bin/python3"
ROLE_NAME="town06_ego"
OVERRIDE_FILE="/tmp/carla_town04_ego_safety_override.json"
MONITOR_IDLE_EXIT_SEC="${CARLA_MONITOR_IDLE_EXIT_SEC:-5}"

START_SUPERVISOR=0
FORCE_MRM=0
NO_WARNING_DISPLAY=0
AUTOPILOT_ARGS=()

while (( $# > 0 )); do
  case "$1" in
    --force-mrm)
      START_SUPERVISOR=1
      FORCE_MRM=1
      shift
      ;;
    --with-driver-monitor)
      START_SUPERVISOR=1
      NO_WARNING_DISPLAY=1
      shift
      ;;
    --no-warning-display)
      NO_WARNING_DISPLAY=1
      START_SUPERVISOR=1
      shift
      ;;
    *)
      AUTOPILOT_ARGS+=("$1")
      shift
      ;;
  esac
done

SUPERVISOR_PID=""
AUTOPILOT_PID=""

stop_stack() {
  if [[ -n "${SUPERVISOR_PID}" ]] && kill -0 "${SUPERVISOR_PID}" 2>/dev/null; then
    kill -TERM "${SUPERVISOR_PID}" 2>/dev/null || true
  fi
  if [[ -n "${AUTOPILOT_PID}" ]] && kill -0 "${AUTOPILOT_PID}" 2>/dev/null; then
    kill -TERM "${AUTOPILOT_PID}" 2>/dev/null || true
  fi
}
trap stop_stack INT TERM EXIT

if (( START_SUPERVISOR )); then
  SUPERVISOR_ARGS=(
    --role-name "${ROLE_NAME}"
    --behavior-override-file "${OVERRIDE_FILE}"
    --action shoulder
    --required-no-response-observations 1
    --display-required-reduced-observations 3
    --display-required-active-observations 4
  )
  if (( NO_WARNING_DISPLAY )); then
    SUPERVISOR_ARGS+=(--no-warning-display)
  fi
  if (( FORCE_MRM )); then
    SUPERVISOR_ARGS+=(--force-mrm)
  else
    # DGX code stays untouched. A finite dataset is considered complete when
    # this PC has received at least one fresh result and then sees no newer
    # result for the configured interval.
    SUPERVISOR_ARGS+=(
      --exit-after-monitor-idle-sec "${MONITOR_IDLE_EXIT_SEC}"
      --auto-resume
      --required-active-observations 4
    )
  fi
  "${PYTHON}" "${BRIDGE_DIR}/carla_safety_supervisor.py" \
    "${SUPERVISOR_ARGS[@]}" &
  SUPERVISOR_PID=$!
fi

"${PYTHON}" "${TOOL_DIR}/run_behavior_autopilot.py" \
  --role-name "${ROLE_NAME}" \
  --behavior-override-file "${OVERRIDE_FILE}" \
  "${AUTOPILOT_ARGS[@]}" &
AUTOPILOT_PID=$!

wait "${AUTOPILOT_PID}"
AUTOPILOT_STATUS=$?
AUTOPILOT_PID=""

if [[ -n "${SUPERVISOR_PID}" ]] && kill -0 "${SUPERVISOR_PID}" 2>/dev/null; then
  kill -TERM "${SUPERVISOR_PID}" 2>/dev/null || true
  wait "${SUPERVISOR_PID}" 2>/dev/null || true
fi
SUPERVISOR_PID=""
trap - INT TERM EXIT
exit "${AUTOPILOT_STATUS}"
