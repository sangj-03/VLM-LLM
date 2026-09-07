#!/usr/bin/env bash

set -euo pipefail

TOOL_DIR="/home/tmo/carla_tools/town04_autopilot"

if ! command -v gnome-terminal >/dev/null 2>&1; then
  echo "gnome-terminal is required to open the three-terminal layout." >&2
  exit 1
fi

gnome-terminal \
  --window \
  --title="1 - CARLA Town04_Opt" \
  -- bash -lc "'${TOOL_DIR}/run_carla_town04_opt.sh'; status=\$?; echo; echo '[CARLA exited, status='\"\${status}\"']'; exec bash"

gnome-terminal \
  --window \
  --title="2 - Spawn ego vehicle" \
  -- bash -lc "'${TOOL_DIR}/spawn_ego_vehicle.py'; status=\$?; echo; echo '[Spawner exited, status='\"\${status}\"']'; exec bash"

gnome-terminal \
  --window \
  --title="3 - BehaviorAgent autopilot" \
  -- bash -lc "'/home/tmo/.local/bin/carla06auto'; status=\$?; echo; echo '[Autopilot exited, status='\"\${status}\"']'; exec bash"

echo "Opened Town04_Opt CARLA, ego spawn, and autopilot in three terminals."
