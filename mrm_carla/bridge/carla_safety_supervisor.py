#!/usr/bin/env python3

"""Turn cached driver-response results into a conservative CARLA action.

This is a simulation safety supervisor, not a production road-planning stack.
It never asks the LLM for steering or braking. By default it keeps searching
for a CARLA shoulder waypoint, changes outward one lane at a time, and only
finishes the manoeuvre after braking on an actual Shoulder lane.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import json
from typing import Any, Dict, List, Optional, Tuple

from carla_llm_overlay import wait_for_vehicle, wait_for_world
import carla
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.local_planner import LocalPlanner, RoadOption
from agents.tools.misc import get_trafficlight_trigger_location


SAFE_ZONE_LANE_TYPES = carla.LaneType.Shoulder | carla.LaneType.Parking
MRM_LANE_TYPES = carla.LaneType.Driving | SAFE_ZONE_LANE_TYPES
DRIVER_RESPONSES = {
    "active", "reduced", "no_visible_response",
}
STOP_TARGET_MARGIN_M = 0.3
MRM_MIN_STOP_HOLD_SEC = 1.0


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def normalized_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def extract_driver_state(response: Dict[str, Any]) -> Dict[str, Any]:
    """Accept both the bridge envelope and a direct driver-state object."""
    nested = response.get("driver_state")
    return nested if isinstance(nested, dict) else response


def driver_response(driver_state: Dict[str, Any],
                    min_confidence: float = 0.0) -> Optional[str]:
    """Return one of the three response labels accepted by vehicle control.

    Newer response-only models may omit ``confidence``.  When it is present we
    enforce the configured threshold. Invalid, malformed, low-confidence, and
    unknown results return ``None``; they never become a fourth driver-response
    state and never start or release a minimal-risk manoeuvre.
    """
    if not isinstance(driver_state, dict):
        return None
    if driver_state.get("assessment_valid") is False:
        return None
    value = str(driver_state.get("driver_response", "")) \
        .strip().lower().replace("-", "_").replace(" ", "_")
    if value == "no_visible":
        value = "no_visible_response"
    if value not in DRIVER_RESPONSES:
        return None
    if "confidence" in driver_state:
        try:
            confidence = float(driver_state["confidence"])
        except (TypeError, ValueError):
            return None
        if not 0.0 <= confidence <= 1.0 or confidence < min_confidence:
            return None
    return value


class WarningDisplayPolicy:
    """Debounce display opening and closing across distinct observations."""

    def __init__(self, required_reduced_observations: int = 3,
                 required_active_observations: int = 4,
                 required_no_visible_observations: int = 3) -> None:
        self.required_reduced_observations = required_reduced_observations
        self.required_active_observations = required_active_observations
        self.required_no_visible_observations = (
            required_no_visible_observations)
        self.reduced_observations = 0
        self.active_observations = 0
        self.no_visible_observations = 0

    def observe(self, response: Optional[str]) -> Tuple[bool, bool]:
        """Return ``(open_display, close_display)`` for one new observation."""
        if response == "reduced":
            self.reduced_observations = min(
                self.reduced_observations + 1,
                self.required_reduced_observations)
            self.active_observations = 0
            self.no_visible_observations = 0
            return (
                self.reduced_observations >=
                self.required_reduced_observations,
                False,
            )
        if response == "active":
            self.reduced_observations = 0
            self.active_observations = min(
                self.active_observations + 1,
                self.required_active_observations)
            self.no_visible_observations = 0
            return (
                False,
                self.active_observations >=
                self.required_active_observations,
            )
        if response == "no_visible_response":
            self.reduced_observations = 0
            self.active_observations = 0
            self.no_visible_observations = min(
                self.no_visible_observations + 1,
                self.required_no_visible_observations)
            return (
                self.no_visible_observations >=
                self.required_no_visible_observations,
                False,
            )

        # Missing/invalid observations break both consecutive sequences.
        self.reduced_observations = 0
        self.active_observations = 0
        self.no_visible_observations = 0
        return False, False


class WarningDisplayManager:
    """Own the viewer process that is shown for a degraded driver response."""

    def __init__(self, script_path: str, server_url: str, token: str,
                 poll_hz: float, window_name: str,
                 enabled: bool = True,
                 restart_cooldown_sec: float = 3.0) -> None:
        self.script_path = os.path.abspath(script_path)
        self.server_url = server_url
        self.token = token
        self.poll_hz = poll_hz
        self.window_name = window_name
        self.enabled = enabled
        self.restart_cooldown_sec = restart_cooldown_sec
        self._process: Optional[subprocess.Popen] = None
        self._last_start_at = -float("inf")
        self._status = "idle" if enabled else "disabled"

    def status(self) -> str:
        process = self._process
        if process is not None and process.poll() is None:
            return "running pid=%d" % process.pid
        return self._status

    def ensure_running(self) -> bool:
        """Start one viewer, or keep the viewer already owned by this manager."""
        if not self.enabled:
            return False
        process = self._process
        if process is not None:
            exit_code = process.poll()
            if exit_code is None:
                return True
            self._process = None
            self._report("viewer exited with code %d" % exit_code)

        now = time.monotonic()
        if now - self._last_start_at < self.restart_cooldown_sec:
            return False
        self._last_start_at = now
        if not os.path.isfile(self.script_path):
            self._report("viewer script not found: %s" % self.script_path)
            return False

        command = [
            sys.executable,
            self.script_path,
            "--server-url", self.server_url,
            "--poll-hz", str(self.poll_hz),
            "--window-name", self.window_name,
        ]
        child_env = os.environ.copy()
        # Keep the bearer token out of the process command line.
        child_env["LLM_BRIDGE_TOKEN"] = self.token
        try:
            self._process = subprocess.Popen(command, env=child_env)
        except OSError as exc:
            self._report(
                "failed to start viewer: %s: %s" % (
                    type(exc).__name__, exc))
            return False
        self._report("viewer started pid=%d" % self._process.pid)
        return True

    def stop(self, reason: str = "requested") -> None:
        """Close only the viewer process that this manager started."""
        process = self._process
        if process is None:
            return
        self._process = None
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        self._report("viewer closed: %s" % reason)

    def _report(self, message: str) -> None:
        if message != self._status:
            print("WARNING DISPLAY: %s" % message, flush=True)
        self._status = message


class DriverStatePoller:
    """Fetch the latest cached result without blocking the CARLA loop."""

    def __init__(self, url: str, token: str, timeout_sec: float,
                 period_sec: float) -> None:
        self.url = url
        self.token = token
        self.timeout_sec = timeout_sec
        self.period_sec = period_sec
        self._lock = threading.Lock()
        self._latest: Optional[Dict[str, Any]] = None
        self._latest_received_at = 0.0
        self._last_error = "waiting for driver state"
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="driver-state-poller", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.timeout_sec + 0.5)

    def latest(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._latest is None:
                return None
            result = dict(self._latest)
            result["result_age_s"] = float(
                result.get("result_age_s", 0.0)) \
                + max(0.0, time.monotonic() - self._latest_received_at)
            return result

    def status(self) -> str:
        with self._lock:
            return self._last_error

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                result = self._get()
                with self._lock:
                    self._latest = result
                    self._latest_received_at = time.monotonic()
                    self._last_error = ""
            except Exception as exc:
                with self._lock:
                    self._last_error = "%s: %s" % (type(exc).__name__, exc)
            self._stop.wait(self.period_sec)

    def _get(self) -> Dict[str, Any]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        request = urllib.request.Request(self.url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                body = response.read(64 * 1024)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise RuntimeError("no driver state available yet")
            details = exc.read(1024).decode("utf-8", errors="replace")
            raise RuntimeError("DGX HTTP %d: %s" % (exc.code, details))
        result = json.loads(body.decode("utf-8"))
        if not isinstance(result, dict):
            raise ValueError("driver state response must be a JSON object")
        return result


def set_hazard_lights(vehicle: carla.Vehicle, enabled: bool) -> None:
    try:
        if enabled:
            lights = (
                carla.VehicleLightState.LeftBlinker
                | carla.VehicleLightState.RightBlinker
            )
        else:
            lights = carla.VehicleLightState.NONE
        vehicle.set_light_state(carla.VehicleLightState(lights))
    except Exception as exc:
        print("LIGHT WARNING: failed to set hazard lights: %s" % exc, flush=True)


def set_turn_signal(vehicle: carla.Vehicle, side: Optional[str]) -> None:
    try:
        if side == "left":
            lights = carla.VehicleLightState.LeftBlinker
        elif side == "right":
            lights = carla.VehicleLightState.RightBlinker
        else:
            lights = carla.VehicleLightState.NONE
        vehicle.set_light_state(carla.VehicleLightState(lights))
    except Exception as exc:
        print("LIGHT WARNING: failed to set turn signal: %s" % exc, flush=True)


def nearby_vehicle_blocks_shoulder(world: carla.World,
                                   ego: carla.Vehicle,
                                   shoulder: carla.Waypoint) -> bool:
    target = shoulder.transform.location
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.id == ego.id:
            continue
        if actor.get_location().distance(target) < 8.0:
            return True
    return False


def lane_end_location(waypoint: carla.Waypoint) -> carla.Location:
    """Return the last forward point on the current driving lane."""
    try:
        candidates = waypoint.next_until_lane_end(2.0)
    except (RuntimeError, AttributeError, TypeError):
        candidates = []
    valid = [candidate for candidate in candidates
             if candidate.road_id == waypoint.road_id
             and candidate.lane_id == waypoint.lane_id
             and candidate.lane_type == carla.LaneType.Driving]
    return ((valid[-1].transform.location if valid
             else waypoint.transform.location))


def upcoming_stop_signal(world: carla.World, vehicle: carla.Vehicle,
                         waypoint: carla.Waypoint
                         ) -> Optional[Tuple[Any, carla.Location, float]]:
    """Return the nearest red/yellow stop line on the current road direction."""
    ego_transform = vehicle.get_transform()
    ego_location = ego_transform.location
    ego_forward = ego_transform.get_forward_vector()
    ego_right = ego_transform.get_right_vector()
    try:
        lane_width = max(2.0, float(waypoint.lane_width))
    except (TypeError, ValueError, AttributeError):
        return None
    speed_mps = vehicle.get_velocity().length()
    lookahead = clamp(20.0 + speed_mps * 2.5, 30.0, 60.0)
    affected_light = vehicle.get_traffic_light()
    lights = list(world.get_actors().filter("*traffic_light*"))
    if affected_light is not None and affected_light not in lights:
        lights.append(affected_light)

    candidates = []
    stop_states = (carla.TrafficLightState.Red,
                   carla.TrafficLightState.Yellow)
    for traffic_light in lights:
        try:
            state = traffic_light.get_state()
        except (RuntimeError, AttributeError):
            continue
        if state not in stop_states:
            continue
        try:
            trigger_location = get_trafficlight_trigger_location(
                traffic_light)
            trigger_waypoint = world.get_map().get_waypoint(
                trigger_location, project_to_road=True,
                lane_type=carla.LaneType.Driving)
        except (RuntimeError, AttributeError, TypeError):
            trigger_location = lane_end_location(waypoint)
            trigger_waypoint = world.get_map().get_waypoint(
                trigger_location, project_to_road=True,
                lane_type=carla.LaneType.Driving)
        if trigger_waypoint is None:
            continue
        if (trigger_waypoint.road_id != waypoint.road_id
                or not ShoulderLanePlanner._same_direction(
                    waypoint, trigger_waypoint)):
            continue
        delta = trigger_location - ego_location
        longitudinal = (delta.x * ego_forward.x
                        + delta.y * ego_forward.y)
        lateral = abs(delta.x * ego_right.x + delta.y * ego_right.y)
        if longitudinal < -1.0 or longitudinal > lookahead:
            continue
        if lateral > lane_width * 1.75 and traffic_light is not affected_light:
            continue
        candidates.append((longitudinal, state,
                           trigger_location))

    if not candidates:
        return None
    longitudinal, state, target = min(candidates, key=lambda item: item[0])
    return state, target, longitudinal


def forward_hazard_reason(world: carla.World, vehicle: carla.Vehicle,
                          waypoint: carla.Waypoint) -> Optional[str]:
    """Prevent an MRM lane change from entering a red light or occupied path."""
    speed_mps = vehicle.get_velocity().length()
    lookahead = clamp(12.0 + speed_mps * 2.5, 20.0, 55.0)
    # Keep the MRM's current-lane signal hold aligned with the normal
    # autopilot stop target. Signals farther away are reported as "ahead" so
    # the planner can keep searching for a safe shoulder without stopping at
    # the old 15 m point.
    signal_stop_horizon = STOP_TARGET_MARGIN_M

    # Use the trigger/stop line, not the traffic-light pole transform. Town04
    # poles can be tens of metres from the line and previously caused an MRM to
    # perform a full emergency stop before it ever searched for a shoulder.
    ego_transform = vehicle.get_transform()
    ego_location = ego_transform.location
    ego_forward = ego_transform.get_forward_vector()
    try:
        for traffic_light in world.get_actors().filter("*traffic_light*"):
            state = traffic_light.get_state()
            if state not in (carla.TrafficLightState.Red,
                              carla.TrafficLightState.Yellow):
                continue
            try:
                trigger_location = get_trafficlight_trigger_location(
                    traffic_light)
                trigger_waypoint = world.get_map().get_waypoint(
                    trigger_location, project_to_road=True,
                    lane_type=carla.LaneType.Driving)
            except (RuntimeError, AttributeError, TypeError):
                trigger_location = lane_end_location(waypoint)
                trigger_waypoint = world.get_map().get_waypoint(
                    trigger_location, project_to_road=True,
                    lane_type=carla.LaneType.Driving)
            if (trigger_waypoint is None
                    or trigger_waypoint.road_id != waypoint.road_id
                    or not ShoulderLanePlanner._same_direction(
                        waypoint, trigger_waypoint)):
                continue
            delta = trigger_location - ego_location
            longitudinal = delta.x * ego_forward.x + delta.y * ego_forward.y
            threshold = lookahead if state == carla.TrafficLightState.Red else 20.0
            if 0.0 < longitudinal < threshold:
                if longitudinal <= signal_stop_horizon:
                    return ("red traffic light stop line"
                            if state == carla.TrafficLightState.Red
                            else "yellow traffic light stop line")
                return ("red traffic light ahead"
                        if state == carla.TrafficLightState.Red
                        else "yellow traffic light ahead")
    except Exception:
        pass

    transform = ego_transform
    forward = ego_forward
    right = transform.get_right_vector()
    origin = ego_location
    lane_width = max(2.0, float(waypoint.lane_width))
    actors = list(world.get_actors().filter("vehicle.*"))
    actors += list(world.get_actors().filter("walker.pedestrian.*"))
    forward_traffic_seen = False
    for actor in actors:
        if actor.id == vehicle.id:
            continue
        delta = actor.get_location() - origin
        longitudinal = delta.x * forward.x + delta.y * forward.y
        lateral = abs(delta.x * right.x + delta.y * right.y)
        obstacle_horizon = max(24.0, min(45.0, lookahead))
        if (2.0 < longitudinal < obstacle_horizon
                and lateral < lane_width * 0.65):
            actor_velocity = actor.get_velocity()
            actor_forward_speed = (
                actor_velocity.x * forward.x + actor_velocity.y * forward.y)
            closing_speed = max(0.0, speed_mps - actor_forward_speed)
            emergency_horizon = clamp(
                4.0 + closing_speed * closing_speed / (2.0 * 3.5),
                6.0, 16.0)
            if longitudinal <= emergency_horizon:
                return "forward path occupied"
            forward_traffic_seen = True

    if forward_traffic_seen:
        return "forward traffic ahead"

    # Do not start a lane change inside an intersection. This topology-only
    # result is deliberately checked after real collision hazards so callers
    # may crawl straight through the junction without overlooking an actor.
    if waypoint.is_junction:
        return "inside intersection"

    distance = 5.0
    while distance <= lookahead:
        if any(future.is_junction for future in waypoint.next(distance)):
            return "junction ahead"
        distance += 5.0
    return None


def current_forward_hazard(world: carla.World,
                           vehicle: carla.Vehicle) -> Optional[str]:
    waypoint = world.get_map().get_waypoint(
        vehicle.get_location(), project_to_road=True,
        lane_type=carla.LaneType.Driving)
    if waypoint is None:
        return "no driving waypoint"
    reason = forward_hazard_reason(world, vehicle, waypoint)
    # A recovered driver may hand control back to BehaviorAgent, which has
    # its own junction planner. Red lights and occupied paths still block the
    # release, but merely being near a junction must not lock MRM forever.
    if reason in ("junction ahead", "inside intersection"):
        return None
    return reason


class ShoulderLanePlanner:
    """Scenario-based MRM: approach a shoulder, then stop at a safe target."""

    MAX_STEER = 0.18
    # The final merge is deliberately brisk: once the swept corridor has
    # passed the occupancy checks below, lingering beside the live lane is
    # less desirable than completing the move promptly.  These limits remain
    # below the planner-wide steering cap and the CLI's 30 km/h MRM ceiling.
    MAX_STEER_STEP = 0.040
    MAX_THROTTLE = 0.40
    MERGE_MAX_STEER = 0.16
    MERGE_MAX_THROTTLE = 0.40
    MERGE_TARGET_SPEED_KPH = 28.0
    POSITIONING_SPEED_MAX_KPH = 30.0
    # MRM is an emergency manoeuvre: begin moving outward almost immediately
    # and use a compact, but still smooth, lane-change trajectory.  The
    # occupancy and swept-path checks below remain mandatory before this plan
    # is accepted.
    DRIVING_LANE_CHANGE_SAME_LANE_M = 2.0
    DRIVING_LANE_CHANGE_OTHER_LANE_M = 4.0
    DRIVING_LANE_CHANGE_DISTANCE_M = 12.0
    SHOULDER_CHANGE_SAME_LANE_M = 2.0
    SHOULDER_CHANGE_DISTANCE_M = 8.0
    MERGE_LOOKAHEAD_M = 8.0
    SHOULDER_CENTER_LOOKAHEAD_M = 4.0
    # Do not hold an MRM for multiple control ticks merely to re-centre.  One
    # verified control tick retains a pose check while allowing the next
    # outward-lane/shoulder transition to start on the same cycle.
    SETTLE_TICKS = 1
    STOP_SETTLE_TICKS = 10
    STOP_HEADING_TOLERANCE_DEG = 3.0
    STOP_TARGET_MIN_AHEAD_M = 3.0
    STOP_TARGET_MAX_AHEAD_M = 30.0
    STOP_TARGET_STEP_M = 2.0
    STOP_TARGET_CLEARANCE_M = 8.0
    STOP_COMPLETE_DISTANCE_M = 1.5
    # Enough room to complete the low-speed merge and keep a short buffer;
    # overly long buffers made narrow Town04 roads search for several minutes.
    STOP_TARGET_FRONT_CLEARANCE_M = 8.0
    STOP_TARGET_REAR_CLEARANCE_M = 5.0
    STOP_TARGET_MIN_SHOULDER_RUN_M = 12.0
    # Keep a modest clearance on both sides while allowing usable narrow
    # shoulders: 0.10 m per side is added to the vehicle's physical width.
    SHOULDER_SIDE_MARGIN_M = 0.10
    ROADSIDE_BARRIER_CLEARANCE_M = 0.15
    ROADSIDE_RAY_LENGTH_M = 4.0
    # Validate the actual merge/stop corridor, not an entire distant road.
    ROADSIDE_SCAN_AHEAD_M = 24.0
    ROADSIDE_SCAN_STEP_M = 3.0
    SHOULDER_TRACK_TOLERANCE_M = 0.12
    # A distant signal is latched for safety, but it must not cancel the
    # outward-lane positioning manoeuvre. Only the final approach owns braking.
    SIGNAL_POSITIONING_STOP_DISTANCE_M = 6.0

    def __init__(self, world: carla.World, vehicle: carla.Vehicle,
                 target_speed_kph: float = 18.0) -> None:
        self.world = world
        self.vehicle = vehicle
        self.map = world.get_map()
        self.target_speed_kph = float(target_speed_kph)
        # Keep intersection traversal and gap-searching conservative even when
        # the normal MRM cruise speed is raised from the command line.
        self.caution_speed_kph = min(self.target_speed_kph, 12.0)
        self.planner = LocalPlanner(vehicle, opt_dict={
            "dt": 0.05,
            "target_speed": self.target_speed_kph,
            "sampling_radius": 1.5,
            "max_throttle": self.MAX_THROTTLE,
            "max_brake": 0.30,
            "max_steering": self.MAX_STEER,
            "base_min_distance": 2.0,
            "distance_ratio": 0.30,
            "lateral_control_dict": {
                "K_P": 0.78, "K_I": 0.0, "K_D": 0.04, "dt": 0.05,
            },
            "longitudinal_control_dict": {
                "K_P": 0.45, "K_I": 0.02, "K_D": 0.0, "dt": 0.05,
            },
        }, map_inst=self.map)
        self.target_lane_id: Optional[int] = None
        self.target_road_id: Optional[int] = None
        self.target_lane_type = None
        self.side: Optional[str] = None
        self.phase = "approach"
        self.stop_target: Optional[carla.Waypoint] = None
        self.stop_ready = False
        self.stop_stable_ticks = 0
        self._safe_surface_cache: Dict[Tuple[int, int, int, int], bool] = {}
        self.lane_change_number = 0
        self.settling = True
        self.stable_ticks = 0
        self.previous_steer = clamp(
            float(vehicle.get_control().steer), -self.MAX_STEER,
            self.MAX_STEER)
        self.minimum_shoulder_width_m = (
            2.0 * float(vehicle.bounding_box.extent.y)
            + 2.0 * self.SHOULDER_SIDE_MARGIN_M)
        print(
            "MRM PREFLIGHT  vehicle_width=%.2fm minimum_shoulder_width=%.2fm"
            % (2.0 * float(vehicle.bounding_box.extent.y),
               self.minimum_shoulder_width_m),
            flush=True)

    def _positioning_speed_kph(self) -> float:
        """Keep moving promptly toward the outer lane before shoulder braking."""
        try:
            current_speed_kph = 3.6 * float(
                self.vehicle.get_velocity().length())
        except (AttributeError, TypeError, ValueError):
            current_speed_kph = self.target_speed_kph
        return clamp(
            max(self.target_speed_kph, current_speed_kph),
            self.target_speed_kph, self.POSITIONING_SPEED_MAX_KPH)

    def _set_planner_speed(self, target_speed_kph: float) -> None:
        """Update an active driving-lane-change plan without rebuilding it."""
        try:
            self.planner.set_speed(target_speed_kph)
        except (AttributeError, RuntimeError, TypeError):
            pass

    @staticmethod
    def _same_direction(source: carla.Waypoint,
                        candidate: carla.Waypoint) -> bool:
        source_forward = source.transform.get_forward_vector()
        candidate_forward = candidate.transform.get_forward_vector()
        return (source_forward.x * candidate_forward.x
                + source_forward.y * candidate_forward.y) >= 0.8

    @staticmethod
    def _is_safe_zone_lane(waypoint: carla.Waypoint) -> bool:
        return waypoint.lane_type in (
            carla.LaneType.Shoulder, carla.LaneType.Parking)

    def _outward_lane(self, waypoint: carla.Waypoint
                      ) -> Tuple[Optional[str], Optional[carla.Waypoint]]:
        """Find the adjacent lane farther from the road centerline.

        CARLA lane-id signs describe the OpenDRIVE reference direction, not
        the driver's left/right. Checking both neighbors avoids steering
        toward the centerline on the opposite carriageway.
        """
        candidates = (
            ("right", waypoint.get_right_lane()),
            ("left", waypoint.get_left_lane()),
        )
        for side, candidate in candidates:
            if (candidate is None
                    or candidate.lane_type != carla.LaneType.Driving
                    or candidate.road_id != waypoint.road_id):
                continue
            if not self._same_direction(waypoint, candidate):
                continue
            if (candidate.lane_id == 0
                    or candidate.lane_id * waypoint.lane_id <= 0
                    or abs(candidate.lane_id) <= abs(waypoint.lane_id)):
                continue
            return side, candidate
        return None, None

    def _outward_shoulder(self, waypoint: carla.Waypoint
                          ) -> Tuple[Optional[str], Optional[carla.Waypoint]]:
        """Find the shoulder immediately outside the outermost Driving lane."""
        candidates = (
            ("right", waypoint.get_right_lane()),
            ("left", waypoint.get_left_lane()),
        )
        for side, candidate in candidates:
            if (candidate is None
                    or not self._is_safe_zone_lane(candidate)
                    or candidate.road_id != waypoint.road_id
                    or not self._shoulder_is_wide_enough(candidate)):
                continue
            if not self._same_direction(waypoint, candidate):
                continue
            if (candidate.lane_id == 0
                    or candidate.lane_id * waypoint.lane_id <= 0
                    or abs(candidate.lane_id) <= abs(waypoint.lane_id)):
                continue
            return side, candidate
        return None, None

    def _shoulder_is_wide_enough(self, waypoint: carla.Waypoint) -> bool:
        """Validate a mapped vehicle-safe shoulder/parking strip.

        Town04 contains some OpenDRIVE Shoulder lanes whose rendered surface
        is pedestrian pavement.  Lane type and width alone are therefore not
        sufficient: the strip must be non-junction, level with and adjacent
        to a same-direction Driving lane, and raycast as a road surface.
        """
        if (not self._is_safe_zone_lane(waypoint)
                or waypoint.is_junction
                or float(waypoint.lane_width)
                < self.minimum_shoulder_width_m):
            return False
        _side, driving = self._inward_driving_lane(waypoint)
        if driving is None:
            return False
        if abs(float(waypoint.transform.location.z)
               - float(driving.transform.location.z)) > 0.12:
            return False
        return self._safe_zone_surface_is_road(waypoint)

    def _shoulder_center_tolerance(self, waypoint: carla.Waypoint) -> float:
        """Maximum centre error that preserves the configured side margin."""
        available = (
            float(waypoint.lane_width) * 0.5
            - float(self.vehicle.bounding_box.extent.y)
            - self.SHOULDER_SIDE_MARGIN_M)
        return clamp(available, 0.10, 0.25)

    def _shoulder_target_offset(self, waypoint: carla.Waypoint) -> float:
        """Return the lateral target within a shoulder/parking strip.

        In an unobstructed strip this is the geometric centre.  When a
        roadside obstacle narrows the usable width, it becomes the centre of
        the remaining vehicle-safe corridor instead of steering to either
        edge of that corridor.
        """
        if not self._is_safe_zone_lane(waypoint):
            return 0.0
        bounds = self._shoulder_safe_center_bounds(
            waypoint, self._roadside_obstacle_distance(waypoint))
        if bounds is None:
            return 0.0
        outward_sign, inner_limit, outer_limit = bounds
        return outward_sign * (inner_limit + outer_limit) * 0.5

    def _inward_driving_lane(self, waypoint: carla.Waypoint
                             ) -> Tuple[Optional[str], Optional[carla.Waypoint]]:
        """Return the same-direction Driving lane adjacent to a Shoulder."""
        for side, candidate in (
                ("left", waypoint.get_left_lane()),
                ("right", waypoint.get_right_lane())):
            if (candidate is not None
                    and candidate.lane_type == carla.LaneType.Driving
                    and candidate.road_id == waypoint.road_id
                    and self._same_direction(waypoint, candidate)):
                return side, candidate
        return None, None

    def _roadside_edge(self, waypoint: carla.Waypoint) -> Optional[str]:
        """Return the physical outer edge when no same-direction lane exists."""
        neighbours = {}
        for side, candidate in (
                ("right", waypoint.get_right_lane()),
                ("left", waypoint.get_left_lane())):
            if (candidate is not None
                    and candidate.road_id == waypoint.road_id
                    and self._same_direction(waypoint, candidate)
                    and (candidate.lane_type == carla.LaneType.Driving
                         or (self._is_safe_zone_lane(candidate)
                             and self._shoulder_is_wide_enough(candidate)))):
                neighbours[side] = candidate
        if "right" not in neighbours and "left" in neighbours:
            return "right"
        if "left" not in neighbours and "right" in neighbours:
            return "left"
        return None

    def _safe_zone_surface_is_road(self, waypoint: carla.Waypoint) -> bool:
        """Reject sidewalk/terrain meshes even if OpenDRIVE says Shoulder."""
        cache = getattr(self, "_safe_surface_cache", None)
        if cache is None:
            cache = {}
            self._safe_surface_cache = cache
        key = (int(waypoint.road_id), int(getattr(waypoint, "section_id", 0)),
               int(waypoint.lane_id), int(float(getattr(waypoint, "s", 0.0)) // 3.0))
        if key in cache:
            return cache[key]
        cast_ray = getattr(self.world, "cast_ray", None)
        if cast_ray is None:
            cache[key] = False
            return False
        centre = waypoint.transform.location
        start = carla.Location(x=centre.x, y=centre.y, z=centre.z + 2.0)
        end = carla.Location(x=centre.x, y=centre.y, z=centre.z - 2.0)
        try:
            hits = cast_ray(start, end)
        except (RuntimeError, TypeError):
            cache[key] = False
            return False
        surface_labels = {
            carla.CityObjectLabel.Roads,
            carla.CityObjectLabel.RoadLines,
            carla.CityObjectLabel.Sidewalks,
            carla.CityObjectLabel.Terrain,
            carla.CityObjectLabel.Ground,
        }
        relevant = [hit for hit in hits if hit.label in surface_labels]
        relevant.sort(key=lambda hit: start.distance(hit.location))
        label = relevant[0].label if relevant else None
        safe = label in (carla.CityObjectLabel.Roads,
                         carla.CityObjectLabel.RoadLines)
        cache[key] = safe
        if not safe:
            print("MRM SAFE ZONE REJECTED  road=%d lane=%d surface=%s" % (
                waypoint.road_id, waypoint.lane_id,
                str(label) if label is not None else "no-road-hit"),
                flush=True)
        return safe

    def _roadside_obstacle_distance(self,
                                    waypoint: carla.Waypoint) -> Optional[float]:
        """Return the nearest roadside obstacle distance from lane centre."""
        if not self._is_safe_zone_lane(waypoint):
            return None
        cast_ray = getattr(self.world, "cast_ray", None)
        edge = self._roadside_edge(waypoint)
        if cast_ray is None or edge is None:
            return None
        right = waypoint.transform.get_right_vector()
        outward_sign = 1.0 if edge == "right" else -1.0
        centre = waypoint.transform.location
        obstacle_labels = {
            carla.CityObjectLabel.GuardRail,
            carla.CityObjectLabel.Fences,
            carla.CityObjectLabel.Walls,
            carla.CityObjectLabel.Buildings,
            carla.CityObjectLabel.Poles,
            carla.CityObjectLabel.Static,
            carla.CityObjectLabel.Other,
            carla.CityObjectLabel.Roads,
            carla.CityObjectLabel.Sidewalks,
            carla.CityObjectLabel.Terrain,
            carla.CityObjectLabel.Bridge,
        }
        nearest = float("inf")
        for height in (0.25, 0.40, 0.70, 1.05):
            start = carla.Location(
                x=centre.x, y=centre.y, z=centre.z + height)
            end = carla.Location(
                x=start.x + right.x * outward_sign
                * self.ROADSIDE_RAY_LENGTH_M,
                y=start.y + right.y * outward_sign
                * self.ROADSIDE_RAY_LENGTH_M,
                z=start.z)
            try:
                hits = cast_ray(start, end)
            except (RuntimeError, TypeError):
                return 0.0
            for hit in hits:
                if hit.label in obstacle_labels:
                    nearest = min(nearest, start.distance(hit.location))
        return nearest

    def _shoulder_safe_center_bounds(
            self, waypoint: carla.Waypoint,
            obstacle_distance: Optional[float]
            ) -> Optional[Tuple[float, float, float]]:
        """Return outward-axis limits for the vehicle centre in this strip."""
        edge = self._roadside_edge(waypoint)
        if edge is None or obstacle_distance is None:
            return None
        outward_sign = 1.0 if edge == "right" else -1.0
        lane_half_width = float(waypoint.lane_width) * 0.5
        vehicle_half_width = float(self.vehicle.bounding_box.extent.y)
        inner_limit = (-lane_half_width + vehicle_half_width
                       + self.SHOULDER_SIDE_MARGIN_M)
        nominal_outer_limit = (lane_half_width - vehicle_half_width
                               - self.SHOULDER_SIDE_MARGIN_M)
        obstacle_outer_limit = (obstacle_distance - vehicle_half_width
                                - self.ROADSIDE_BARRIER_CLEARANCE_M)
        outer_limit = min(nominal_outer_limit, obstacle_outer_limit)
        if outer_limit <= inner_limit:
            return None
        return outward_sign, inner_limit, outer_limit

    def _roadside_barrier_clearance(self,
                                    waypoint: carla.Waypoint) -> float:
        """Measure roadside clearance after centring in the usable corridor.

        The OpenDRIVE shoulder width does not account for guardrail meshes
        that protrude into the nominal lane.  Horizontal rays catch those
        static meshes before the manoeuvre is authorised.
        """
        obstacle_distance = self._roadside_obstacle_distance(waypoint)
        if obstacle_distance is None:
            return float("inf")
        if math.isinf(obstacle_distance):
            return obstacle_distance
        bounds = self._shoulder_safe_center_bounds(
            waypoint, obstacle_distance)
        if bounds is None:
            return 0.0
        _outward_sign, inner_limit, outer_limit = bounds
        target_outward_offset = (inner_limit + outer_limit) * 0.5
        return (obstacle_distance - target_outward_offset
                - float(self.vehicle.bounding_box.extent.y))

    def _shoulder_track_error_safe(self, lateral_error: float,
                                   desired_offset: float = 0.0) -> bool:
        """Require the vehicle to be within the symmetric centre tolerance."""
        return abs(lateral_error) <= self.SHOULDER_TRACK_TOLERANCE_M

    def _shoulder_corridor_clear(self,
                                 waypoint: carla.Waypoint) -> bool:
        """Validate rail clearance for the complete lateral merge corridor."""
        candidates = [waypoint]
        distance = self.ROADSIDE_SCAN_STEP_M
        while distance <= self.ROADSIDE_SCAN_AHEAD_M:
            ahead = waypoint.next(distance)
            candidate = next((item for item in ahead
                              if item.road_id == waypoint.road_id
                              and item.lane_id == waypoint.lane_id
                              and self._shoulder_is_wide_enough(item)), None)
            if candidate is None:
                return False
            candidates.append(candidate)
            distance += self.ROADSIDE_SCAN_STEP_M
        clearances = [
            self._roadside_barrier_clearance(candidate)
            for candidate in candidates]
        minimum = min(clearances)
        if minimum < self.ROADSIDE_BARRIER_CLEARANCE_M:
            print(
                "MRM SHOULDER REJECTED  roadside_clearance=%.2fm "
                "required=%.2fm lane=%d" % (
                    minimum, self.ROADSIDE_BARRIER_CLEARANCE_M,
                    waypoint.lane_id),
                flush=True)
            return False
        return True

    def _target_lane_blocked(self, target: carla.Waypoint) -> bool:
        ego_transform = self.vehicle.get_transform()
        ego_forward = ego_transform.get_forward_vector()
        ego_location = ego_transform.location
        ego_velocity = self.vehicle.get_velocity()
        ego_forward_speed = (
            ego_velocity.x * ego_forward.x + ego_velocity.y * ego_forward.y)
        actors = list(self.world.get_actors().filter("vehicle.*"))
        actors += list(self.world.get_actors().filter("walker.pedestrian.*"))
        for actor in actors:
            if actor.id == self.vehicle.id:
                continue
            actor_wp = self.map.get_waypoint(
                actor.get_location(), project_to_road=True,
                lane_type=MRM_LANE_TYPES)
            if actor_wp is None:
                continue
            if (actor_wp.road_id != target.road_id
                    or actor_wp.lane_id != target.lane_id):
                continue
            delta = actor.get_location() - ego_location
            longitudinal = delta.x * ego_forward.x + delta.y * ego_forward.y
            is_vehicle = actor.type_id.startswith("vehicle.")
            rear_clearance = -15.0 if is_vehicle else -3.0
            front_clearance = 28.0 if is_vehicle else 18.0
            if rear_clearance < longitudinal < front_clearance:
                return True
            if is_vehicle and longitudinal <= rear_clearance:
                actor_velocity = actor.get_velocity()
                actor_forward_speed = (
                    actor_velocity.x * ego_forward.x
                    + actor_velocity.y * ego_forward.y)
                closing_speed = actor_forward_speed - ego_forward_speed
                if closing_speed > 0.5 and -longitudinal / closing_speed < 5.0:
                    return True
        return False

    @staticmethod
    def _point_to_segment_distance(point: carla.Location,
                                   start: carla.Location,
                                   end: carla.Location) -> float:
        segment_x = end.x - start.x
        segment_y = end.y - start.y
        segment_z = end.z - start.z
        length_squared = (
            segment_x ** 2 + segment_y ** 2 + segment_z ** 2)
        if length_squared <= 1e-6:
            return point.distance(start)
        projection = clamp(
            ((point.x - start.x) * segment_x
             + (point.y - start.y) * segment_y
             + (point.z - start.z) * segment_z) / length_squared,
            0.0, 1.0)
        closest = carla.Location(
            x=start.x + projection * segment_x,
            y=start.y + projection * segment_y,
            z=start.z + projection * segment_z)
        return point.distance(closest)

    def _path_blocked(self, plan: List[Tuple[Any, Any]]) -> bool:
        """Check the swept lane-change centerline for dynamic actors."""
        ego_location = self.vehicle.get_location()
        ego_half_width = float(self.vehicle.bounding_box.extent.y)
        path_locations = [
            path_wp.transform.location for path_wp, _option in plan
            if ego_location.distance(path_wp.transform.location) <= 40.0
        ]
        if not path_locations:
            return False

        actors = list(self.world.get_actors().filter("vehicle.*"))
        actors += list(self.world.get_actors().filter("walker.pedestrian.*"))
        for actor in actors:
            if actor.id == self.vehicle.id:
                continue
            actor_location = actor.get_location()
            if ego_location.distance(actor_location) > 45.0:
                continue
            if actor.type_id.startswith("vehicle."):
                clearance = (
                    ego_half_width + float(actor.bounding_box.extent.y) + 0.75)
            else:
                clearance = ego_half_width + 0.9
            if len(path_locations) == 1:
                distance = actor_location.distance(path_locations[0])
            else:
                distance = min(
                    self._point_to_segment_distance(
                        actor_location, start, end)
                    for start, end in zip(
                        path_locations, path_locations[1:]))
            if distance < clearance:
                return True
        return False

    def _lane_alignment(self, waypoint: carla.Waypoint) -> Tuple[float, float]:
        vehicle_transform = self.vehicle.get_transform()
        delta = vehicle_transform.location - waypoint.transform.location
        right = waypoint.transform.get_right_vector()
        lateral_offset = delta.x * right.x + delta.y * right.y
        heading_error = abs(normalized_angle_deg(
            vehicle_transform.rotation.yaw - waypoint.transform.rotation.yaw))
        return lateral_offset, heading_error

    def _limit_control(self, control: carla.VehicleControl
                       ) -> carla.VehicleControl:
        requested_steer = clamp(
            float(control.steer), -self.MAX_STEER, self.MAX_STEER)
        control.steer = clamp(
            requested_steer,
            self.previous_steer - self.MAX_STEER_STEP,
            self.previous_steer + self.MAX_STEER_STEP)
        self.previous_steer = float(control.steer)
        control.throttle = clamp(
            float(control.throttle), 0.0, self.MAX_THROTTLE)
        control.brake = clamp(float(control.brake), 0.0, 0.30)
        if control.brake > 0.02:
            control.throttle = 0.0
        control.hand_brake = False
        control.reverse = False
        control.manual_gear_shift = False
        return control

    def _lane_center_control(self, waypoint: carla.Waypoint,
                             target_speed_kph: float,
                             brake_floor: float = 0.0,
                             lookahead_m: float = 10.0
                             ) -> carla.VehicleControl:
        ahead = waypoint.next(lookahead_m)
        target_waypoint = ahead[0] if ahead else waypoint
        lateral_controller = self.planner._vehicle_controller._lat_controller
        previous_offset = lateral_controller._offset
        lateral_controller._offset = self._shoulder_target_offset(
            target_waypoint)
        try:
            control = self.planner._vehicle_controller.run_step(
                target_speed_kph, target_waypoint)
        finally:
            lateral_controller._offset = previous_offset
        if target_speed_kph <= 0.0:
            control.throttle = 0.0
        control.brake = max(float(control.brake), brake_floor)
        return self._limit_control(control)

    def _shoulder_run_is_usable(self, target: carla.Waypoint) -> bool:
        """Require a continuous shoulder before and after the stop point."""
        if not self._shoulder_is_wide_enough(target) or target.is_junction:
            return False
        for direction, required_distance in (
                ("previous", self.STOP_TARGET_REAR_CLEARANCE_M),
                ("next", self.STOP_TARGET_FRONT_CLEARANCE_M)):
            distance = 2.0
            while distance <= required_distance:
                waypoints = getattr(target, direction)(distance)
                if not any(
                        self._shoulder_is_wide_enough(candidate)
                        and candidate.road_id == target.road_id
                        and candidate.lane_id == target.lane_id
                        and not candidate.is_junction
                        and self._same_direction(target, candidate)
                        for candidate in waypoints):
                    return False
                distance += 2.0
        return True

    def _shoulder_stop_target_score(self, target: carla.Waypoint
                                    ) -> Optional[float]:
        """Score a legal shoulder stop point; None means dynamically unsafe.

        CARLA actor/map ground truth is deliberately the primary safety gate.
        It is more reliable than a VLM for simulated actor position, lane
        occupancy, and metric clearance. A camera/VLM can later be added only
        as a conservative rejection gate for unmodelled obstacles.
        """
        if not self._shoulder_run_is_usable(target):
            return None
        target_location = target.transform.location
        forward = target.transform.get_forward_vector()
        right = target.transform.get_right_vector()
        lane_half_width = float(target.lane_width) * 0.5
        best_margin = float("inf")
        for actor in self.world.get_actors():
            if actor.id == self.vehicle.id:
                continue
            if not (actor.type_id.startswith("vehicle.")
                    or actor.type_id.startswith("walker.pedestrian.")):
                continue
            delta = actor.get_location() - target_location
            longitudinal = delta.x * forward.x + delta.y * forward.y
            lateral = abs(delta.x * right.x + delta.y * right.y)
            actor_half_width = float(actor.bounding_box.extent.y)
            if lateral > lane_half_width + actor_half_width + 0.4:
                continue
            is_vehicle = actor.type_id.startswith("vehicle.")
            front = (self.STOP_TARGET_FRONT_CLEARANCE_M
                     if is_vehicle else 6.0)
            rear = (self.STOP_TARGET_REAR_CLEARANCE_M
                    if is_vehicle else 3.0)
            if -rear <= longitudinal <= front:
                return None
            if is_vehicle and longitudinal < -rear:
                velocity = actor.get_velocity()
                closing_speed = (velocity.x * forward.x + velocity.y * forward.y)
                if closing_speed > 0.5 and -longitudinal / closing_speed < 7.0:
                    return None
            best_margin = min(best_margin, abs(longitudinal) - (
                front if longitudinal >= 0.0 else rear))
        # Prefer the candidate with the largest nearest-actor longitudinal
        # margin; a completely empty shoulder gets a large finite score.
        return 1000.0 if math.isinf(best_margin) else best_margin

    def _shoulder_stop_target_clear(self, target: carla.Waypoint) -> bool:
        return self._shoulder_stop_target_score(target) is not None

    def _find_shoulder_stop_target(self, waypoint: carla.Waypoint
                                   ) -> Optional[carla.Waypoint]:
        """Select the nearest legal, clear vehicle-safe stopping point."""
        best: Optional[Tuple[float, float, carla.Waypoint]] = None
        distance = self.STOP_TARGET_MIN_AHEAD_M
        while distance <= self.STOP_TARGET_MAX_AHEAD_M:
            for candidate in waypoint.next(distance):
                if (self._is_safe_zone_lane(candidate)
                        and candidate.road_id == waypoint.road_id
                        and self._same_direction(waypoint, candidate)):
                    score = self._shoulder_stop_target_score(candidate)
                    if score is not None:
                        # Route distance is the primary key. Clearance only
                        # breaks ties between equally near safe candidates.
                        ranked = (-distance, score, candidate)
                        if best is None or ranked[:2] > best[:2]:
                            best = ranked
            distance += self.STOP_TARGET_STEP_M
        return best[2] if best is not None else None

    def _approach_stop_target(self, waypoint: carla.Waypoint):
        target = self.stop_target
        if target is None:
            self.phase = "approach"
            return self._continue_current_lane(
                waypoint, "lost shoulder stop target; replanning")
        if not self._shoulder_stop_target_clear(target):
            self.stop_target = None
            self.phase = "approach"
            self.stop_ready = False
            self.stop_stable_ticks = 0
            return self._continue_current_lane(
                waypoint, "shoulder stop target became occupied; replanning",
                target_speed_kph=self.MERGE_TARGET_SPEED_KPH)
        vehicle_location = self.vehicle.get_location()
        delta = target.transform.location - vehicle_location
        forward = waypoint.transform.get_forward_vector()
        longitudinal = delta.x * forward.x + delta.y * forward.y
        if longitudinal < -2.5:
            self.stop_target = None
            self.phase = "approach"
            self.stop_ready = False
            self.stop_stable_ticks = 0
            return self._continue_current_lane(
                waypoint, "passed shoulder stop target; selecting next target",
                target_speed_kph=self.MERGE_TARGET_SPEED_KPH)
        remaining = max(0.0, longitudinal)
        speed_kph = 3.6 * self.vehicle.get_velocity().length()
        target_speed_kph = min(
            self.caution_speed_kph,
            3.6 * math.sqrt(max(0.0, 2.0 * remaining)))
        brake_floor = 0.0
        if remaining <= 1.2:
            target_speed_kph = 0.0
            brake_floor = clamp(0.08 + speed_kph / 100.0, 0.08, 0.30)
        elif remaining <= 3.0:
            # Approach the fixed target with a real creep command. Applying a
            # brake floor here used to cancel all throttle and strand the ego
            # 2-3 m before the selected stop point forever.
            target_speed_kph = min(target_speed_kph, 3.0)
        # Longitudinal deceleration uses the fixed stop target, while lateral
        # control follows the current shoulder centre with a short lookahead.
        # Steering at a waypoint beyond the stop point cuts Town04's curve and
        # leaves the vehicle unnecessarily close to the roadside barrier.
        control = self._lane_center_control(
            waypoint, target_speed_kph=target_speed_kph,
            brake_floor=brake_floor,
            lookahead_m=self.SHOULDER_CENTER_LOOKAHEAD_M)
        if remaining <= self.STOP_COMPLETE_DISTANCE_M:
            control.throttle = 0.0
            control.brake = max(control.brake, 0.20)
            control = self._limit_control(control)
        lateral_offset, heading_error = self._lane_alignment(waypoint)
        desired_offset = self._shoulder_target_offset(waypoint)
        lateral_error = lateral_offset - desired_offset
        pose_safe = (
            self._is_safe_zone_lane(waypoint)
            and -2.0 <= longitudinal <= self.STOP_COMPLETE_DISTANCE_M
            and self._shoulder_track_error_safe(
                lateral_error, desired_offset)
            and heading_error <= self.STOP_HEADING_TOLERANCE_DEG)
        if pose_safe and speed_kph <= 0.55:
            self.stop_stable_ticks += 1
        else:
            self.stop_stable_ticks = 0
        self.stop_ready = self.stop_stable_ticks >= self.STOP_SETTLE_TICKS
        return (control.steer, control.brake, control.throttle, None,
                "shoulder stop target lane %d remaining=%.1fm speed=%.1fkm/h "
                "heading=%.1fdeg lateral_error=%.2fm stable=%d/%d" % (
                    target.lane_id, longitudinal, speed_kph, heading_error,
                    lateral_error, self.stop_stable_ticks,
                    self.STOP_SETTLE_TICKS))

    def _hold_current_lane(self, waypoint: carla.Waypoint,
                           reason: str):
        """Stop temporarily for a real traffic-light or collision hazard."""
        speed_kph = 3.6 * self.vehicle.get_velocity().length()
        brake_floor = clamp(0.10 + speed_kph / 100.0, 0.10, 0.30)
        control = self._lane_center_control(
            waypoint, target_speed_kph=0.0, brake_floor=brake_floor)
        return (control.steer, control.brake, control.throttle, None,
                "holding current lane: %s" % reason)

    def _approach_stop_signal(self, waypoint: carla.Waypoint,
                              state: Any, target: carla.Location,
                              distance_to_line: float):
        """Approach a red/yellow line and hold one metre before it."""
        remaining = max(0.0, distance_to_line - STOP_TARGET_MARGIN_M)
        speed_kph = 3.6 * self.vehicle.get_velocity().length()
        if remaining <= 0.20:
            target_speed_kph = 0.0
            brake_floor = 0.30
        else:
            target_speed_kph = min(
                self.caution_speed_kph,
                3.6 * math.sqrt(max(0.0, 2.0 * 3.5 * remaining)))
            brake_floor = 0.0
        control = self._lane_center_control(
            waypoint, target_speed_kph=target_speed_kph,
            brake_floor=brake_floor,
            lookahead_m=clamp(remaining, 3.0, 8.0))
        if remaining <= 0.20:
            control.throttle = 0.0
            control.brake = max(float(control.brake), 0.30)
            control = self._limit_control(control)
        return (control.steer, control.brake, control.throttle, None,
                "%s traffic light: holding 0.3m before stop line "
                "distance=%.1fm speed=%.1fkm/h" % (
                    str(state), distance_to_line, speed_kph))

    def _continue_current_lane(self, waypoint: carla.Waypoint,
                               reason: str,
                               target_speed_kph: Optional[float] = None,
                               center_guard: bool = False):
        """Keep moving slowly until a safe shoulder transition is available."""
        if target_speed_kph is None:
            target_speed_kph = self.target_speed_kph
        control = self._lane_center_control(
            waypoint, target_speed_kph=target_speed_kph)
        if center_guard:
            # A failed shoulder search must never leave the previous outward
            # steering command active on the outermost Driving lane.
            edge = self._roadside_edge(waypoint)
            lateral_offset, _heading_error = self._lane_alignment(waypoint)
            correction = 0.0
            if edge == "right" and lateral_offset > 0.05:
                correction = -clamp(0.06 + lateral_offset * 0.30,
                                    0.06, 0.16)
            elif edge == "left" and lateral_offset < -0.05:
                correction = clamp(0.06 + abs(lateral_offset) * 0.30,
                                   0.06, 0.16)
            if correction:
                control.steer = clamp(
                    float(control.steer) + correction,
                    -self.MAX_STEER, self.MAX_STEER)
                self.previous_steer = float(control.steer)
            reason = "%s; outer-edge lane-center guard" % reason
        return (control.steer, control.brake, control.throttle, None,
                "seeking shoulder: %s" % reason)

    def reset(self) -> None:
        self.planner.set_global_plan([], stop_waypoint_creation=True,
                                     clean_queue=True)
        self.target_lane_id = None
        self.target_road_id = None
        self.target_lane_type = None
        self.side = None
        self.phase = "approach"
        self.stop_target = None
        self.stop_ready = False
        self.stop_stable_ticks = 0
        self.lane_change_number = 0
        self.settling = True
        self.stable_ticks = 0
        self.previous_steer = clamp(
            float(self.vehicle.get_control().steer), -self.MAX_STEER,
            self.MAX_STEER)
        controller = self.planner._vehicle_controller
        controller.past_steering = self.previous_steer
        controller._lat_controller._e_buffer.clear()
        controller._lon_controller._error_buffer.clear()

    def start_recovery(self) -> None:
        """Prepare a controlled Shoulder-to-Driving handback manoeuvre."""
        self.planner.set_global_plan([], stop_waypoint_creation=True,
                                     clean_queue=True)
        self.target_lane_id = None
        self.target_road_id = None
        self.target_lane_type = None
        self.side = None
        self.phase = "recovery"
        self.stop_target = None
        self.stop_ready = False
        self.stop_stable_ticks = 0
        self.stable_ticks = 0
        self.previous_steer = clamp(
            float(self.vehicle.get_control().steer), -self.MAX_STEER,
            self.MAX_STEER)
        controller = self.planner._vehicle_controller
        controller.past_steering = self.previous_steer
        controller._lat_controller._e_buffer.clear()
        controller._lon_controller._error_buffer.clear()

    def recovery_step(self):
        """Return to a same-direction Driving lane before releasing control."""
        waypoint = self.map.get_waypoint(
            self.vehicle.get_location(), project_to_road=True,
            lane_type=MRM_LANE_TYPES)
        if waypoint is None:
            return (0.0, 1.0, 0.0, None,
                    "recovery: no road waypoint", False)

        if self._is_safe_zone_lane(waypoint):
            side, target = self._inward_driving_lane(waypoint)
            if target is None:
                control = self._lane_center_control(
                    waypoint, target_speed_kph=0.0, brake_floor=0.20)
                return (control.steer, control.brake, control.throttle, None,
                        "recovery: no adjacent driving lane", False)
            if self._target_lane_blocked(target):
                control = self._lane_center_control(
                    waypoint, target_speed_kph=0.0, brake_floor=0.20)
                return (control.steer, control.brake, control.throttle, side,
                        "recovery: driving lane occupied; waiting", False)
            ahead = target.next(18.0)
            target_ahead = next(
                (candidate for candidate in ahead
                 if candidate.lane_type == carla.LaneType.Driving
                 and candidate.road_id == target.road_id
                 and candidate.lane_id == target.lane_id
                 and self._same_direction(target, candidate)), target)
            control = self.planner._vehicle_controller.run_step(
                self.MERGE_TARGET_SPEED_KPH, target_ahead)
            control = self._limit_control(control)
            control.steer = clamp(
                float(control.steer), -self.MERGE_MAX_STEER,
                self.MERGE_MAX_STEER)
            self.previous_steer = float(control.steer)
            self.stable_ticks = 0
            return (control.steer, control.brake, control.throttle, side,
                    "recovery: merging from shoulder to driving lane", False)

        if waypoint.lane_type != carla.LaneType.Driving:
            return (0.0, 0.30, 0.0, None,
                    "recovery: unsupported lane type", False)
        lateral_offset, heading_error = self._lane_alignment(waypoint)
        speed_kph = 3.6 * self.vehicle.get_velocity().length()
        if (abs(lateral_offset) <= 0.30
                and heading_error <= 4.0
                and speed_kph <= self.MERGE_TARGET_SPEED_KPH + 2.0):
            self.stable_ticks += 1
        else:
            self.stable_ticks = 0
        control = self._lane_center_control(
            waypoint, target_speed_kph=self.MERGE_TARGET_SPEED_KPH)
        complete = self.stable_ticks >= self.SETTLE_TICKS
        return (
            control.steer, control.brake, control.throttle, None,
            "recovery: centering driving lane offset=%.2fm heading=%.1fdeg "
            "stable=%d/%d" % (
                lateral_offset, heading_error, self.stable_ticks,
                self.SETTLE_TICKS),
            complete)

    def _shoulder_change_path(self, waypoint: carla.Waypoint,
                              side: str) -> List[Tuple[Any, Any]]:
        """Create a short, one-waypoint transition from Driving to Shoulder.

        CARLA's BasicAgent helper rejects non-Driving target lanes, so the
        final edge into a real shoulder needs an equivalent small plan here.
        """
        plan = [(waypoint, RoadOption.LANEFOLLOW)]
        distance = 0.0
        while distance < self.SHOULDER_CHANGE_SAME_LANE_M:
            next_wps = plan[-1][0].next(1.5)
            if not next_wps:
                return []
            next_wp = next_wps[0]
            distance += next_wp.transform.location.distance(
                plan[-1][0].transform.location)
            plan.append((next_wp, RoadOption.LANEFOLLOW))

        # Use a long diagonal so the vehicle does not overshoot the shoulder
        # centre and scrape roadside barriers on the final lateral transition.
        next_wps = plan[-1][0].next(self.SHOULDER_CHANGE_DISTANCE_M)
        if not next_wps:
            return []
        source = next_wps[0]
        target = (source.get_right_lane() if side == "right"
                  else source.get_left_lane())
        if (target is None or not self._shoulder_is_wide_enough(target)):
            return []
        option = (RoadOption.CHANGELANERIGHT if side == "right"
                  else RoadOption.CHANGELANELEFT)
        plan.append((target, option))

        distance = 0.0
        while distance < 12.0:
            next_wps = plan[-1][0].next(1.5)
            if not next_wps:
                return []
            next_wp = next_wps[0]
            if not self._shoulder_is_wide_enough(next_wp):
                return []
            distance += next_wp.transform.location.distance(
                plan[-1][0].transform.location)
            plan.append((next_wp, RoadOption.LANEFOLLOW))
        return plan

    def _merge_into_shoulder(self, waypoint: carla.Waypoint):
        """Track the adjacent shoulder centre at low speed until stabilised."""
        target = None
        if (self._is_safe_zone_lane(waypoint)
                and waypoint.road_id == self.target_road_id
                and waypoint.lane_id == self.target_lane_id):
            target = waypoint
        elif waypoint.lane_type == carla.LaneType.Driving:
            candidate = (waypoint.get_right_lane() if self.side == "right"
                         else waypoint.get_left_lane())
            if (candidate is not None
                    and candidate.road_id == self.target_road_id
                    and candidate.lane_id == self.target_lane_id
                    and self._shoulder_is_wide_enough(candidate)):
                target = candidate
        if target is None:
            self.target_lane_id = None
            self.target_road_id = None
            self.target_lane_type = None
            self.side = None
            self.phase = "approach"
            self.settling = True
            self.stable_ticks = 0
            return self._continue_current_lane(
                waypoint, "lost wide shoulder during merge; replanning",
                target_speed_kph=self.caution_speed_kph, center_guard=True)

        on_target_lane = (
            self._is_safe_zone_lane(waypoint)
            and waypoint.road_id == self.target_road_id
            and waypoint.lane_id == self.target_lane_id)
        lookahead = (self.SHOULDER_CENTER_LOOKAHEAD_M
                     if on_target_lane else self.MERGE_LOOKAHEAD_M)
        ahead = target.next(lookahead)
        target_ahead = next((candidate for candidate in ahead
                             if candidate.road_id == target.road_id
                             and candidate.lane_id == target.lane_id
                             and self._shoulder_is_wide_enough(candidate)),
                            target)
        lateral_controller = self.planner._vehicle_controller._lat_controller
        previous_offset = lateral_controller._offset
        desired_offset = self._shoulder_target_offset(target_ahead)
        lateral_controller._offset = desired_offset
        try:
            control = self.planner._vehicle_controller.run_step(
                self.MERGE_TARGET_SPEED_KPH, target_ahead)
        finally:
            lateral_controller._offset = previous_offset
        control = self._limit_control(control)
        control.steer = clamp(
            float(control.steer), -self.MERGE_MAX_STEER,
            self.MERGE_MAX_STEER)
        self.previous_steer = float(control.steer)
        control.throttle = min(float(control.throttle), self.MERGE_MAX_THROTTLE)

        lateral_offset, heading_error = self._lane_alignment(target)
        lateral_error = lateral_offset - desired_offset
        tolerance = min(
            self._shoulder_center_tolerance(target),
            self.SHOULDER_TRACK_TOLERANCE_M)
        if (on_target_lane
                and self._shoulder_track_error_safe(
                    lateral_error, desired_offset)
                and heading_error <= 2.0):
            self.stable_ticks += 1
        else:
            self.stable_ticks = 0
        if self.stable_ticks >= self.SETTLE_TICKS:
            reached_lane = self.target_lane_id
            self.target_lane_id = None
            self.target_road_id = None
            self.target_lane_type = None
            self.side = None
            self.phase = "approach"
            self.settling = False
            self.stable_ticks = 0
            print(
                "MRM SHOULDER MERGE COMPLETE  lane=%d offset=%.2fm "
                "target=%.2fm error=%.2fm limit=%.2fm" % (
                    reached_lane, lateral_offset, desired_offset,
                    lateral_error, tolerance),
                flush=True)
        return (
            control.steer, control.brake, control.throttle, self.side,
            "low-speed shoulder merge lane %s offset=%.2fm target=%.2fm "
            "error=%.2fm limit=%.2fm heading=%.1fdeg" % (
                self.target_lane_id if self.target_lane_id is not None
                else "complete", lateral_offset, desired_offset,
                lateral_error, tolerance, heading_error))

    def _start_next_lane_change(self, waypoint: carla.Waypoint,
                                side: str, target: carla.Waypoint) -> bool:
        if self._is_safe_zone_lane(target):
            if not self._shoulder_corridor_clear(target):
                return False
            plan = self._shoulder_change_path(waypoint, side)
        else:
            plan = BasicAgent._generate_lane_change_path(
                waypoint,
                direction=side,
                distance_same_lane=self.DRIVING_LANE_CHANGE_SAME_LANE_M,
                distance_other_lane=self.DRIVING_LANE_CHANGE_OTHER_LANE_M,
                lane_change_distance=self.DRIVING_LANE_CHANGE_DISTANCE_M,
                check=False,
                lane_changes=1,
                step_distance=1.5)
        if not plan:
            return False
        if any(path_wp.is_junction
               or (path_wp.lane_type != carla.LaneType.Driving
                   and not self._is_safe_zone_lane(path_wp))
               for path_wp, _option in plan):
            return False
        if any(not self._same_direction(previous[0], current[0])
               for previous, current in zip(plan, plan[1:])):
            return False

        expected_option = (RoadOption.CHANGELANERIGHT
                           if side == "right"
                           else RoadOption.CHANGELANELEFT)
        change_indexes = [
            index for index, (_path_wp, option) in enumerate(plan)
            if option == expected_option
        ]
        if len(change_indexes) != 1 or change_indexes[0] == 0:
            return False
        change_index = change_indexes[0]
        changed_waypoint = plan[change_index][0]
        selector = (self._outward_shoulder
                    if self._is_safe_zone_lane(target)
                    else self._outward_lane)
        # Validate at the longitudinal station of the changed waypoint. The
        # previous implementation compared it with a waypoint 24-34 m behind;
        # normal OpenDRIVE road-id/lane-id transitions therefore rejected many
        # valid shoulder paths and caused an immediate in-lane stop.
        source_at_target = (
            changed_waypoint.get_left_lane()
            if side == "right" else changed_waypoint.get_right_lane())
        if (source_at_target is None
                or source_at_target.lane_type != carla.LaneType.Driving
                or not self._same_direction(source_at_target,
                                            changed_waypoint)):
            return False
        actual_side, expected_target = selector(source_at_target)
        if (actual_side != side or expected_target is None
                or expected_target.road_id != changed_waypoint.road_id
                or expected_target.lane_id != changed_waypoint.lane_id
                or expected_target.lane_type != changed_waypoint.lane_type):
            return False
        if self._target_lane_blocked(changed_waypoint):
            return False
        if self._path_blocked(plan):
            return False

        if self._is_safe_zone_lane(target):
            self.planner.set_global_plan(
                [], stop_waypoint_creation=True, clean_queue=True)
        else:
            self.planner.set_global_plan(
                plan, stop_waypoint_creation=True, clean_queue=True)
        final_waypoint = plan[-1][0]
        self.target_lane_id = final_waypoint.lane_id
        self.target_road_id = final_waypoint.road_id
        self.target_lane_type = final_waypoint.lane_type
        self.side = side
        if self._is_safe_zone_lane(target):
            self.phase = "shoulder_merge"
        self.settling = False
        self.stable_ticks = 0
        self.lane_change_number += 1
        print(
            "MRM LANE PLAN  step=%d from_lane=%d target_lane=%d type=%s points=%d" % (
                self.lane_change_number, waypoint.lane_id,
                changed_waypoint.lane_id, changed_waypoint.lane_type, len(plan)),
            flush=True)
        return True

    def step(self):
        waypoint = self.map.get_waypoint(
            self.vehicle.get_location(), project_to_road=True,
            lane_type=MRM_LANE_TYPES)
        if waypoint is None:
            return 0.0, 1.0, 0.0, None, "no driving waypoint; braking"

        signal_stop = upcoming_stop_signal(self.world, self.vehicle, waypoint)
        deferred_hazard_reason = None
        if signal_stop is not None:
            state, target, distance_to_line = signal_stop
            if distance_to_line <= self.SIGNAL_POSITIONING_STOP_DISTANCE_M:
                return self._approach_stop_signal(
                    waypoint, state, target, distance_to_line)
            deferred_hazard_reason = "%s traffic light ahead" % str(state)

        hazard_reason = forward_hazard_reason(
            self.world, self.vehicle, waypoint)
        if hazard_reason is not None:
            if hazard_reason == "inside intersection":
                return self._continue_current_lane(
                    waypoint, hazard_reason,
                    target_speed_kph=self.caution_speed_kph)
            if hazard_reason in (
                    "junction ahead", "forward traffic ahead",
                    "red traffic light ahead", "yellow traffic light ahead"):
                deferred_hazard_reason = hazard_reason
            else:
                return self._hold_current_lane(waypoint, hazard_reason)

        # Unit-test doubles and partially initialised planners have no lane
        # change controller. Keep the old conservative fallback for them;
        # the live planner continues below and positions outward first.
        if not hasattr(self, "planner") or self.planner is None:
            return self._continue_current_lane(
                waypoint, deferred_hazard_reason or "hazard ahead",
                target_speed_kph=getattr(
                    self, "target_speed_kph", self.caution_speed_kph))

        if getattr(self, "phase", "approach") == "stop_target":
            return self._approach_stop_target(waypoint)
        if getattr(self, "phase", "approach") == "shoulder_merge":
            return self._merge_into_shoulder(waypoint)

        if not self.planner.done():
            if self._path_blocked(list(self.planner.get_plan())):
                return self._hold_current_lane(
                    waypoint, "lane-change trajectory occupied")
            self._set_planner_speed(self._positioning_speed_kph())
            control = self._limit_control(self.planner.run_step())
            return (control.steer, control.brake, control.throttle, self.side,
                    "following lane-change trajectory to lane %d" %
                    self.target_lane_id)

        if self.target_lane_id is not None:
            reached_target = (
                waypoint.lane_id == self.target_lane_id
                and waypoint.road_id == self.target_road_id
                and waypoint.lane_type == self.target_lane_type)
            expected_lane = self.target_lane_id
            self.target_lane_id = None
            self.target_road_id = None
            self.target_lane_type = None
            self.side = None
            self.settling = True
            self.stable_ticks = 0
            if not reached_target:
                return self._continue_current_lane(
                    waypoint,
                    "lane change incomplete; expected lane %d, current lane %d"
                    % (expected_lane, waypoint.lane_id),
                    target_speed_kph=self.caution_speed_kph)
            print(
                "MRM LANE REACHED  lane=%d; centering before next change" %
                waypoint.lane_id,
                flush=True)

        side, target = self._outward_lane(waypoint)
        if target is None:
            side, target = self._outward_shoulder(waypoint)
        if target is not None:
            lateral_offset, heading_error = self._lane_alignment(waypoint)
            speed_kph = 3.6 * self.vehicle.get_velocity().length()
            if self.settling:
                positioning_speed_kph = self._positioning_speed_kph()
                if (abs(lateral_offset) <= 0.30
                        and heading_error <= 4.0
                        and speed_kph <= positioning_speed_kph + 2.0):
                    self.stable_ticks += 1
                else:
                    self.stable_ticks = 0
                if self.stable_ticks < self.SETTLE_TICKS:
                    control = self._lane_center_control(
                        waypoint, target_speed_kph=positioning_speed_kph)
                    return (
                        control.steer, control.brake, control.throttle, side,
                        "centering lane %d before next lane change "
                        "(offset=%.2fm heading=%.1fdeg speed=%.1fkm/h)" % (
                            waypoint.lane_id, lateral_offset, heading_error,
                            speed_kph))
                self.settling = False

            if self._target_lane_blocked(target):
                return self._continue_current_lane(
                    waypoint, "adjacent lane occupied; waiting for a gap",
                    target_speed_kph=self.caution_speed_kph)
            if not self._start_next_lane_change(waypoint, side, target):
                return self._continue_current_lane(
                    waypoint,
                    "lane-change path unavailable here; replanning ahead",
                    center_guard=True)
            if getattr(self, "phase", "approach") == "shoulder_merge":
                return self._merge_into_shoulder(waypoint)
            self._set_planner_speed(self._positioning_speed_kph())
            control = self._limit_control(self.planner.run_step())
            return (control.steer, control.brake, control.throttle, side,
                    "following lane-change trajectory to lane %d" %
                    target.lane_id)

        if (self._is_safe_zone_lane(waypoint)
                and self.settling):
            lateral_offset, heading_error = self._lane_alignment(waypoint)
            tolerance = min(
                self._shoulder_center_tolerance(waypoint),
                self.SHOULDER_TRACK_TOLERANCE_M)
            desired_offset = self._shoulder_target_offset(waypoint)
            lateral_error = lateral_offset - desired_offset
            speed_kph = 3.6 * self.vehicle.get_velocity().length()
            if (self._shoulder_track_error_safe(
                    lateral_error, desired_offset)
                    and heading_error <= 2.0
                    and speed_kph <= self.caution_speed_kph + 2.0):
                self.stable_ticks += 1
            else:
                self.stable_ticks = 0
            if self.stable_ticks < self.SETTLE_TICKS:
                control = self._lane_center_control(
                    waypoint, target_speed_kph=self.caution_speed_kph)
                return (
                    control.steer, control.brake, control.throttle, None,
                    "centering shoulder lane %d (offset=%.2fm target=%.2fm "
                    "error=%.2fm limit=%.2fm heading=%.1fdeg speed=%.1fkm/h)" % (
                        waypoint.lane_id, lateral_offset, desired_offset,
                        lateral_error, tolerance,
                        heading_error, speed_kph))
            self.settling = False
            print(
                "MRM SHOULDER CENTERED  lane=%d offset=%.2fm target=%.2fm "
                "error=%.2fm limit=%.2fm" % (
                    waypoint.lane_id, lateral_offset, desired_offset,
                    lateral_error, tolerance),
                flush=True)

        # Do not treat an outermost Driving lane as a roadside destination.
        # Continue along it and retry on the next road segment until an actual
        # Shoulder becomes available. Only a real Shoulder is a completed MRM.
        if not self._is_safe_zone_lane(waypoint):
            return self._continue_current_lane(
                waypoint,
                "no adjacent shoulder yet; searching ahead%s" % (
                    "; %s" % deferred_hazard_reason
                    if deferred_hazard_reason else ""),
                target_speed_kph=self._positioning_speed_kph(),
                center_guard=True)

        stop_target = self._find_shoulder_stop_target(waypoint)
        if stop_target is None:
            return self._continue_current_lane(
                waypoint, "no clear shoulder stop target; searching ahead",
                target_speed_kph=self.MERGE_TARGET_SPEED_KPH)
        self.stop_target = stop_target
        self.phase = "stop_target"
        print("MRM STOP TARGET  lane=%d ahead=%.1fm" % (
            stop_target.lane_id, self.vehicle.get_location().distance(
                stop_target.transform.location)), flush=True)
        return self._approach_stop_target(waypoint)


def set_vehicle_autopilot(vehicle: carla.Vehicle, enabled: bool, tm_port: int = 8000) -> bool:
    """Safely toggle CARLA Traffic Manager autopilot to prevent control authority conflicts.

    Probes candidate ports if default tm_port is already bound by an existing TrafficManager.
    """
    candidate_ports = [tm_port]
    for p in [tm_port + 2, tm_port + 1, 8002, 8003, 8004, 8005, 6000]:
        if p not in candidate_ports:
            candidate_ports.append(p)

    for port in candidate_ports:
        try:
            vehicle.set_autopilot(enabled, port)
            print("AUTOPILOT AUTHORITY: vehicle.set_autopilot(%s, tm_port=%d) succeeded." % (enabled, port), flush=True)
            return True
        except Exception as exc:
            if "bind error" in str(exc).lower():
                continue
            break

    try:
        vehicle.set_autopilot(enabled)
        print("AUTOPILOT AUTHORITY: vehicle.set_autopilot(%s) succeeded." % enabled, flush=True)
        return True
    except Exception as exc:
        print("AUTOPILOT AUTHORITY WARNING: Failed to set autopilot (%s): %s" % (enabled, exc), flush=True)
        return False


def write_behavior_override(path: str, role_name: str, active: bool,
                            reason: str, action: str = "brake",
                            steer: float = 0.0, brake: float = 1.0,
                            throttle: float = 0.0,
                            hand_brake: bool = False,
                            speed_limit_factor: float = 1.0,
                            persistent: bool = False) -> None:
    if not path:
        return
    payload = {
        "protocol_version": 1,
        "role_name": role_name,
        "active": bool(active),
        "action": action if active else "release",
        "steer": clamp(steer, -1.0, 1.0) if active else 0.0,
        "brake": clamp(brake, 0.0, 1.0) if active else 0.0,
        "throttle": clamp(throttle, 0.0, 1.0) if active else 0.0,
        "hand_brake": bool(hand_brake) if active else False,
        # Only a terminal hold may outlive the supervisor heartbeat. The sole
        # actuator owner validates this again before accepting it.
        "persistent": bool(persistent and active and action == "hold"),
        "speed_limit_factor": clamp(speed_limit_factor, 0.1, 1.0),
        "reason": reason[:200],
        "updated_at_unix_s": time.time(),
    }
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".safety_override.", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, separators=(",", ":"))
            handle.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def clear_behavior_override(path: str, role_name: str) -> None:
    write_behavior_override(path, role_name, False, "safety supervisor normal")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("CARLA safety supervisor that publishes commands to the "
                     "single actuator owner, run_behavior_autopilot.py"))
    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000,
                        help="CARLA Traffic Manager port for autopilot control")
    parser.add_argument("--role-name", default="town06_ego")
    parser.add_argument(
        "--server-url", default="http://127.0.0.1:8000/driver_state/latest")
    parser.add_argument(
        "--token", default=os.environ.get("LLM_BRIDGE_TOKEN", "").strip())
    parser.add_argument("--poll-hz", type=float, default=5.0)
    parser.add_argument("--request-timeout", type=float, default=1.0)
    parser.add_argument("--max-state-age", type=float, default=3.0)
    parser.add_argument(
        "--exit-after-monitor-idle-sec", type=float, default=0.0,
        help=(
            "after at least one fresh observation, treat this many seconds "
            "without a newer result as the end of a finite DGX dataset; "
            "0 disables local idle completion detection"),
    )
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument(
        "--required-no-response-observations",
        "--required-danger-observations",
        dest="required_no_response_observations",
        type=int,
        default=1,
        help=(
            "consecutive no_visible_response observations required before "
            "starting the minimal-risk manoeuvre"),
    )
    parser.add_argument(
        "--required-active-observations",
        "--required-safe-observations",
        dest="required_active_observations",
        type=int,
        default=4,
        help="consecutive active observations required for optional auto-resume",
    )
    parser.add_argument("--auto-resume", action="store_true",
                        help=(
                            "automatically re-enable normal control after "
                            "consecutive active responses"))
    parser.add_argument("--enable-autopilot-on-start", action="store_true",
                        help="explicitly turn on autopilot when supervisor starts")
    parser.add_argument("--toggle-carla-autopilot", action="store_true",
                        help=(
                            "also call vehicle.set_autopilot() on MRM/resume. "
                            "Leave this off when run_behavior_autopilot.py owns control."))
    parser.add_argument(
        "--behavior-override-file",
        default="/tmp/carla_town04_ego_safety_override.json",
        help=(
            "JSON file read by run_behavior_autopilot.py. While MRM is active "
            "the BehaviorAgent loop yields control to the safety planner."))
    parser.add_argument(
        "--no-warning-display",
        action="store_true",
        help="do not automatically open view_cabin_live.py for reduced response")
    parser.add_argument(
        "--display-script",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "view_cabin_live.py"),
        help="path to the driver-response viewer launched for reduced response")
    parser.add_argument(
        "--display-poll-hz", type=float, default=5.0,
        help="polling rate passed to the automatically launched viewer")
    parser.add_argument(
        "--display-required-reduced-observations", type=int, default=3,
        help="consecutive reduced observations required to open the viewer")
    parser.add_argument(
        "--display-required-active-observations", type=int, default=4,
        help="consecutive active observations required to close the viewer")
    parser.add_argument(
        "--display-window-name", default="Driver Response Warning",
        help="window title passed to the automatically launched viewer")
    parser.add_argument("--overlay-delay", type=float, default=0.01)
    parser.add_argument(
        "--action", choices=("brake", "shoulder"), default="shoulder",
        help="move to the nearest available shoulder, or use brake for current-lane stop")
    parser.add_argument(
        "--mrm-speed-kph", type=float, default=30.0,
        help=(
            "target speed while searching for and changing toward a shoulder "
            "(default: 30 km/h; junction and gap waiting are capped at 12 km/h)"))
    parser.add_argument(
        "--force-mrm", action="store_true",
        help=("bypass DGX/VLM and immediately publish a deterministic "
              "minimal-risk manoeuvre; run carla06auto at the same time"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.poll_hz <= 0.0 or args.request_timeout <= 0.0:
        raise ValueError("poll and timeout values must be positive")
    if not 0.0 <= args.min_confidence <= 1.0:
        raise ValueError("--min-confidence must be in [0, 1]")
    if (args.exit_after_monitor_idle_sec != 0.0
            and args.exit_after_monitor_idle_sec < 5.0):
        raise ValueError(
            "--exit-after-monitor-idle-sec must be 0 or at least 5 seconds")
    if args.required_no_response_observations < 1:
        raise ValueError(
            "--required-no-response-observations must be positive")
    if args.required_active_observations < 1:
        raise ValueError("--required-active-observations must be positive")
    if args.display_poll_hz <= 0.0:
        raise ValueError("--display-poll-hz must be positive")
    if args.display_required_reduced_observations < 1:
        raise ValueError(
            "--display-required-reduced-observations must be positive")
    if args.display_required_active_observations < 1:
        raise ValueError(
            "--display-required-active-observations must be positive")
    if not 5.0 <= args.mrm_speed_kph <= 30.0:
        raise ValueError("--mrm-speed-kph must be between 5 and 30")

    _, world = wait_for_world(args.carla_host, args.carla_port)
    vehicle = wait_for_vehicle(world, args.role_name)
    shoulder_planner = ShoulderLanePlanner(
        world, vehicle, target_speed_kph=args.mrm_speed_kph)
    poller = DriverStatePoller(
        args.server_url, args.token, args.request_timeout, 1.0 / args.poll_hz)
    warning_display = WarningDisplayManager(
        args.display_script, args.server_url, args.token,
        args.display_poll_hz, args.display_window_name,
        enabled=not args.no_warning_display)
    display_policy = WarningDisplayPolicy(
        args.display_required_reduced_observations,
        args.display_required_active_observations,
        args.required_no_response_observations)
    # --force-mrm bypasses only the trigger. Keep polling when auto-resume is
    # requested so the deterministic test can also verify the complete
    # Shoulder -> Driving -> BehaviorAgent recovery path.
    poller_started = not args.force_mrm or args.auto_resume
    if poller_started:
        poller.start()

    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    if args.enable_autopilot_on_start:
        set_vehicle_autopilot(vehicle, True, args.tm_port)
    clear_behavior_override(args.behavior_override_file, args.role_name)

    print(
        "SAFETY SUPERVISOR READY actor_id=%d action=%s state_url=%s" % (
            vehicle.id, args.action, args.server_url),
        flush=True)
    print(
        "driver_response policy: reduced=warning only, "
        "no_visible_response=%d consecutive observations before %s, "
        "active=normal."
        % (args.required_no_response_observations, args.action),
        flush=True)
    print(
        "warning display: %s (%s), open after reduced=%d, "
        "close after active=%d" % (
            "automatic" if not args.no_warning_display else "disabled",
            args.display_script,
            args.display_required_reduced_observations,
            args.display_required_active_observations),
        flush=True)

    last_observation_id = -1
    no_response_observations = 0
    active_observations = 0
    resume_requested = False
    stopped_since = None
    mode = "normal"
    hazard_active = False
    last_status = 0.0
    low_speed_since = None
    last_response: Optional[str] = None
    completion_announced = False
    monitor_session_started = False
    release_behavior_override_on_exit = False
    if args.force_mrm:
        mode = "mrm"
        shoulder_planner.reset()
        set_hazard_lights(vehicle, True)
        hazard_active = True
        print(
            "FORCE MRM: DGX/VLM bypassed; publishing deterministic shoulder "
            "control to the BehaviorAgent actuator.", flush=True)
    try:
        while not stop_requested and vehicle.is_alive:
            world.wait_for_tick(10.0)
            if args.overlay_delay:
                time.sleep(args.overlay_delay)
            now = time.monotonic()
            velocity = vehicle.get_velocity()
            speed_mps = math.sqrt(
                velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
            state_response = poller.latest() if poller_started else None
            action_reason = "monitoring"
            result_age = 999.0

            if state_response is not None:
                result_age = float(state_response.get("result_age_s", 999.0))
                state = extract_driver_state(state_response)
                observation_id = int(state_response.get("observation_id", -1))
                if result_age <= args.max_state_age and observation_id > last_observation_id:
                    monitor_session_started = True
                    last_observation_id = observation_id
                    last_response = driver_response(
                        state, args.min_confidence)
                    open_display, close_display = display_policy.observe(
                        last_response)
                    if open_display:
                        warning_display.ensure_running()
                    elif close_display:
                        warning_display.stop(
                            "%d consecutive active responses" %
                            args.display_required_active_observations)

                    if last_response == "no_visible_response":
                        no_response_observations += 1
                        active_observations = 0
                        resume_requested = False
                        if not hazard_active:
                            set_hazard_lights(vehicle, True)
                            hazard_active = True
                        action_reason = (
                            "no visible voluntary driver response (%d/%d)"
                            % (no_response_observations,
                               args.required_no_response_observations))
                    elif last_response == "reduced":
                        no_response_observations = 0
                        active_observations = 0
                        resume_requested = False
                        if mode == "normal":
                            clear_behavior_override(
                                args.behavior_override_file, args.role_name)
                        if mode == "normal" and hazard_active:
                            set_hazard_lights(vehicle, False)
                            hazard_active = False
                        action_reason = (
                            "reduced driver response (%d/%d): warning display "
                            "%s; normal driving retained" % (
                                display_policy.reduced_observations,
                                args.display_required_reduced_observations,
                                "and warning display active"
                                if open_display else
                                "active; waiting to open warning display"))
                    elif last_response == "active":
                        no_response_observations = 0
                        active_observations += 1
                        if mode == "normal":
                            clear_behavior_override(
                                args.behavior_override_file, args.role_name)
                        if mode == "normal" and hazard_active:
                            set_hazard_lights(vehicle, False)
                            hazard_active = False
                        action_reason = "active driver response"
                    else:
                        # Missing, invalid, low-confidence, and obsolete labels
                        # are not driver-response states. They neither start an
                        # MRM nor prove recovery. Keep an existing warning until
                        # an explicit active result arrives.
                        no_response_observations = 0
                        active_observations = 0
                        resume_requested = False
                        if mode == "normal":
                            clear_behavior_override(
                                args.behavior_override_file, args.role_name)
                        action_reason = (
                            "driver response unavailable: no automatic control")

                    # Normal -> MRM is driven only by an explicit, consecutive
                    # no_visible_response result. risk_level is intentionally
                    # ignored so distraction cannot trigger a shoulder stop.
                    if (mode in ("normal", "recovering")
                            and no_response_observations >=
                            args.required_no_response_observations):
                        mode = "mrm"
                        shoulder_planner.reset()
                        resume_requested = False
                        stopped_since = None
                        reason_msg = str(state.get("reason", "")).strip() or (
                            "driver_response=no_visible_response; no clear "
                            "voluntary response detected")
                        action_reason = reason_msg
                        print(
                            "\n!!! NO VISIBLE DRIVER RESPONSE !!!\n"
                            "Starting cooperative minimal-risk shoulder stop.\n"
                            "Reason: %s (no_response_observations=%d)\n" % (
                                reason_msg, no_response_observations),
                            flush=True)
                        if not hazard_active:
                            set_hazard_lights(vehicle, True)
                            hazard_active = True
                        if args.toggle_carla_autopilot:
                            set_vehicle_autopilot(
                                vehicle, False, args.tm_port)

                    # A recovered driver cannot interrupt an in-progress MRM.
                    # Queue active observations and release them only after a
                    # verified shoulder stop has completed.
                    elif (args.auto_resume and mode == "mrm"
                          and active_observations >=
                          args.required_active_observations):
                        if not resume_requested:
                            print(
                                "ACTIVE RESPONSE QUEUED: completing the "
                                "shoulder stop before recovery.", flush=True)
                        resume_requested = True
                        action_reason = (
                            "active driver response queued until shoulder "
                            "stop completes")
                    elif (args.auto_resume and mode == "stopped"
                          and active_observations >=
                          args.required_active_observations):
                        mode = "recovering"
                        resume_requested = False
                        stopped_since = None
                        print(
                            "\n--- ACTIVE RESPONSE RECOVERY ---\n"
                            "Driver response is active (%d consecutive "
                            "observations). Returning to a Driving lane before "
                            "normal-control handback.\n"
                            % active_observations,
                            flush=True)
                        no_response_observations = 0
                        active_observations = 0
                        low_speed_since = None
                        shoulder_planner.start_recovery()

                completed_observation_id = int(
                    state_response.get("completed_observation_id", -1))
                dataset_complete = bool(
                    state_response.get("monitoring_complete", False))
                completion_ready = (
                    dataset_complete
                    and completed_observation_id >= 0
                    # The final result can arrive between two supervisor
                    # polls.  The bridge's completion marker is accepted only
                    # for its latest stored result, so do not require that the
                    # local debounce counter saw that observation first.
                    and max(last_observation_id, observation_id)
                    >= completed_observation_id)
                if completion_ready and not completion_announced:
                    completion_announced = True
                    release_behavior_override_on_exit = True
                    print(
                        "MODEL MONITOR COMPLETE: releasing safety override "
                        "and stopping supervisor immediately.", flush=True)
                    stop_requested = True

                idle_completion_ready = (
                    args.exit_after_monitor_idle_sec > 0.0
                    and monitor_session_started
                    and result_age >= args.exit_after_monitor_idle_sec)
                if idle_completion_ready and not completion_announced:
                    completion_announced = True
                    release_behavior_override_on_exit = True
                    print(
                        "MODEL MONITOR IDLE: no newer DGX observation for "
                        "%.1fs; releasing safety override and stopping "
                        "supervisor immediately." % result_age, flush=True)
                    stop_requested = True

            if mode == "mrm":
                if args.action == "shoulder":
                    steer, brake, throttle, side, action_reason = (
                        shoulder_planner.step())
                    if side is None:
                        # Waiting, moving straight, and final shoulder braking
                        # use hazards. A single blinker is reserved for an
                        # active or imminent lateral transition.
                        set_hazard_lights(vehicle, True)
                    else:
                        set_turn_signal(vehicle, side)
                    write_behavior_override(
                        args.behavior_override_file, args.role_name, True,
                        action_reason, action="shoulder", steer=steer,
                        brake=brake, throttle=throttle)
                else:
                    write_behavior_override(
                        args.behavior_override_file, args.role_name, True,
                        action_reason, action="brake", steer=0.0, brake=1.0)
                    action_reason = "minimal-risk brake stop"
                speed = vehicle.get_velocity()
                speed_mps = math.sqrt(
                    speed.x ** 2 + speed.y ** 2 + speed.z ** 2)
                if (args.action == "brake" and speed_mps < 0.15) or (
                        args.action == "shoulder"
                        and shoulder_planner.stop_ready
                        and speed_mps < 0.15):
                    mode = "stopped"
                    stopped_since = now
            elif mode == "stopped":
                action_reason = (
                    "stopped safely after no visible driver response; "
                    "holding brake")
                write_behavior_override(
                    args.behavior_override_file, args.role_name, True,
                    action_reason, action="hold", steer=0.0, brake=1.0,
                    throttle=0.0, hand_brake=True)
                # Once stopped, replace the lane-change indicator with
                # hazard lights on both sides.
                set_hazard_lights(vehicle, True)
                if (args.auto_resume and resume_requested
                        and stopped_since is not None
                        and now - stopped_since >= MRM_MIN_STOP_HOLD_SEC):
                    mode = "recovering"
                    resume_requested = False
                    stopped_since = None
                    active_observations = 0
                    low_speed_since = None
                    shoulder_planner.start_recovery()
                    print(
                        "\n--- QUEUED ACTIVE RESPONSE RECOVERY ---\n"
                        "Verified shoulder stop held; returning to a Driving "
                        "lane before normal-control handback.\n",
                        flush=True)
            elif mode == "recovering":
                (steer, brake, throttle, side, action_reason,
                 recovery_complete) = shoulder_planner.recovery_step()
                if side is None:
                    set_hazard_lights(vehicle, True)
                else:
                    set_turn_signal(vehicle, side)
                write_behavior_override(
                    args.behavior_override_file, args.role_name, True,
                    action_reason, action="recovery", steer=steer,
                    brake=brake, throttle=throttle)
                if recovery_complete:
                    print(
                        "RECOVERY COMPLETE: Driving lane aligned; releasing "
                        "to BehaviorAgent.", flush=True)
                    clear_behavior_override(
                        args.behavior_override_file, args.role_name)
                    set_hazard_lights(vehicle, False)
                    hazard_active = False
                    mode = "normal"
                    no_response_observations = 0
                    active_observations = 0
                    resume_requested = False
                    stopped_since = None
                    shoulder_planner.reset()
                    if args.toggle_carla_autopilot:
                        set_vehicle_autopilot(vehicle, True, args.tm_port)
                    if completion_announced:
                        stop_requested = True

            if now - last_status >= 2.0:
                print(
                    "SAFETY mode=%s response=%s no_response_obs=%d "
                    "active_obs=%d state_age=%.2fs speed=%.2fm/s "
                    "display_reduced_obs=%d display_active_obs=%d "
                    "reason=%s display=%s poll=%s" % (
                        mode, last_response or "unavailable",
                        no_response_observations,
                        active_observations, result_age,
                        speed_mps if 'speed_mps' in locals() else -1.0,
                        display_policy.reduced_observations,
                        display_policy.active_observations,
                        action_reason,
                        warning_display.status(),
                        poller.status() or "ok"),
                    flush=True)
                last_status = now
    finally:
        warning_display.stop("supervisor shutdown")
        if poller_started:
            poller.stop()
        if release_behavior_override_on_exit:
            print(
                "Model monitor ended: releasing control to BehaviorAgent.",
                flush=True)
            set_hazard_lights(vehicle, False)
            clear_behavior_override(args.behavior_override_file, args.role_name)
            if args.toggle_carla_autopilot:
                set_vehicle_autopilot(vehicle, True, args.tm_port)
        elif mode in ("mrm", "stopped", "recovering"):
            print("Safety supervisor exiting while stopped/in MRM. Holding parking brake.", flush=True)
            write_behavior_override(
                args.behavior_override_file, args.role_name, True,
                "safety supervisor exiting while stopped/in MRM",
                action="hold", steer=0.0, brake=1.0, throttle=0.0,
                hand_brake=True, persistent=True)
        else:
            set_hazard_lights(vehicle, False)
            clear_behavior_override(args.behavior_override_file, args.role_name)
        print("Safety supervisor stopped.", flush=True)


if __name__ == "__main__":
    main()
