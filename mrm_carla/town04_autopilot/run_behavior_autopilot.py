#!/usr/bin/env python3
"""Sole actuator owner for Town04 normal driving and safety overrides."""

import argparse
import glob
import json
import os
import random
import signal
import sys
import time

CARLA_PYTHONAPI_DIR = "/home/tmo/carla/PythonAPI"
sys.path.insert(0, os.path.join(CARLA_PYTHONAPI_DIR, "carla"))
egg_matches = glob.glob(os.path.join(
    CARLA_PYTHONAPI_DIR, "carla/dist/carla-*%d.%d-linux-x86_64.egg" %
    (sys.version_info.major, sys.version_info.minor)))
if egg_matches:
    sys.path.insert(0, egg_matches[0])

import carla  # noqa: E402
from agents.navigation.behavior_agent import BehaviorAgent  # noqa: E402
from agents.navigation.local_planner import RoadOption  # noqa: E402
from agents.tools.misc import get_trafficlight_trigger_location, is_within_distance  # noqa: E402


STOP_TARGET_MARGIN_M = 0.3
SIGNAL_CONTROL_DISTANCE_M = 6.0
ROUTE_STALL_TIMEOUT_SEC = 5.0
LANE_CHANGE_MIN_INTERVAL_SEC = 20.0
LANE_CHANGE_MAX_INTERVAL_SEC = 45.0
LANE_CHANGE_SIGNAL_DURATION_SEC = 5.0
_CROSSWALK_CACHE = {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CARLA 0.9.15 stock BehaviorAgent on town06_ego")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--role-name", default="town06_ego")
    parser.add_argument("--behavior", choices=("cautious", "normal", "aggressive"), default="normal")
    parser.add_argument("--destination-distance", type=float, default=350.0)
    parser.add_argument("--speed-increase", type=float, default=5.0,
                        help="Extra normal-mode cruise speed in km/h (default: 5)")
    parser.add_argument("--max-steering", type=float, default=0.65,
                        help="Maximum steering command; tight Town04 ramps need up to 0.65")
    parser.add_argument("--steer-rate", type=float, default=0.09,
                        help="Maximum steering-command change per tick")
    parser.add_argument(
        "--junction-speed-kph", type=float, default=30.0,
        help="speed cap for junctions and ramps (default: 30 km/h)")
    parser.add_argument("--no-follow-camera", action="store_true",
                        help="Do not move CARLA's spectator camera behind the ego vehicle")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--behavior-override-file",
        default="/tmp/carla_town04_ego_safety_override.json",
        help="MRM supervisor control-ownership handoff JSON file")
    parser.add_argument(
        "--override-max-age-sec", type=float, default=1.0,
        help="ignore stale safety commands older than this many seconds")
    return parser.parse_args()


def read_safety_override(path, role_name, max_age_sec):
    """Return a validated supervisor command, or None for normal driving."""
    try:
        with open(path, "r") as handle:
            command = json.load(handle)
        updated_at = float(command.get("updated_at_unix_s", 0.0))
        if (int(command.get("protocol_version", -1)) != 1
                or not command.get("active")
                or command.get("role_name") != role_name):
            return None
        action = str(command.get("action", ""))
        persistent = bool(command.get("persistent", False))
        if persistent:
            # A persistent command is accepted only as a strict terminal park
            # hold. Steering/throttle commands always require a fresh heartbeat.
            if (action != "hold"
                    or float(command.get("throttle", 0.0)) != 0.0
                    or abs(float(command.get("steer", 0.0))) > 0.001
                    or float(command.get("brake", 0.0)) < 0.99
                    or not bool(command.get("hand_brake", False))):
                return None
        elif time.time() - updated_at > max_age_sec:
            return None
        return carla.VehicleControl(
            throttle=clamp(float(command.get("throttle", 0.0)), 0.0, 1.0),
            steer=clamp(float(command.get("steer", 0.0)), -1.0, 1.0),
            brake=clamp(float(command.get("brake", 0.0)), 0.0, 1.0),
            hand_brake=bool(command.get("hand_brake", False)),
            reverse=False,
            manual_gear_shift=False)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def get_ego(world, role_name):
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.attributes.get("role_name") == role_name:
            return actor
    return None


def choose_destination(world, vehicle, minimum_distance):
    """Choose a deterministic point ahead instead of a random U-turn route."""
    carla_map = world.get_map()
    waypoint = carla_map.get_waypoint(
        vehicle.get_location(), project_to_road=True,
        lane_type=carla.LaneType.Driving)
    if waypoint is not None:
        travelled = 0.0
        while travelled < minimum_distance:
            candidates = waypoint.next(min(2.0, minimum_distance - travelled))
            if not candidates:
                break
            yaw = waypoint.transform.rotation.yaw
            waypoint = min(
                candidates,
                key=lambda item: abs(
                    (item.transform.rotation.yaw - yaw + 180.0) % 360.0
                    - 180.0))
            travelled += 2.0
        if travelled >= min(20.0, minimum_distance):
            return waypoint.transform.location

    current = vehicle.get_location()
    points = list(carla_map.get_spawn_points())
    points.sort(key=lambda point: point.location.distance(current), reverse=True)
    if not points:
        raise RuntimeError("Town04_Opt has no destination points")
    return points[0].location


def update_follow_camera(world, vehicle):
    """Place CARLA's Unreal spectator just behind and above the ego vehicle."""
    vehicle_transform = vehicle.get_transform()
    camera_location = vehicle_transform.transform(carla.Location(x=-8.0, z=4.0))
    camera_rotation = carla.Rotation(
        pitch=-15.0, yaw=vehicle_transform.rotation.yaw, roll=0.0)
    world.get_spectator().set_transform(
        carla.Transform(camera_location, camera_rotation))


def clamp(value, low, high):
    return max(low, min(high, value))


def smooth_steering(control, previous_steer, max_delta):
    """Avoid abrupt full-lock steering that can cause a loop at junctions."""
    control.steer = clamp(control.steer, previous_steer - max_delta,
                          previous_steer + max_delta)
    return control


def upcoming_stop_light(agent, max_distance):
    """Return the nearest red/yellow signal and its actual stop-line target."""
    ego_transform = agent._vehicle.get_transform()
    ego_location = ego_transform.location
    ego_waypoint = agent._map.get_waypoint(
        ego_location, project_to_road=True, lane_type=carla.LaneType.Driving)
    if ego_waypoint is None:
        return None, None, None
    stop_states = (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow)
    ego_forward = ego_waypoint.transform.get_forward_vector()
    ego_right = ego_transform.get_right_vector()
    lane_width = max(2.0, float(ego_waypoint.lane_width))
    route_lanes = {(ego_waypoint.road_id, ego_waypoint.lane_id)}
    travelled = 0.0
    previous = ego_location
    for route_waypoint, _option in list(agent._local_planner.get_plan()):
        travelled += previous.distance(route_waypoint.transform.location)
        if travelled > max_distance:
            break
        route_lanes.add((route_waypoint.road_id, route_waypoint.lane_id))
        previous = route_waypoint.transform.location

    affected_light = agent._vehicle.get_traffic_light()
    lights = list(agent._world.get_actors().filter("*traffic_light*"))
    if affected_light is not None and affected_light not in lights:
        lights.append(affected_light)

    candidates = []
    for light in lights:
        try:
            state = (light.get_state()
                     if hasattr(light, "get_state") else light.state)
        except (RuntimeError, AttributeError, TypeError):
            continue
        if state not in stop_states:
            continue

        # CARLA's stop waypoints identify the actual line for this lane. The
        # pole/trigger transform can belong to a neighbouring road section.
        stop_waypoints = []
        try:
            stop_waypoints = list(light.get_stop_waypoints())
        except (AttributeError, RuntimeError, TypeError):
            pass
        stop_candidates = []
        for stop_waypoint in stop_waypoints:
            stop_location = stop_waypoint.transform.location
            stop_forward = stop_waypoint.transform.get_forward_vector()
            if (ego_forward.x * stop_forward.x
                    + ego_forward.y * stop_forward.y
                    + ego_forward.z * stop_forward.z) < 0:
                continue
            delta = stop_location - ego_location
            longitudinal = (delta.x * ego_forward.x
                            + delta.y * ego_forward.y)
            lateral = abs(delta.x * ego_right.x + delta.y * ego_right.y)
            if (longitudinal < -1.0 or longitudinal > max_distance
                    or lateral > lane_width * 1.75):
                continue
            lane_matches_route = (
                stop_waypoint.road_id, stop_waypoint.lane_id) in route_lanes
            current_lane_signal = (
                stop_waypoint.road_id == ego_waypoint.road_id
                and stop_waypoint.lane_id == ego_waypoint.lane_id)
            if not (current_lane_signal or lane_matches_route
                    or light is affected_light):
                continue
            stop_candidates.append((longitudinal, stop_waypoint))

        trigger_location = None
        trigger_waypoint = None
        trigger_longitudinal = None
        if stop_candidates:
            trigger_longitudinal, trigger_waypoint = min(
                stop_candidates, key=lambda item: item[0])
            trigger_location = trigger_waypoint.transform.location
        else:
            # Fallback for custom maps that do not expose stop waypoints.
            try:
                trigger_location = get_trafficlight_trigger_location(light)
                trigger_waypoint = agent._map.get_waypoint(
                    trigger_location, project_to_road=True,
                    lane_type=carla.LaneType.Driving)
            except (RuntimeError, TypeError, AttributeError):
                trigger_waypoint = None
            if trigger_waypoint is None:
                trigger_location = lane_end_location(ego_waypoint)
                trigger_waypoint = agent._map.get_waypoint(
                    trigger_location, project_to_road=True,
                    lane_type=carla.LaneType.Driving)
            if trigger_waypoint is None:
                continue
            light_forward = trigger_waypoint.transform.get_forward_vector()
            if (ego_forward.x * light_forward.x
                    + ego_forward.y * light_forward.y
                    + ego_forward.z * light_forward.z) < 0:
                continue
            trigger_delta = trigger_location - ego_location
            trigger_longitudinal = (
                trigger_delta.x * ego_forward.x
                + trigger_delta.y * ego_forward.y)
            if (trigger_longitudinal < -1.0
                    or trigger_longitudinal > max_distance):
                continue
            trigger_lateral = abs(
                trigger_delta.x * ego_right.x
                + trigger_delta.y * ego_right.y)
            if trigger_lateral > lane_width * 1.75:
                continue
            lane_matches_route = (
                trigger_waypoint.road_id, trigger_waypoint.lane_id
                ) in route_lanes
            current_lane_signal = (
                trigger_waypoint.road_id == ego_waypoint.road_id
                and trigger_waypoint.lane_id == ego_waypoint.lane_id)
            if not (current_lane_signal or lane_matches_route
                    or light is affected_light):
                continue
        target_location = crosswalk_target(
            agent._world, agent._map, ego_location, ego_forward, ego_right,
            lane_width, route_lanes, max_distance,
            before_longitudinal=trigger_longitudinal)
        if target_location is None:
            target_location = trigger_location
        target_delta = target_location - ego_location
        longitudinal = (target_delta.x * ego_forward.x
                        + target_delta.y * ego_forward.y)
        target_transform = carla.Transform(
            target_location, trigger_waypoint.transform.rotation)
        if longitudinal < -1.0 or not is_within_distance(
                target_transform, ego_transform, max_distance, [0, 120]):
            continue
        candidates.append((longitudinal, light, target_location))

    if not candidates:
        return None, None, None
    longitudinal, light, trigger_location = min(
        candidates, key=lambda item: item[0])
    # Use along-lane distance, not Euclidean distance, for the braking gate.
    return light, longitudinal, trigger_location


def vehicle_speed_mps(vehicle):
    velocity = vehicle.get_velocity()
    return (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5


def traffic_light_detection_distance(vehicle):
    """Return a speed-dependent range with braking and reaction margin."""
    speed = vehicle_speed_mps(vehicle)
    front_offset = float(vehicle.bounding_box.extent.x) + 0.60
    braking_distance = speed * speed / (2.0 * 3.5)
    return clamp(braking_distance + front_offset + 4.0, 20.0, 60.0)


def signal_stop_is_due(distance_to_line):
    """Use the long lookahead to latch a signal, not to brake early."""
    return float(distance_to_line) <= SIGNAL_CONTROL_DISTANCE_M


def distance_to_stop_line(vehicle, stop_line_location):
    """Measure remaining distance along the vehicle's current heading."""
    if stop_line_location is None:
        return 0.0
    delta = stop_line_location - vehicle.get_location()
    forward = vehicle.get_transform().get_forward_vector()
    return max(0.0, delta.x * forward.x + delta.y * forward.y)


def stop_at_line(control, vehicle, distance_to_line, stop_buffer=None):
    """Stop the vehicle reference point just before the selected target.

    This is a closed-loop approach: braking is computed from the remaining
    distance, and a vehicle that settles early is allowed to creep forward.
    """
    if stop_buffer is None:
        stop_buffer = STOP_TARGET_MARGIN_M
    speed = vehicle_speed_mps(vehicle)
    remaining = max(0.0, distance_to_line - stop_buffer)
    control.throttle = 0.0
    control.brake = 0.0
    control.hand_brake = False
    # Never wait at a signal with steering lock already wound in.  That was
    # released as an immediate hard turn on the first green-light tick.
    if distance_to_line <= STOP_TARGET_MARGIN_M or speed < 2.0:
        control.steer = 0.0
    else:
        control.steer = clamp(float(control.steer), -0.10, 0.10)
    if remaining <= 0.15:
        control.throttle = 0.0
        control.brake = 1.0
    elif speed < 0.25:
        # Recover from an early stop instead of remaining several metres back.
        control.throttle = 0.10 if remaining > 0.45 else 0.05
        control.brake = 0.0
    else:
        required_decel = speed * speed / (2.0 * max(remaining, 0.20))
        # Fail safe when the remaining time-to-line is short.  Early braking
        # is recoverable by the creep branch; crossing a red light is not.
        if remaining / max(speed, 0.1) < 0.9:
            control.brake = 1.0
        else:
            control.brake = clamp(required_decel / 9.0, 0.0, 1.0)
        if control.brake < 0.03:
            control.brake = 0.0
            control.throttle = min(float(control.throttle), 0.08)
    return control


def hold_at_signal():
    return carla.VehicleControl(
        throttle=0.0, steer=0.0, brake=1.0, hand_brake=False,
        reverse=False, manual_gear_shift=False)


def route_near_junction(agent, max_distance=25.0):
    """Check the actual planned route, rather than arbitrary map branches."""
    location = agent._vehicle.get_location()
    travelled = 0.0
    previous = location
    for waypoint, option in list(agent._local_planner.get_plan()):
        travelled += previous.distance(waypoint.transform.location)
        if travelled > max_distance:
            return False
        if (waypoint.is_junction
                or option in (RoadOption.LEFT, RoadOption.RIGHT)):
            return True
        previous = waypoint.transform.location
    return False


def vehicle_in_junction(carla_map, vehicle):
    """Return true only when the ego vehicle is inside a junction polygon."""
    waypoint = carla_map.get_waypoint(
        vehicle.get_location(), project_to_road=True,
        lane_type=carla.LaneType.Driving)
    return waypoint is not None and bool(waypoint.is_junction)


def reset_agent_controller(agent, steer=0.0):
    """Discard PID history accumulated before a long traffic-light stop."""
    controller = agent._local_planner._vehicle_controller
    controller.past_steering = float(steer)
    controller._lat_controller._e_buffer.clear()
    controller._lon_controller._error_buffer.clear()


def cap_agent_cruise_speed(agent, speed_cap_kph):
    """Prevent speed-limit noise from changing the main-road cruise target."""
    try:
        speed_cap_kph = float(speed_cap_kph)
    except (TypeError, ValueError):
        return
    if speed_cap_kph <= 0.0:
        return
    original_update_information = agent._update_information

    def update_information_with_cruise_cap():
        original_update_information()
        if agent._speed_limit > speed_cap_kph:
            agent._speed_limit = speed_cap_kph

    agent._update_information = update_information_with_cruise_cap


def lane_change_candidate(waypoint, direction):
    """Return a same-direction driving lane permitted by its lane marking."""
    if waypoint is None or waypoint.is_junction:
        return None
    if direction == "left":
        marking = waypoint.left_lane_marking
        candidate = waypoint.get_left_lane()
        allowed = marking.lane_change in (
            carla.LaneChange.Left, carla.LaneChange.Both)
    else:
        marking = waypoint.right_lane_marking
        candidate = waypoint.get_right_lane()
        allowed = marking.lane_change in (
            carla.LaneChange.Right, carla.LaneChange.Both)
    if not allowed or candidate is None:
        return None
    if (candidate.lane_type != carla.LaneType.Driving
            or candidate.road_id != waypoint.road_id
            or candidate.lane_id * waypoint.lane_id <= 0):
        return None
    source_forward = waypoint.transform.get_forward_vector()
    candidate_forward = candidate.transform.get_forward_vector()
    if (source_forward.x * candidate_forward.x
            + source_forward.y * candidate_forward.y
            + source_forward.z * candidate_forward.z) < 0.85:
        return None
    return candidate


def lane_is_clear(world, vehicle, waypoint, radius=12.0):
    """Return false when a vehicle occupies the target lane near the ego car."""
    centre = waypoint.transform.location
    forward = waypoint.transform.get_forward_vector()
    right = waypoint.transform.get_right_vector()
    lane_width = max(2.0, float(waypoint.lane_width))
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.id == vehicle.id:
            continue
        delta = actor.get_location() - centre
        longitudinal = abs(delta.x * forward.x + delta.y * forward.y)
        lateral = abs(delta.x * right.x + delta.y * right.y)
        if longitudinal <= radius and lateral <= lane_width * 0.65:
            return False
    return True


def request_lane_change(agent, world, vehicle):
    """Request one safe occasional lane change, returning its direction."""
    waypoint = world.get_map().get_waypoint(
        vehicle.get_location(), project_to_road=True,
        lane_type=carla.LaneType.Driving)
    if waypoint is None or waypoint.is_junction:
        return None
    directions = ["left", "right"]
    random.shuffle(directions)
    for direction in directions:
        candidate = lane_change_candidate(waypoint, direction)
        if candidate is None or not lane_is_clear(world, vehicle, candidate):
            continue
        agent.lane_change(
            direction, same_lane_time=1.0, other_lane_time=1.0,
            lane_change_time=2.0)
        return direction
    return None


def set_lane_change_signal(vehicle, direction):
    """Toggle only the requested blinker while preserving other vehicle lights."""
    try:
        current = int(vehicle.get_light_state())
        blinkers = (int(carla.VehicleLightState.LeftBlinker)
                    | int(carla.VehicleLightState.RightBlinker))
        current &= ~blinkers
        if direction == "left":
            current |= int(carla.VehicleLightState.LeftBlinker)
        elif direction == "right":
            current |= int(carla.VehicleLightState.RightBlinker)
        vehicle.set_light_state(carla.VehicleLightState(current))
    except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
        print("LIGHT WARNING: failed to set lane-change signal: %s" % exc,
              flush=True)


def lane_end_location(waypoint):
    """Return the last forward point on the current lane when available."""
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


def crosswalk_locations(world):
    """Load named RoadLines crosswalks when the CARLA map exposes them."""
    cache_key = id(world)
    if cache_key in _CROSSWALK_CACHE:
        return _CROSSWALK_CACHE[cache_key]
    locations = []
    try:
        label = carla.CityObjectLabel.RoadLines
        objects = world.get_environment_objects(label)
        for obj in objects:
            name = str(getattr(obj, "name", "")).lower()
            if not any(token in name for token in (
                    "crosswalk", "zebra", "stopline", "stop_line",
                    "stop line")):
                continue
            transform = obj.transform
            bbox_location = getattr(obj.bounding_box, "location", None)
            if bbox_location is not None:
                locations.append(transform.transform(bbox_location))
            else:
                locations.append(transform.location)
    except (AttributeError, RuntimeError, TypeError):
        locations = []
    _CROSSWALK_CACHE[cache_key] = locations
    return locations


def crosswalk_target(world, carla_map, ego_location, ego_forward,
                     ego_right, lane_width, route_lanes, max_distance,
                     before_longitudinal=None):
    """Return the nearest crosswalk center on the planned lane."""
    candidates = []
    for location in crosswalk_locations(world):
        delta = location - ego_location
        longitudinal = delta.x * ego_forward.x + delta.y * ego_forward.y
        lateral = abs(delta.x * ego_right.x + delta.y * ego_right.y)
        if longitudinal <= 0.0 or longitudinal > max_distance:
            continue
        if (before_longitudinal is not None
                and longitudinal > before_longitudinal + 1.0):
            continue
        if lateral > lane_width * 1.5:
            continue
        crosswalk_waypoint = carla_map.get_waypoint(
            location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if crosswalk_waypoint is None:
            continue
        lane_key = (crosswalk_waypoint.road_id, crosswalk_waypoint.lane_id)
        if lane_key not in route_lanes:
            continue
        candidates.append((longitudinal, location))
    if not candidates:
        return None
    # Select the crossing immediately before the signal, not the first one
    # encountered from far away. This prevents an unnecessarily early stop.
    return max(candidates, key=lambda item: item[0])[1]


def limit_junction_control(control, vehicle, target_speed_kph,
                           max_steer, max_throttle):
    """Bound speed and steering energy while entering an intersection."""
    speed_kph = 3.6 * vehicle_speed_mps(vehicle)
    control.steer = clamp(float(control.steer), -max_steer, max_steer)
    control.throttle = min(float(control.throttle), max_throttle)
    if abs(control.steer) >= max_steer * 0.75:
        control.throttle = min(control.throttle, 0.10)
    if speed_kph > target_speed_kph + 0.5:
        control.throttle = 0.0
        control.brake = max(float(control.brake), 0.18)
    control.hand_brake = False
    return control


def main():
    args = parse_args()
    if args.override_max_age_sec <= 0.0:
        raise ValueError("--override-max-age-sec must be positive")
    if not 5.0 <= args.junction_speed_kph <= 40.0:
        raise ValueError("--junction-speed-kph must be between 5 and 40")
    stopping = [False]

    def stop(_signum, _frame):
        stopping[0] = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)
    try:
        world = client.get_world()
    except RuntimeError as exc:
        raise SystemExit("Cannot connect to CARLA at %s:%d: %s" % (args.host, args.port, exc))
    carla_map = world.get_map()
    map_name = carla_map.name.rsplit("/", 1)[-1]
    if map_name != "Town04_Opt":
        raise SystemExit(
            "Town04_Opt is required, but current map is %s" % map_name)
    vehicle = get_ego(world, args.role_name)
    if vehicle is None:
        raise SystemExit("No vehicle with role_name=%r. Run carla06spawn first." % args.role_name)

    # This is CARLA 0.9.15's unmodified navigation agent: signals, speed
    # limits, vehicle avoidance, and route planning are handled locally.
    agent_options = {
        "max_throttle": 0.80,
        # The custom detector below owns signal handling. The stock
        # BehaviorAgent latch can retain a wrong-lane red light indefinitely.
        "ignore_traffic_lights": True,
        "base_tlight_threshold": STOP_TARGET_MARGIN_M,
        "max_brake": 1.0,
        "base_vehicle_threshold": 12.0,
        "max_steering": args.max_steering,
        # Slightly soften proportional steering response to avoid fine
        # left-right twitching on Town04's long curves.
        "lateral_control_dict": {"K_P": 0.90, "K_I": 0.01, "K_D": 0.05, "dt": 1.0 / 20.0},
    }
    agent = BehaviorAgent(vehicle, behavior=args.behavior, opt_dict=agent_options)
    # Keep a stable main-road cruise target. BehaviorAgent otherwise recomputes
    # speed_limit - speed_lim_dist every tick, which can oscillate on long roads.
    agent._behavior.max_speed += args.speed_increase
    agent._behavior.speed_lim_dist = 0.0
    cap_agent_cruise_speed(agent, vehicle.get_speed_limit())
    # Do not let the stock tailgating heuristic rewrite the route for a lane change;
    # it can rewrite a stable Town04 route into an abrupt lane change.
    agent._behavior.tailgate_counter = 1000000000
    agent.set_destination(
        choose_destination(world, vehicle, args.destination_distance),
        start_location=vehicle.get_location(), clean_queue=True)
    if not args.no_follow_camera:
        update_follow_camera(world, vehicle)
    print("BehaviorAgent ready: vehicle=%d map=%s behavior=%s" %
          (vehicle.id, carla_map.name, args.behavior), flush=True)
    previous_steer = 0.0
    active_stop_light_id = None
    active_stop_line_location = None
    green_ticks = 0
    route_replan_pending = False
    release_active = False
    release_started = 0.0
    release_location = None
    stalled_since = None
    next_lane_change_at = time.monotonic() + random.uniform(
        LANE_CHANGE_MIN_INTERVAL_SEC, LANE_CHANGE_MAX_INTERVAL_SEC)
    lane_change_signal_direction = None
    lane_change_signal_until = 0.0
    safety_override_active = False
    try:
        while not stopping[0]:
            if not vehicle.is_alive:
                print("Ego vehicle was removed.", flush=True)
                break
            safety_control = read_safety_override(
                args.behavior_override_file, args.role_name,
                args.override_max_age_sec)
            if safety_control is not None:
                # This process remains the sole apply_control() caller. The
                # supervisor only publishes an atomic planner command here.
                if not safety_override_active:
                    print(
                        "CONTROL AUTHORITY: BehaviorAgent yielding to safety "
                        "override commands.", flush=True)
                safety_override_active = True
                vehicle.apply_control(safety_control)
                if not args.no_follow_camera:
                    update_follow_camera(world, vehicle)
                world.wait_for_tick(1.0)
                continue
            if safety_override_active:
                # The agent did not run while MRM owned the plan. Its old route
                # and PID history are no longer valid at the new lane/location.
                previous_steer = clamp(
                    float(vehicle.get_control().steer),
                    -args.max_steering, args.max_steering)
                reset_agent_controller(agent, steer=previous_steer)
                agent.set_destination(
                    choose_destination(
                        world, vehicle, args.destination_distance),
                    start_location=vehicle.get_location(), clean_queue=True)
                active_stop_light_id = None
                green_ticks = 0
                route_replan_pending = False
                release_active = True
                release_started = time.monotonic()
                release_location = vehicle.get_location()
                safety_override_active = False
                print(
                    "CONTROL AUTHORITY: safety override released; route and "
                    "PID rebuilt from current vehicle pose.", flush=True)
            detection_distance = traffic_light_detection_distance(vehicle)
            stop_light, stop_distance, stop_line_location = upcoming_stop_light(
                agent, detection_distance)
            junction_limited = False
            now = time.monotonic()
            if (lane_change_signal_direction is not None
                    and now >= lane_change_signal_until):
                set_lane_change_signal(vehicle, None)
                lane_change_signal_direction = None
            if (now >= next_lane_change_at
                    and stop_light is None
                    and active_stop_light_id is None
                    and not release_active
                    and not route_near_junction(agent, 25.0)):
                changed_direction = request_lane_change(agent, world, vehicle)
                next_lane_change_at = now + random.uniform(
                    LANE_CHANGE_MIN_INTERVAL_SEC,
                    LANE_CHANGE_MAX_INTERVAL_SEC)
                if changed_direction is not None:
                    route_replan_pending = True
                    lane_change_signal_direction = changed_direction
                    lane_change_signal_until = (
                        now + LANE_CHANGE_SIGNAL_DURATION_SEC)
                    set_lane_change_signal(vehicle, changed_direction)
                    print(
                        "Occasional lane change requested: %s." %
                        changed_direction, flush=True)
            if stop_light is not None:
                if agent.done():
                    route_replan_pending = True
                    control = carla.VehicleControl()
                else:
                    control = agent.run_step(debug=args.debug)
                if active_stop_light_id != stop_light.id:
                    print(
                        "%s light %d detected at %.1fm: keeping cruise until "
                        "%.1fm before braking." % (
                            str(stop_light.state), stop_light.id, stop_distance,
                            SIGNAL_CONTROL_DISTANCE_M), flush=True)
                active_stop_light_id = stop_light.id
                active_stop_line_location = stop_line_location
                green_ticks = 0
                release_active = False
                if signal_stop_is_due(stop_distance):
                    control = stop_at_line(control, vehicle, stop_distance)
            else:
                control = None
                if active_stop_light_id is not None:
                    remembered_light = world.get_actor(active_stop_light_id)
                    remembered_state = (
                        remembered_light.get_state()
                        if remembered_light is not None
                        else carla.TrafficLightState.Unknown)
                    passed_stop_line = False
                    if active_stop_line_location is not None:
                        current_transform = vehicle.get_transform()
                        delta = (active_stop_line_location
                                 - current_transform.location)
                        forward = current_transform.get_forward_vector()
                        passed_stop_line = (
                            delta.x * forward.x + delta.y * forward.y < -1.5)
                    if passed_stop_line:
                        active_stop_light_id = None
                        active_stop_line_location = None
                        green_ticks = 0
                    else:
                        remembered_distance = distance_to_stop_line(
                            vehicle, active_stop_line_location)
                        remembered_control = agent.run_step(debug=args.debug)
                        if remembered_state != carla.TrafficLightState.Green:
                            green_ticks = 0
                            if active_stop_line_location is None:
                                control = hold_at_signal()
                            elif signal_stop_is_due(remembered_distance):
                                # The trigger waypoint can briefly disappear
                                # at an intersection. Keep approaching the
                                # saved line and only brake inside 6 metres.
                                control = stop_at_line(
                                    remembered_control, vehicle,
                                    remembered_distance)
                            else:
                                control = remembered_control
                        else:
                            green_ticks += 1
                            # A green light seen while still farther than 6m
                            # must not cause an unnecessary full stop.
                            if (green_ticks < 10
                                    and signal_stop_is_due(remembered_distance)):
                                control = hold_at_signal()
                            else:
                                control = remembered_control
                                if green_ticks >= 10:
                                    if route_replan_pending or agent.done():
                                        agent.set_destination(
                                            choose_destination(
                                                world, vehicle,
                                                args.destination_distance),
                                            start_location=vehicle.get_location(),
                                            clean_queue=True)
                                        print(
                                            "Route rebuilt from stop line after green.",
                                            flush=True)
                                    route_replan_pending = False
                                    active_stop_light_id = None
                                    active_stop_line_location = None
                                    green_ticks = 0
                                    previous_steer = 0.0
                                    reset_agent_controller(agent, steer=0.0)
                                    release_active = True
                                    release_started = time.monotonic()
                                    release_location = vehicle.get_location()
                                    print(
                                        "Green stable: controlled junction rollout.",
                                        flush=True)

                if control is None:
                    if agent.done():
                        agent.set_destination(
                            choose_destination(
                                world, vehicle, args.destination_distance),
                            start_location=vehicle.get_location(),
                            clean_queue=True)
                        reset_agent_controller(agent, steer=previous_steer)
                        if vehicle_in_junction(carla_map, vehicle):
                            release_active = True
                            release_started = time.monotonic()
                            release_location = vehicle.get_location()
                        print("New forward destination selected.", flush=True)
                    control = agent.run_step(debug=args.debug)
                    if release_active:
                        current_waypoint = carla_map.get_waypoint(
                            vehicle.get_location(), project_to_road=True,
                            lane_type=carla.LaneType.Driving)
                        # Keep the rollout state after a green light, but do
                        # not slow down until the vehicle is actually inside
                        # the junction polygon.
                        if (current_waypoint is not None
                                and current_waypoint.is_junction):
                            control = limit_junction_control(
                                control, vehicle,
                                target_speed_kph=args.junction_speed_kph,
                                max_steer=args.max_steering,
                                max_throttle=0.45)
                            junction_limited = True
                        travelled = (vehicle.get_location().distance(
                            release_location)
                            if release_location is not None else 0.0)
                        elapsed = time.monotonic() - release_started
                        if (elapsed >= 3.0 and travelled >= 18.0
                                and current_waypoint is not None
                                and not current_waypoint.is_junction):
                            release_active = False
                            print(
                                "Controlled junction rollout complete.",
                                flush=True)
                    elif vehicle_in_junction(carla_map, vehicle):
                        control = limit_junction_control(
                            control, vehicle,
                            target_speed_kph=args.junction_speed_kph,
                            max_steer=args.max_steering,
                            max_throttle=0.45)
                        junction_limited = True

            # A local planner can retain a stale queue at a topology edge even
            # though the process and CARLA connection are healthy.  Rebuild
            # from the current pose only when no signal/override owns the stop.
            speed_mps = vehicle_speed_mps(vehicle)
            if (stop_light is None and active_stop_light_id is None
                    and control.throttle > 0.08 and control.brake < 0.8
                    and speed_mps < 0.25):
                if stalled_since is None:
                    stalled_since = time.monotonic()
                elif time.monotonic() - stalled_since >= ROUTE_STALL_TIMEOUT_SEC:
                    agent.set_destination(
                        choose_destination(
                            world, vehicle, args.destination_distance),
                        start_location=vehicle.get_location(),
                        clean_queue=True)
                    reset_agent_controller(agent, steer=previous_steer)
                    control = agent.run_step(debug=args.debug)
                    stalled_since = None
                    print(
                        "Route rebuilt after %.1fs stationary progress stall."
                        % ROUTE_STALL_TIMEOUT_SEC, flush=True)
            else:
                stalled_since = None

            steer_rate = (0.035 if release_active
                          else 0.045 if junction_limited
                          else args.steer_rate)
            control = smooth_steering(control, previous_steer, steer_rate)
            previous_steer = control.steer
            vehicle.apply_control(control)
            if not args.no_follow_camera:
                update_follow_camera(world, vehicle)
            world.wait_for_tick(1.0)
    finally:
        set_lane_change_signal(vehicle, None)
        if vehicle.is_alive:
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
        print("BehaviorAgent stopped.", flush=True)


if __name__ == "__main__":
    main()
