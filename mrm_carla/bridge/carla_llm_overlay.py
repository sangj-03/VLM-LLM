#!/usr/bin/env python3

"""Send CARLA telemetry to a remote LLM and overlay brake/steer commands.

This program is intentionally independent from the existing BehaviorAgent
autopilot.  It copies the most recently applied autopilot control and changes
only steer/brake (and clears throttle while the LLM is braking).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import queue
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


CARLA_PYTHONAPI_DIR = os.environ.get(
    "CARLA_PYTHONAPI_DIR", "/home/tmo/carla/PythonAPI")
sys.path.insert(0, os.path.join(CARLA_PYTHONAPI_DIR, "carla"))
egg_matches = glob.glob(os.path.join(
    CARLA_PYTHONAPI_DIR,
    "carla/dist/carla-*%d.%d-linux-x86_64.egg" % (
        sys.version_info.major, sys.version_info.minor)))
if egg_matches:
    sys.path.insert(0, egg_matches[0])

import carla  # noqa: E402


PROTOCOL_VERSION = 1


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def vector_dict(vector: Any) -> Dict[str, float]:
    return {
        "x": round(float(vector.x), 5),
        "y": round(float(vector.y), 5),
        "z": round(float(vector.z), 5),
    }


def magnitude(vector: Any) -> float:
    return math.sqrt(vector.x ** 2 + vector.y ** 2 + vector.z ** 2)


def normalized_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def require_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a number" % name)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("%s must be finite" % name)
    return result


@dataclass(frozen=True)
class RemoteCommand:
    request_id: int
    steer: float
    brake: float
    expires_at: float
    reason: str


class RemoteControlClient:
    """One-slot asynchronous HTTP client so CARLA never waits for the LLM."""

    def __init__(self, server_url: str, timeout_sec: float,
                 token: str, max_abs_steer: float) -> None:
        self._server_url = server_url
        self._timeout_sec = timeout_sec
        self._token = token
        self._max_abs_steer = max_abs_steer
        self._requests = queue.Queue(maxsize=1)  # type: queue.Queue
        self._lock = threading.Lock()
        self._latest_command: Optional[RemoteCommand] = None
        self._last_error = "waiting for first response"
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._worker, name="llm-http-worker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=self._timeout_sec + 0.5)

    def submit_latest(self, telemetry: Dict[str, Any]) -> None:
        try:
            self._requests.put_nowait(telemetry)
            return
        except queue.Full:
            pass
        try:
            self._requests.get_nowait()
        except queue.Empty:
            pass
        try:
            self._requests.put_nowait(telemetry)
        except queue.Full:
            pass

    def latest_command(self) -> Optional[RemoteCommand]:
        with self._lock:
            command = self._latest_command
        if command is None or time.monotonic() > command.expires_at:
            return None
        return command

    def status(self) -> Tuple[Optional[RemoteCommand], str]:
        with self._lock:
            return self._latest_command, self._last_error

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                telemetry = self._requests.get(timeout=0.2)
            except queue.Empty:
                continue
            if telemetry is None:
                continue
            try:
                response = self._post_json(telemetry)
                command = self._parse_response(response, telemetry["request_id"])
                with self._lock:
                    self._latest_command = command
                    self._last_error = ""
            except Exception as exc:  # network/model failures must not stop CARLA
                with self._lock:
                    self._last_error = "%s: %s" % (type(exc).__name__, exc)

    def _post_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "carla-llm-overlay/1",
        }
        if self._token:
            headers["Authorization"] = "Bearer " + self._token
        request = urllib.request.Request(
            self._server_url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(
                    request, timeout=self._timeout_sec) as response:
                response_body = response.read(65537)
        except urllib.error.HTTPError as exc:
            details = exc.read(1024).decode("utf-8", errors="replace")
            raise RuntimeError("server HTTP %d: %s" % (exc.code, details))
        if len(response_body) > 65536:
            raise ValueError("server response is larger than 64 KiB")
        parsed = json.loads(response_body.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("server response must be a JSON object")
        return parsed

    def _parse_response(self, response: Dict[str, Any],
                        expected_request_id: int) -> RemoteCommand:
        if int(response.get("protocol_version", -1)) != PROTOCOL_VERSION:
            raise ValueError("unsupported protocol_version")
        if int(response.get("request_id", -1)) != expected_request_id:
            raise ValueError("response request_id does not match request")
        command = response.get("command")
        if not isinstance(command, dict):
            raise ValueError("response.command must be a JSON object")
        steer = require_number(command.get("steer"), "command.steer")
        brake = require_number(command.get("brake"), "command.brake")
        if not -1.0 <= steer <= 1.0:
            raise ValueError("command.steer must be in [-1, 1]")
        if not 0.0 <= brake <= 1.0:
            raise ValueError("command.brake must be in [0, 1]")
        valid_for_ms = require_number(
            response.get("valid_for_ms", 1000), "valid_for_ms")
        valid_for_ms = clamp(valid_for_ms, 100.0, 5000.0)
        return RemoteCommand(
            request_id=expected_request_id,
            steer=clamp(steer, -self._max_abs_steer, self._max_abs_steer),
            brake=brake,
            expires_at=time.monotonic() + valid_for_ms / 1000.0,
            reason=str(command.get("reason", ""))[:200],
        )


class TelemetryBuilder:
    def __init__(self, world: carla.World, vehicle: carla.Vehicle,
                 role_name: str, nearby_radius_m: float,
                 max_nearby_actors: int) -> None:
        self.world = world
        self.world_map = world.get_map()
        self.vehicle = vehicle
        self.role_name = role_name
        self.nearby_radius_m = nearby_radius_m
        self.max_nearby_actors = max_nearby_actors

    def build(self, request_id: int) -> Dict[str, Any]:
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        acceleration = self.vehicle.get_acceleration()
        angular_velocity = self.vehicle.get_angular_velocity()
        control = self.vehicle.get_control()
        snapshot = self.world.get_snapshot()

        return {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "sent_at_unix_s": round(time.time(), 6),
            "simulation": {
                "map": self.world_map.name.rsplit("/", 1)[-1],
                "frame": int(snapshot.frame),
                "elapsed_seconds": round(
                    float(snapshot.timestamp.elapsed_seconds), 5),
            },
            "vehicle": {
                "actor_id": int(self.vehicle.id),
                "role_name": self.role_name,
                "type_id": self.vehicle.type_id,
                "speed_mps": round(magnitude(velocity), 5),
                "speed_kph": round(3.6 * magnitude(velocity), 3),
                "speed_limit_kph": round(
                    float(self.vehicle.get_speed_limit()), 3),
                "location_m": vector_dict(transform.location),
                "rotation_deg": {
                    "pitch": round(float(transform.rotation.pitch), 4),
                    "yaw": round(float(transform.rotation.yaw), 4),
                    "roll": round(float(transform.rotation.roll), 4),
                },
                "velocity_mps": vector_dict(velocity),
                "acceleration_mps2": vector_dict(acceleration),
                "angular_velocity_deg_s": vector_dict(angular_velocity),
                "autopilot_control": self._control_dict(control),
            },
            "road": self._road_data(transform),
            "traffic_light": self._traffic_light_data(transform),
            "nearby_vehicles": self._nearby_actor_data(
                transform, "vehicle.*"),
            "nearby_walkers": self._nearby_actor_data(
                transform, "walker.pedestrian.*"),
        }

    @staticmethod
    def _control_dict(control: carla.VehicleControl) -> Dict[str, Any]:
        return {
            "throttle": round(float(control.throttle), 5),
            "steer": round(float(control.steer), 5),
            "brake": round(float(control.brake), 5),
            "hand_brake": bool(control.hand_brake),
            "reverse": bool(control.reverse),
            "gear": int(control.gear),
        }

    def _road_data(self, transform: carla.Transform) -> Dict[str, Any]:
        waypoint = self.world_map.get_waypoint(
            transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if waypoint is None:
            return {"on_driving_lane": False}

        center = waypoint.transform.location
        right = waypoint.transform.get_right_vector()
        delta_x = transform.location.x - center.x
        delta_y = transform.location.y - center.y
        delta_z = transform.location.z - center.z
        lane_offset = delta_x * right.x + delta_y * right.y + delta_z * right.z
        heading_error = normalized_angle_deg(
            transform.rotation.yaw - waypoint.transform.rotation.yaw)
        curvature = 0.0
        next_waypoints = waypoint.next(5.0)
        if next_waypoints:
            yaw_delta_rad = math.radians(normalized_angle_deg(
                next_waypoints[0].transform.rotation.yaw
                - waypoint.transform.rotation.yaw))
            curvature = yaw_delta_rad / 5.0

        return {
            "on_driving_lane": True,
            "road_id": int(waypoint.road_id),
            "section_id": int(waypoint.section_id),
            "lane_id": int(waypoint.lane_id),
            "is_junction": bool(waypoint.is_junction),
            "lane_width_m": round(float(waypoint.lane_width), 4),
            "lane_offset_m": round(float(lane_offset), 5),
            "lane_offset_sign": "positive_is_right_of_center",
            "heading_error_deg": round(float(heading_error), 5),
            "curvature_rad_per_m": round(float(curvature), 7),
        }

    def _traffic_light_data(self, transform: carla.Transform) -> Dict[str, Any]:
        light = self.vehicle.get_traffic_light()
        if light is not None:
            return {
                "affecting_vehicle": True,
                "actor_id": int(light.id),
                "state": str(light.state).rsplit(".", 1)[-1],
                "distance_m": round(
                    transform.location.distance(light.get_location()), 4),
            }

        forward = transform.get_forward_vector()
        candidates = []
        for candidate in self.world.get_actors().filter("traffic.traffic_light*"):
            location = candidate.get_location()
            delta_x = location.x - transform.location.x
            delta_y = location.y - transform.location.y
            longitudinal = delta_x * forward.x + delta_y * forward.y
            distance = transform.location.distance(location)
            if 0.0 <= longitudinal <= 60.0 and distance <= 60.0:
                candidates.append((distance, candidate))
        if not candidates:
            return {"affecting_vehicle": False, "state": "Unknown"}
        distance, light = min(candidates, key=lambda item: item[0])
        return {
            "affecting_vehicle": False,
            "actor_id": int(light.id),
            "state": str(light.state).rsplit(".", 1)[-1],
            "distance_m": round(float(distance), 4),
            "note": "nearest light ahead; it may control another lane",
        }

    def _nearby_actor_data(self, transform: carla.Transform,
                           pattern: str) -> List[Dict[str, Any]]:
        ego_location = transform.location
        ego_velocity = self.vehicle.get_velocity()
        forward = transform.get_forward_vector()
        right = transform.get_right_vector()
        actors = []
        for actor in self.world.get_actors().filter(pattern):
            if actor.id == self.vehicle.id:
                continue
            location = actor.get_location()
            distance = ego_location.distance(location)
            if distance > self.nearby_radius_m:
                continue
            delta_x = location.x - ego_location.x
            delta_y = location.y - ego_location.y
            delta_z = location.z - ego_location.z
            actor_velocity = actor.get_velocity()
            longitudinal = (
                delta_x * forward.x + delta_y * forward.y + delta_z * forward.z)
            lateral = delta_x * right.x + delta_y * right.y + delta_z * right.z
            relative_velocity = (
                (actor_velocity.x - ego_velocity.x) * forward.x
                + (actor_velocity.y - ego_velocity.y) * forward.y
                + (actor_velocity.z - ego_velocity.z) * forward.z)
            actors.append((distance, {
                "actor_id": int(actor.id),
                "type_id": actor.type_id,
                "distance_m": round(float(distance), 4),
                "longitudinal_m": round(float(longitudinal), 4),
                "lateral_m": round(float(lateral), 4),
                "relative_longitudinal_speed_mps": round(
                    float(relative_velocity), 4),
                "speed_mps": round(float(magnitude(actor_velocity)), 4),
            }))
        actors.sort(key=lambda item: item[0])
        return [item[1] for item in actors[:self.max_nearby_actors]]


def copy_control_with_overlay(base: carla.VehicleControl, steer: float,
                              brake: float,
                              preserve_autopilot_brake: bool) -> carla.VehicleControl:
    output = carla.VehicleControl()
    output.throttle = float(base.throttle)
    output.steer = float(steer)
    output.brake = max(float(base.brake), brake) \
        if preserve_autopilot_brake else float(brake)
    output.hand_brake = bool(base.hand_brake)
    output.reverse = bool(base.reverse)
    output.manual_gear_shift = bool(base.manual_gear_shift)
    output.gear = int(base.gear)
    if output.brake > 0.05:
        output.throttle = 0.0
    return output


def wait_for_world(host: str, port: int) -> Tuple[carla.Client, carla.World]:
    client = carla.Client(host, port)
    client.set_timeout(2.0)
    printed = False
    while True:
        try:
            return client, client.get_world()
        except RuntimeError:
            if not printed:
                print("CARLA server waiting at %s:%d ..." % (host, port), flush=True)
                printed = True
            time.sleep(1.0)


def wait_for_vehicle(world: carla.World, role_name: str) -> carla.Vehicle:
    printed = False
    while True:
        matches = [
            actor for actor in world.get_actors().filter("vehicle.*")
            if actor.attributes.get("role_name") == role_name]
        if matches:
            return min(matches, key=lambda actor: actor.id)
        if not printed:
            print("Ego vehicle waiting (role_name=%s) ..." % role_name, flush=True)
            printed = True
        time.sleep(1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay remote LLM brake/steer on CARLA BehaviorAgent")
    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--role-name", default="town06_ego")
    parser.add_argument(
        "--server-url", default="http://10.3.70.193:8000/control")
    parser.add_argument(
        "--token", default=os.environ.get("LLM_BRIDGE_TOKEN", "").strip(),
        help="shared bearer token (or set LLM_BRIDGE_TOKEN)")
    parser.add_argument("--send-hz", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=3.0)
    parser.add_argument("--overlay-delay", type=float, default=0.01,
                        help="seconds to wait after a CARLA tick for autopilot")
    parser.add_argument("--max-abs-steer", type=float, default=0.65)
    parser.add_argument("--max-steer-step", type=float, default=0.08)
    parser.add_argument("--nearby-radius", type=float, default=50.0)
    parser.add_argument("--max-nearby-actors", type=int, default=12)
    parser.add_argument(
        "--allow-llm-to-release-autopilot-brake", action="store_true",
        help="unsafe: permit LLM brake below the current autopilot brake")
    parser.add_argument(
        "--stale-action", choices=("release", "brake"), default="release",
        help="release lets the unchanged autopilot take over after timeout")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.send_hz <= 0.0:
        raise ValueError("--send-hz must be positive")
    if args.request_timeout <= 0.0:
        raise ValueError("--request-timeout must be positive")
    if not 0.0 <= args.overlay_delay <= 1.0:
        raise ValueError("--overlay-delay must be between 0 and 1 second")
    if not 0.0 < args.max_abs_steer <= 1.0:
        raise ValueError("--max-abs-steer must be in (0, 1]")
    if not 0.0 < args.max_steer_step <= 2.0:
        raise ValueError("--max-steer-step must be in (0, 2]")


def main() -> None:
    args = parse_args()
    validate_args(args)
    _, world = wait_for_world(args.carla_host, args.carla_port)
    vehicle = wait_for_vehicle(world, args.role_name)
    telemetry_builder = TelemetryBuilder(
        world, vehicle, args.role_name,
        args.nearby_radius, args.max_nearby_actors)
    remote = RemoteControlClient(
        args.server_url, args.request_timeout, args.token,
        args.max_abs_steer)
    remote.start()

    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    print(
        "LLM OVERLAY READY  actor_id=%d role_name=%s server=%s send_hz=%.2f" % (
            vehicle.id, args.role_name, args.server_url, args.send_hz),
        flush=True)
    print(
        "Autopilot source is unchanged; only brake/steer are overlaid. "
        "Stale action=%s." % args.stale_action,
        flush=True)

    request_id = 0
    next_send_time = 0.0
    last_status_time = 0.0
    last_overlay_steer: Optional[float] = None
    send_period = 1.0 / args.send_hz

    try:
        while not stop_requested and vehicle.is_alive:
            world.wait_for_tick(10.0)
            if args.overlay_delay:
                time.sleep(args.overlay_delay)

            now = time.monotonic()
            if now >= next_send_time:
                request_id += 1
                try:
                    telemetry = telemetry_builder.build(request_id)
                    remote.submit_latest(telemetry)
                except RuntimeError as exc:
                    print("Telemetry skipped: %s" % exc, flush=True)
                next_send_time = now + send_period

            command = remote.latest_command()
            if command is not None:
                base = vehicle.get_control()
                if last_overlay_steer is None:
                    last_overlay_steer = float(base.steer)
                steer_delta = clamp(
                    command.steer - last_overlay_steer,
                    -args.max_steer_step,
                    args.max_steer_step)
                last_overlay_steer = clamp(
                    last_overlay_steer + steer_delta,
                    -args.max_abs_steer,
                    args.max_abs_steer)
                output = copy_control_with_overlay(
                    base, last_overlay_steer, command.brake,
                    not args.allow_llm_to_release_autopilot_brake)
                vehicle.apply_control(output)
            elif args.stale_action == "brake":
                base = vehicle.get_control()
                vehicle.apply_control(copy_control_with_overlay(
                    base, base.steer, 1.0, True))
                last_overlay_steer = None
            else:
                # Applying nothing hands control back to the unchanged autopilot.
                last_overlay_steer = None

            if now - last_status_time >= 2.0:
                latest, error = remote.status()
                active = remote.latest_command()
                if active is None:
                    print("LLM OVERLAY inactive  status=%s" % (
                        error or "response expired"), flush=True)
                else:
                    print(
                        "LLM OVERLAY active  request=%d brake=%.2f steer=%+.2f "
                        "reason=%s" % (
                            active.request_id, active.brake, active.steer,
                            active.reason or "-"),
                        flush=True)
                last_status_time = now
    except KeyboardInterrupt:
        pass
    finally:
        remote.stop()
        # Do not apply a final control: the existing autopilot remains the owner.
        print("LLM overlay stopped; autopilot control is untouched.", flush=True)


if __name__ == "__main__":
    main()
