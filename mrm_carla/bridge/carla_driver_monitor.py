#!/usr/bin/env python3

"""Send a cabin image with synchronized CARLA telemetry to the DGX monitor.

Telemetry is sampled into a short local ring buffer but never triggers model
inference by itself.  A new cabin image is the only inference trigger.  The
HTTP worker keeps at most one request in flight and one newest pending request.
"""

from __future__ import annotations

import argparse
import base64
from collections import deque
import json
import math
import os
import queue
import signal
import stat
import threading
import cv2
import time
import urllib.error
import urllib.request
from typing import Any, Deque, Dict, Optional, Tuple

from carla_llm_overlay import TelemetryBuilder, wait_for_vehicle, wait_for_world


PROTOCOL_VERSION = 1
MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_RESPONSE_BYTES = 64 * 1024
MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
RISK_LEVELS = frozenset(("normal", "caution", "critical", "unknown"))


def require_boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError("%s must be a boolean" % name)
    return value


def require_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a number" % name)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("%s must be finite" % name)
    return result


class CabinImageReader:
    """Read only newly written, stable image files."""

    def __init__(self, path: str, max_age_sec: float,
                 allow_reuse: bool) -> None:
        self.path = os.path.abspath(path)
        self.max_age_sec = max_age_sec
        self.allow_reuse = allow_reuse
        self._last_fingerprint: Optional[Tuple[int, int]] = None

    def read_new(self) -> Optional[Tuple[Dict[str, str], float]]:
        try:
            before = os.stat(self.path)
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("cabin image path is not a regular file")
        suffix = os.path.splitext(self.path)[1].lower()
        mime_type = MIME_TYPES.get(suffix)
        if mime_type is None:
            raise ValueError("cabin image must be JPEG, PNG, or WebP")
        if before.st_size <= 0:
            raise ValueError("cabin image is empty")
        if before.st_size > MAX_IMAGE_BYTES:
            raise ValueError(
                "cabin image exceeds %d bytes; resize or compress it" %
                MAX_IMAGE_BYTES)
        fingerprint = (before.st_mtime_ns, before.st_size)
        if not self.allow_reuse and fingerprint == self._last_fingerprint:
            return None
        captured_at = before.st_mtime
        age = max(0.0, time.time() - captured_at)
        if age > self.max_age_sec:
            raise ValueError(
                "cabin image is stale (age %.2fs, limit %.2fs)" %
                (age, self.max_age_sec))
        with open(self.path, "rb") as image_file:
            image_bytes = image_file.read(MAX_IMAGE_BYTES + 1)
        after = os.stat(self.path)
        after_fingerprint = (after.st_mtime_ns, after.st_size)
        if fingerprint != after_fingerprint:
            raise RuntimeError("cabin image changed while it was being read")
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise ValueError("cabin image is too large")
        self._last_fingerprint = fingerprint
        return ({
            "mime_type": mime_type,
            "data_base64": base64.b64encode(image_bytes).decode("ascii"),
        }, captured_at)


class MonitorHttpClient:
    """Latest-only asynchronous monitor client."""

    def __init__(self, server_url: str, token: str, timeout_sec: float,
                 max_result_age_sec: float) -> None:
        self.server_url = server_url
        self.token = token
        self.timeout_sec = timeout_sec
        self.max_result_age_sec = max_result_age_sec
        self._requests = queue.Queue(maxsize=1)  # type: queue.Queue
        self._lock = threading.Lock()
        self._latest_result: Optional[Dict[str, Any]] = None
        self._last_error = "waiting for first cabin image"
        self._dropped_pending = 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._worker, name="driver-monitor-http", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=self.timeout_sec + 0.5)

    def submit_latest(self, observation: Dict[str, Any]) -> None:
        item = (observation, time.monotonic())
        try:
            self._requests.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            self._requests.get_nowait()
            with self._lock:
                self._dropped_pending += 1
        except queue.Empty:
            pass
        try:
            self._requests.put_nowait(item)
        except queue.Full:
            pass

    def status(self) -> Tuple[Optional[Dict[str, Any]], str, int]:
        with self._lock:
            return (
                self._latest_result,
                self._last_error,
                self._dropped_pending,
            )

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._requests.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                continue
            observation, submitted_at = item
            try:
                response = self._post_json(observation)
                result = self._parse_response(
                    response, observation["observation_id"])
                result_age = time.monotonic() - submitted_at
                if result_age > self.max_result_age_sec:
                    raise RuntimeError(
                        "discarded stale result (age %.2fs)" % result_age)
                result["end_to_end_age_ms"] = round(result_age * 1000.0, 2)
                with self._lock:
                    self._latest_result = result
                    self._last_error = ""
            except Exception as exc:  # monitoring errors must not stop CARLA
                with self._lock:
                    self._last_error = "%s: %s" % (type(exc).__name__, exc)

    def _post_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "carla-driver-monitor/1",
        }
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        request = urllib.request.Request(
            self.server_url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(
                    request, timeout=self.timeout_sec) as response:
                response_body = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            details = exc.read(2048).decode("utf-8", errors="replace")
            raise RuntimeError("server HTTP %d: %s" % (exc.code, details))
        if len(response_body) > MAX_RESPONSE_BYTES:
            raise ValueError("server response exceeds 64 KiB")
        parsed = json.loads(response_body.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("server response must be a JSON object")
        return parsed

    @staticmethod
    def _parse_response(response: Dict[str, Any],
                        expected_observation_id: int) -> Dict[str, Any]:
        if int(response.get("protocol_version", -1)) != PROTOCOL_VERSION:
            raise ValueError("unsupported protocol_version")
        if int(response.get("observation_id", -1)) != expected_observation_id:
            raise ValueError("observation_id does not match request")
        state = response.get("driver_state")
        if not isinstance(state, dict):
            raise ValueError("driver_state must be a JSON object")
        risk_level = str(state.get("risk_level", "unknown")).lower()
        if risk_level not in RISK_LEVELS:
            raise ValueError("invalid risk_level")
        confidence = require_number(state.get("confidence"), "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        parsed = {
            "observation_id": expected_observation_id,
            "driver_present": require_boolean(
                state.get("driver_present"), "driver_present"),
            "looking_forward": require_boolean(
                state.get("looking_forward"), "looking_forward"),
            "eyes_closed": require_boolean(
                state.get("eyes_closed"), "eyes_closed"),
            "phone_use": require_boolean(state.get("phone_use"), "phone_use"),
            "unsafe_behavior": require_boolean(
                state.get("unsafe_behavior"), "unsafe_behavior"),
            "risk_level": risk_level,
            "confidence": confidence,
            "reason": str(state.get("reason", ""))[:200],
            "inference_ms": require_number(
                response.get("inference_ms", 0.0), "inference_ms"),
        }
        return parsed


def motion_state(speed_kph: float) -> str:
    if speed_kph >= 2.0:
        return "moving"
    if speed_kph >= 0.3:
        return "creeping"
    return "stationary"


def summarize_telemetry(
        history: Deque[Tuple[float, Dict[str, Any]]],
        window_sec: float, now_unix: float) -> Dict[str, Any]:
    samples = [
        telemetry for captured_at, telemetry in history
        if captured_at >= now_unix - window_sec
    ]
    speeds = [float(item["vehicle"].get("speed_kph", 0.0)) for item in samples]
    if not speeds:
        raise RuntimeError("telemetry history is empty")
    stationary_duration = 0.0
    previous_time = now_unix
    for captured_at, telemetry in reversed(history):
        speed = float(telemetry["vehicle"].get("speed_kph", 0.0))
        if speed >= 0.3:
            break
        stationary_duration += max(0.0, previous_time - captured_at)
        previous_time = captured_at
        if captured_at < now_unix - window_sec:
            break
    current_speed = speeds[-1]
    return {
        "window_s": window_sec,
        "sample_count": len(speeds),
        "motion_state": motion_state(current_speed),
        "current_speed_kph": round(current_speed, 3),
        "average_speed_kph": round(sum(speeds) / len(speeds), 3),
        "minimum_speed_kph": round(min(speeds), 3),
        "maximum_speed_kph": round(max(speeds), 3),
        "stationary_duration_s": round(stationary_duration, 3),
    }


def nearest_telemetry(
        history: Deque[Tuple[float, Dict[str, Any]]],
        captured_at: float) -> Tuple[Dict[str, Any], float]:
    if not history:
        raise RuntimeError("telemetry is unavailable")
    telemetry_time, telemetry = min(
        history, key=lambda item: abs(item[0] - captured_at))
    return telemetry, abs(telemetry_time - captured_at)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trigger DGX driver monitoring only when a new cabin image exists")
    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--role-name", default="town06_ego")
    parser.add_argument(
        "--server-url", default="http://127.0.0.1:8000/monitor")
    parser.add_argument(
        "--token", default=os.environ.get("LLM_BRIDGE_TOKEN", "").strip(),
        help="shared bearer token (or set LLM_BRIDGE_TOKEN)")
    parser.add_argument("--cabin-image-path", required=True,
                        help="JPEG/PNG/WebP file updated by the cabin camera")
    parser.add_argument("--image-period", type=float, default=4.0)
    parser.add_argument("--telemetry-hz", type=float, default=10.0)
    parser.add_argument("--telemetry-window", type=float, default=4.0)
    parser.add_argument("--max-sync-skew", type=float, default=0.75,
                        help="maximum image/telemetry timestamp difference in seconds")
    parser.add_argument("--max-image-age", type=float, default=8.0)
    parser.add_argument("--max-result-age", type=float, default=6.0)
    parser.add_argument("--request-timeout", type=float, default=5.0)
    parser.add_argument("--nearby-radius", type=float, default=50.0)
    parser.add_argument("--max-nearby-actors", type=int, default=12)
    parser.add_argument(
        "--allow-reuse-image", action="store_true",
        help="allow repeated inference on an unchanged image (testing only)")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "--image-period": args.image_period,
        "--telemetry-hz": args.telemetry_hz,
        "--telemetry-window": args.telemetry_window,
        "--max-sync-skew": args.max_sync_skew,
        "--max-image-age": args.max_image_age,
        "--max-result-age": args.max_result_age,
        "--request-timeout": args.request_timeout,
    }
    for name, value in positive.items():
        if value <= 0.0:
            raise ValueError("%s must be positive" % name)


def main() -> None:
    args = parse_args()
    validate_args(args)
    _, world = wait_for_world(args.carla_host, args.carla_port)
    vehicle = wait_for_vehicle(world, args.role_name)
    builder = TelemetryBuilder(
        world, vehicle, args.role_name,
        args.nearby_radius, args.max_nearby_actors)
    image_reader = CabinImageReader(
        args.cabin_image_path, args.max_image_age, args.allow_reuse_image)
    remote = MonitorHttpClient(
        args.server_url, args.token, args.request_timeout, args.max_result_age)
    remote.start()

    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    history = deque()  # type: Deque[Tuple[float, Dict[str, Any]]]
    request_id = 0
    observation_id = 0
    next_telemetry_time = 0.0
    next_image_time = 0.0
    last_status_time = 0.0
    telemetry_period = 1.0 / args.telemetry_hz
    history_retention = max(args.telemetry_window * 2.0, 10.0)

    print(
        "DRIVER MONITOR READY  actor_id=%d image_period=%.2fs server=%s" % (
            vehicle.id, args.image_period, args.server_url),
        flush=True)
    print(
        "Telemetry is cached locally; only a new cabin image triggers inference.",
        flush=True)

    try:
        while not stop_requested and vehicle.is_alive:
            world.wait_for_tick(10.0)
            now_monotonic = time.monotonic()
            now_unix = time.time()

            if now_monotonic >= next_telemetry_time:
                request_id += 1
                try:
                    telemetry = builder.build(request_id)
                    telemetry_time = float(
                        telemetry.get("sent_at_unix_s", now_unix))
                    history.append((telemetry_time, telemetry))
                    while history and history[0][0] < now_unix - history_retention:
                        history.popleft()
                except RuntimeError as exc:
                    print("Telemetry skipped: %s" % exc, flush=True)
                next_telemetry_time = now_monotonic + telemetry_period

            if now_monotonic >= next_image_time:
                try:
                    image_result = image_reader.read_new()
                    if image_result is None:
                        print(
                            "Monitor skipped: no new cabin image; no inference.",
                            flush=True)
                    else:
                        cabin_image, image_captured_at = image_result
                        preview_img = cv2.imread(args.cabin_image_path)
                        if preview_img is not None:
                            cv2.imshow("In-Cabin Driver Monitor Preview", preview_img)
                            cv2.waitKey(1)

                        telemetry, sync_skew = nearest_telemetry(
                            history, image_captured_at)
                        if sync_skew > args.max_sync_skew:
                            raise RuntimeError(
                                "image/telemetry skew %.3fs exceeds %.3fs" %
                                (sync_skew, args.max_sync_skew))
                        observation_id += 1
                        observation = {
                            "protocol_version": PROTOCOL_VERSION,
                            "observation_id": observation_id,
                            "captured_at_unix_s": image_captured_at,
                            "telemetry_skew_ms": round(sync_skew * 1000.0, 2),
                            "telemetry": telemetry,
                            "driving_summary": summarize_telemetry(
                                history, args.telemetry_window, now_unix),
                            "cabin_image": cabin_image,
                        }
                        remote.submit_latest(observation)
                        print(
                            "Monitor queued: observation=%d telemetry_skew=%.1fms" %
                            (observation_id, sync_skew * 1000.0),
                            flush=True)
                except (OSError, RuntimeError, ValueError) as exc:
                    print(
                        "Monitor skipped: %s; no inference." % exc,
                        flush=True)
                next_image_time = now_monotonic + args.image_period

            if now_monotonic - last_status_time >= 2.0:
                result, error, dropped = remote.status()
                if result is None:
                    print("DRIVER MONITOR inactive  status=%s" % error, flush=True)
                else:
                    # Draw inference result text overlay if we have a frame
                    print(
                        "DRIVER MONITOR result  observation=%d risk=%s "
                        "forward=%s unsafe=%s confidence=%.2f age=%.0fms "
                        "dropped=%d reason=%s" % (
                            result["observation_id"], result["risk_level"],
                            result["looking_forward"], result["unsafe_behavior"],
                            result["confidence"], result["end_to_end_age_ms"],
                            dropped, result["reason"] or "-"),
                        flush=True)
                last_status_time = now_monotonic
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        remote.stop()
        print("Driver monitor stopped; no vehicle control was applied.", flush=True)


if __name__ == "__main__":
    main()
