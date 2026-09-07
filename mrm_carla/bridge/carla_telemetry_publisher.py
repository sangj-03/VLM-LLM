#!/usr/bin/env python3

"""Publish live CARLA telemetry to DGX without starting any LLM inference."""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from carla_llm_overlay import TelemetryBuilder, wait_for_vehicle, wait_for_world


class TelemetryPublisher:
    """Latest-only async publisher; POST /telemetry is cache-only on DGX."""

    def __init__(self, server_url: str, token: str, timeout_sec: float) -> None:
        self.server_url = server_url
        self.token = token
        self.timeout_sec = timeout_sec
        self._queue = queue.Queue(maxsize=1)  # type: queue.Queue
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_status = "waiting for telemetry"
        self._thread = threading.Thread(
            target=self._worker, name="telemetry-publisher", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=self.timeout_sec + 0.5)

    def submit_latest(self, telemetry: Dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(telemetry)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(telemetry)
        except queue.Full:
            pass

    def status(self) -> str:
        with self._lock:
            return self._last_status

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                telemetry = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if telemetry is None:
                continue
            try:
                result = self._post(telemetry)
                if result.get("inference_started") is not False:
                    raise RuntimeError(
                        "DGX did not confirm cache-only telemetry handling")
                with self._lock:
                    self._last_status = (
                        "sent request_id=%s inference_started=false" %
                        telemetry.get("request_id"))
            except Exception as exc:
                with self._lock:
                    self._last_status = "%s: %s" % (type(exc).__name__, exc)

    def _post(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(telemetry, separators=(",", ":")).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "carla-telemetry-publisher/1",
        }
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        request = urllib.request.Request(
            self.server_url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(
                    request, timeout=self.timeout_sec) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read(2048).decode("utf-8", errors="replace")
            raise RuntimeError("DGX HTTP %d: %s" % (exc.code, details))
        if not isinstance(parsed, dict):
            raise ValueError("DGX response must be a JSON object")
        return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send CARLA telemetry to DGX cache without LLM inference")
    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--role-name", default="town06_ego")
    parser.add_argument(
        "--server-url", default="http://127.0.0.1:8000/telemetry")
    parser.add_argument(
        "--token", default=os.environ.get("LLM_BRIDGE_TOKEN", "").strip())
    parser.add_argument("--send-hz", type=float, default=10.0)
    parser.add_argument("--request-timeout", type=float, default=3.0)
    parser.add_argument("--nearby-radius", type=float, default=50.0)
    parser.add_argument("--max-nearby-actors", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.send_hz <= 0.0:
        raise ValueError("--send-hz must be positive")
    _, world = wait_for_world(args.carla_host, args.carla_port)
    vehicle = wait_for_vehicle(world, args.role_name)
    builder = TelemetryBuilder(
        world, vehicle, args.role_name,
        args.nearby_radius, args.max_nearby_actors)
    publisher = TelemetryPublisher(
        args.server_url, args.token, args.request_timeout)
    publisher.start()

    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    print(
        "TELEMETRY PUBLISHER READY actor_id=%d server=%s send_hz=%.2f" % (
            vehicle.id, args.server_url, args.send_hz),
        flush=True)
    print("This process never calls LLM inference.", flush=True)

    request_id = 0
    next_send = 0.0
    period = 1.0 / args.send_hz
    last_status = 0.0
    try:
        while not stop_requested and vehicle.is_alive:
            world.wait_for_tick(10.0)
            now = time.monotonic()
            if now >= next_send:
                request_id += 1
                try:
                    publisher.submit_latest(builder.build(request_id))
                except RuntimeError as exc:
                    print("Telemetry skipped: %s" % exc, flush=True)
                next_send = now + period
            if now - last_status >= 2.0:
                print("DGX telemetry: %s" % publisher.status(), flush=True)
                last_status = now
    finally:
        publisher.stop()
        print("Telemetry publisher stopped; no inference was requested.", flush=True)


if __name__ == "__main__":
    main()
