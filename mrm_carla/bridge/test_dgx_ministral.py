#!/usr/bin/env python3

"""Check vLLM, telemetry gating, and optional image inference on DGX Spark."""

import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request


VLLM_PORT = int(os.environ.get("VLLM_HOST_PORT", "8001"))
BRIDGE_PORT = int(os.environ.get("BRIDGE_PORT", "8000"))
BRIDGE_TOKEN = os.environ.get("LLM_BRIDGE_TOKEN", "").strip()
CABIN_IMAGE_PATH = os.environ.get("CABIN_IMAGE_PATH", "")


def get_json(url):
    with urllib.request.urlopen(url, timeout=3.0) as response:
        return json.loads(response.read().decode("utf-8"))


def check_health(url):
    with urllib.request.urlopen(url, timeout=3.0) as response:
        response.read()
        if response.status != 200:
            raise RuntimeError("health endpoint returned HTTP %d" % response.status)


def post_json(url, payload, token):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15.0) as response:
        return json.loads(response.read().decode("utf-8"))


def sample_telemetry():
    return {
        "protocol_version": 1,
        "request_id": 1,
        "simulation": {"map": "Town04_Opt", "frame": 1},
        "vehicle": {
            "role_name": "town06_ego",
            "speed_kph": 20.0,
            "autopilot_control": {
                "throttle": 0.4,
                "steer": 0.0,
                "brake": 0.0,
            },
        },
        "road": {
            "on_driving_lane": True,
            "lane_offset_m": 0.2,
            "heading_error_deg": 1.0,
            "curvature_rad_per_m": 0.0,
        },
        "traffic_light": {
            "affecting_vehicle": False,
            "state": "Unknown",
        },
        "nearby_vehicles": [],
        "nearby_walkers": [],
    }


def monitor_observation(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type not in ("image/jpeg", "image/png", "image/webp"):
        raise ValueError("CABIN_IMAGE_PATH must be JPEG, PNG, or WebP")
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read(2 * 1024 * 1024 + 1)
    if not image_bytes or len(image_bytes) > 2 * 1024 * 1024:
        raise ValueError("cabin image must be between 1 byte and 2 MiB")
    return {
        "protocol_version": 1,
        "observation_id": 1,
        "captured_at_unix_s": time.time(),
        "telemetry": sample_telemetry(),
        "driving_summary": {
            "motion_state": "moving",
            "current_speed_kph": 20.0,
        },
        "cabin_image": {
            "mime_type": mime_type,
            "data_base64": base64.b64encode(image_bytes).decode("ascii"),
        },
    }


def main():
    if not BRIDGE_TOKEN:
        print("ERROR: LLM_BRIDGE_TOKEN is not set.", file=sys.stderr)
        return 2

    try:
        check_health("http://127.0.0.1:%d/health" % VLLM_PORT)
        print("OK: vLLM health endpoint")

        bridge_health = get_json("http://127.0.0.1:%d/health" % BRIDGE_PORT)
        print("OK: bridge health endpoint, backend=%s" % bridge_health["backend"])

        telemetry_result = post_json(
            "http://127.0.0.1:%d/telemetry" % BRIDGE_PORT,
            sample_telemetry(),
            BRIDGE_TOKEN,
        )
        if telemetry_result.get("inference_started") is not False:
            raise RuntimeError("telemetry unexpectedly started inference")
        print("OK: telemetry cached without inference")

        if not CABIN_IMAGE_PATH:
            print(
                "SKIP: set CABIN_IMAGE_PATH to run multimodal inference",
                file=sys.stderr)
            return 0

        result = post_json(
            "http://127.0.0.1:%d/monitor" % BRIDGE_PORT,
            monitor_observation(CABIN_IMAGE_PATH),
            BRIDGE_TOKEN,
        )
        driver_state = result["driver_state"]
        print("OK: Ministral multimodal driver-monitor response")
        print(json.dumps(driver_state, ensure_ascii=False, indent=2))
        return 0
    except (OSError, RuntimeError, urllib.error.URLError, ValueError, KeyError,
            json.JSONDecodeError) as exc:
        print("FAILED: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
