#!/usr/bin/env python3

"""DGX-side HTTP service that turns CARLA telemetry into JSON controls.

No third-party web framework is required.  The OpenAI-compatible backend works
with local servers such as vLLM.  The Ollama backend is also supported.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional


PROTOCOL_VERSION = 1
MAX_REQUEST_BYTES = 4 * 1024 * 1024
MAX_IMAGE_BYTES = 2 * 1024 * 1024
SYSTEM_PROMPT = """You are a supervisory controller for a CARLA vehicle.
Return exactly one JSON object and no Markdown:
{"steer": number, "brake": number, "reason": "short text"}

Rules:
- steer must be between -1 and 1. Positive steer turns right.
- brake must be between 0 and 1.
- lane_offset_m is positive when the ego is right of lane center.
- heading_error_deg is vehicle yaw minus lane-center yaw.
- Prefer small, smooth steering corrections.
- Brake for close actors or unsafe road conditions. Otherwise use 0.
- You do not control throttle; the existing CARLA autopilot controls it.
"""

DRIVER_MONITOR_PROMPT = """You are a driver-monitoring observer for a CARLA vehicle.
You receive one in-cabin image captured at the same time as structured vehicle
telemetry. Return exactly one JSON object and no Markdown:
{"driver_response": "active|reduced|no_visible_response",
 "driver_present": boolean, "unconscious": boolean, "looking_forward": boolean,
 "eyes_closed": boolean, "phone_use": boolean,
 "unsafe_behavior": boolean,
 "risk_level": "normal|caution|high|unknown",
 "confidence": number, "reason": "short text"}

Rules:
- Base visual claims only on the supplied in-cabin image.
- Set unconscious=true only when the driver appears unable to respond or is
  clearly unconscious; otherwise set unconscious=false.
- Set driver_response=active for a visibly responsive driver,
  driver_response=reduced for distraction/drowsiness that still shows a
  voluntary response, and driver_response=no_visible_response only for clear
  unresponsiveness or a moving vehicle with no visible driver.
- Use telemetry only to understand whether the vehicle is moving, creeping, or
  temporarily stationary. A stationary vehicle may still be in an active
  driving session, for example at a red light.
- If the image is usable and a driver is visible, do not use risk_level
  unknown. Choose normal, caution, or high from the visual evidence.
- Use normal when the driver is present, eyes are open, no phone or object
  interaction is visible, and the driver appears to be looking forward or no
  unsafe behavior is visible.
- Use caution when the driver is distracted, looking away, using a phone, eyes
  appear closed, or another unsafe behavior is visible but the evidence is not
  severe or not certain.
- Use high when the vehicle is moving and the driver is clearly not looking
  forward, eyes are closed, phone use is clear, or unsafe behavior is clear.
- Use unknown only when the image is missing/unusable, the driver cannot be
  assessed, or the cabin view does not show enough of the driver to make a
  normal/caution/high decision.
- Do not produce throttle, steering, braking, or pull-over commands. A separate
  deterministic safety supervisor decides vehicle actions.
- confidence must be between 0 and 1.
"""

DRIVER_RISK_LEVELS = frozenset(("normal", "caution", "high", "critical", "unknown"))
DRIVER_RESPONSES = frozenset(("active", "reduced", "no_visible_response"))


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def require_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a number" % name)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("%s must be finite" % name)
    return result


def require_boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError("%s must be a boolean" % name)
    return value


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start < 0:
        raise ValueError("model output contains no JSON object")
    parsed, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(parsed, dict):
        raise ValueError("model output JSON must be an object")
    return parsed


def validate_telemetry(payload: Dict[str, Any]) -> None:
    if int(payload.get("protocol_version", -1)) != PROTOCOL_VERSION:
        raise ValueError("unsupported protocol_version")
    request_id = payload.get("request_id")
    if isinstance(request_id, bool) or not isinstance(request_id, int):
        raise ValueError("request_id must be an integer")
    if not isinstance(payload.get("vehicle"), dict):
        raise ValueError("vehicle must be a JSON object")
    if not isinstance(payload.get("road"), dict):
        raise ValueError("road must be a JSON object")


def validate_command(command: Dict[str, Any]) -> Dict[str, Any]:
    steer = require_number(command.get("steer"), "steer")
    brake = require_number(command.get("brake"), "brake")
    if not -1.0 <= steer <= 1.0:
        raise ValueError("steer must be in [-1, 1]")
    if not 0.0 <= brake <= 1.0:
        raise ValueError("brake must be in [0, 1]")
    return {
        "steer": round(steer, 5),
        "brake": round(brake, 5),
        "reason": str(command.get("reason", ""))[:200],
    }


def validate_driver_assessment(assessment: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(assessment, dict):
        raise ValueError("driver assessment must be a JSON object")
    risk_level = str(assessment.get("risk_level", "unknown")).lower()
    if risk_level not in DRIVER_RISK_LEVELS:
        raise ValueError(
            "risk_level must be one of %s" % sorted(DRIVER_RISK_LEVELS))
    confidence = require_number(assessment.get("confidence"), "confidence")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    driver_present = require_boolean(
        assessment.get("driver_present"), "driver_present")
    unconscious = require_boolean(
        assessment.get("unconscious", False), "unconscious")
    response = str(assessment.get("driver_response", "")).strip().lower()
    if not response:
        # Backward compatibility for a DGX worker that has not yet received
        # the new prompt. Unknown observations remain invalid and cannot
        # trigger or release a minimal-risk manoeuvre.
        if unconscious:
            response = "no_visible_response"
        elif risk_level == "normal":
            response = "active"
        else:
            response = "reduced"
    if response not in DRIVER_RESPONSES:
        raise ValueError(
            "driver_response must be one of %s" % sorted(DRIVER_RESPONSES))
    return {
        "assessment_valid": risk_level != "unknown",
        "driver_response": response,
        "driver_present": driver_present,
        "unconscious": unconscious,
        "looking_forward": require_boolean(
            assessment.get("looking_forward"), "looking_forward"),
        "eyes_closed": require_boolean(
            assessment.get("eyes_closed"), "eyes_closed"),
        "phone_use": require_boolean(
            assessment.get("phone_use"), "phone_use"),
        "unsafe_behavior": require_boolean(
            assessment.get("unsafe_behavior"), "unsafe_behavior"),
        "risk_level": risk_level,
        "confidence": round(confidence, 5),
        "reason": str(assessment.get("reason", ""))[:200],
    }


def validate_cabin_image(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("cabin_image must be a JSON object")
    mime_type = str(value.get("mime_type", "")).lower()
    if mime_type not in ("image/jpeg", "image/png", "image/webp"):
        raise ValueError("cabin_image.mime_type must be JPEG, PNG, or WebP")
    encoded = value.get("data_base64")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("cabin_image.data_base64 is required")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("cabin_image.data_base64 is invalid") from exc
    if not decoded:
        raise ValueError("cabin_image is empty")
    if len(decoded) > MAX_IMAGE_BYTES:
        raise ValueError("cabin_image exceeds %d bytes" % MAX_IMAGE_BYTES)
    signatures_valid = {
        "image/jpeg": decoded.startswith(b"\xff\xd8\xff"),
        "image/png": decoded.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/webp": (
            len(decoded) >= 12
            and decoded.startswith(b"RIFF")
            and decoded[8:12] == b"WEBP"),
    }
    if not signatures_valid[mime_type]:
        raise ValueError("cabin_image bytes do not match mime_type")
    return {"mime_type": mime_type, "data_base64": encoded}


class DecisionEngine:
    def __init__(self, backend: str, model: str, llm_url: str,
                 llm_api_key: str, llm_timeout: float) -> None:
        self.backend = backend
        self.model = model
        self.llm_url = llm_url
        self.llm_api_key = llm_api_key
        self.llm_timeout = llm_timeout
        self._inference_lock = threading.Lock()

    def decide(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy telemetry-only control inference."""
        if self.backend == "dry-run":
            return self._dry_run_decision(telemetry)
        compact_telemetry = json.dumps(
            telemetry, ensure_ascii=False, separators=(",", ":"))
        with self._inference_lock:
            if self.backend == "openai":
                model_text = self._call_openai_compatible(
                    SYSTEM_PROMPT, compact_telemetry, 120)
            elif self.backend == "ollama":
                model_text = self._call_ollama(
                    SYSTEM_PROMPT, compact_telemetry, None)
            else:
                raise RuntimeError("unknown backend: %s" % self.backend)
        return validate_command(extract_json_object(model_text))

    def assess_driver(self, observation: Dict[str, Any],
                      cabin_image: Dict[str, str]) -> Dict[str, Any]:
        """Run multimodal inference; callers must provide a validated image."""
        if self.backend == "dry-run":
            return validate_driver_assessment({
                "driver_present": True,
                "looking_forward": True,
                "eyes_closed": False,
                "phone_use": False,
                "unsafe_behavior": False,
                "risk_level": "normal",
                "confidence": 1.0,
                "reason": "dry-run monitor response",
            })

        model_context = {
            key: value for key, value in observation.items()
            if key != "cabin_image"
        }
        context_json = json.dumps(
            model_context, ensure_ascii=False, separators=(",", ":"))
        with self._inference_lock:
            if self.backend == "openai":
                user_content = [
                    {"type": "text", "text": context_json},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:%s;base64,%s" % (
                                cabin_image["mime_type"],
                                cabin_image["data_base64"]),
                        },
                    },
                ]
                model_text = self._call_openai_compatible(
                    DRIVER_MONITOR_PROMPT, user_content, 180)
            elif self.backend == "ollama":
                model_text = self._call_ollama(
                    DRIVER_MONITOR_PROMPT,
                    context_json,
                    [cabin_image["data_base64"]])
            else:
                raise RuntimeError("unknown backend: %s" % self.backend)
        return validate_driver_assessment(extract_json_object(model_text))

    @staticmethod
    def _dry_run_decision(telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Deterministic controller for checking the network before loading an LLM."""
        road = telemetry.get("road", {})
        lane_offset = float(road.get("lane_offset_m", 0.0))
        heading_error = float(road.get("heading_error_deg", 0.0))
        steer = clamp(-0.28 * lane_offset - 0.018 * heading_error, -0.45, 0.45)

        brake = 0.0
        reason = "dry-run lane correction"
        front_actors = [
            actor for actor in telemetry.get("nearby_vehicles", [])
            + telemetry.get("nearby_walkers", [])
            if 0.0 < float(actor.get("longitudinal_m", -1.0))
            and abs(float(actor.get("lateral_m", 999.0))) < 2.2]
        if front_actors:
            nearest = min(front_actors, key=lambda actor: actor["longitudinal_m"])
            distance = float(nearest["longitudinal_m"])
            if distance < 4.0:
                brake = 1.0
                reason = "dry-run emergency obstacle brake"
            elif distance < 8.0:
                brake = 0.45
                reason = "dry-run obstacle slowdown"
        return validate_command({
            "steer": steer,
            "brake": brake,
            "reason": reason,
        })

    def _call_openai_compatible(self, system_prompt: str,
                                user_content: Any,
                                max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        response = self._post_to_model(payload)
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("unexpected OpenAI-compatible response") from exc
        if isinstance(content, dict):
            return json.dumps(content)
        return str(content)

    def _call_ollama(self, system_prompt: str, user_content: str,
                     images: Optional[List[str]]) -> str:
        user_message = {"role": "user", "content": user_content}
        if images:
            user_message["images"] = images
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system_prompt},
                user_message,
            ],
            "options": {"temperature": 0.0},
        }
        response = self._post_to_model(payload)
        try:
            content = response["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise ValueError("unexpected Ollama response") from exc
        if isinstance(content, dict):
            return json.dumps(content)
        return str(content)

    def _post_to_model(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.llm_api_key:
            headers["Authorization"] = "Bearer " + self.llm_api_key
        request = urllib.request.Request(
            self.llm_url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(
                    request, timeout=self.llm_timeout) as response:
                response_body = response.read(MAX_REQUEST_BYTES + 1)
        except urllib.error.HTTPError as exc:
            details = exc.read(2048).decode("utf-8", errors="replace")
            raise RuntimeError(
                "model endpoint HTTP %d: %s" % (exc.code, details))
        if len(response_body) > MAX_REQUEST_BYTES:
            raise ValueError("model response is too large")
        parsed = json.loads(response_body.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("model endpoint response must be a JSON object")
        return parsed


class ControlRequestHandler(BaseHTTPRequestHandler):
    server_version = "CarlaDriverMonitor/2"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            telemetry_age = self.server.telemetry_age_sec()
            self._write_json(200, {
                "ok": True,
                "protocol_version": PROTOCOL_VERSION,
                "backend": self.server.engine.backend,
                "inference_busy": self.server.inference_slot.locked(),
                "telemetry_available": telemetry_age is not None,
                "telemetry_age_s": telemetry_age,
                "legacy_control_enabled": self.server.enable_legacy_control,
                "endpoints": {
                    "telemetry": "POST /telemetry (cache only; no inference)",
                    "telemetry_latest": "GET /telemetry/latest (authenticated debug view)",
                    "monitor": "POST /monitor (cabin image required)",
                    "monitor_complete": "POST /monitor/complete (authenticated; no inference)",
                    "driver_state_latest": "GET /driver_state/latest (authenticated)",
                },
                "time_unix_s": time.time(),
            })
        elif self.path == "/telemetry/latest":
            if not self._authenticated():
                self._write_json(401, {"error": "unauthorized"})
                return
            telemetry = self.server.latest_telemetry()
            if telemetry is None:
                self._write_json(404, {
                    "error": "telemetry unavailable or stale",
                })
                return
            self._write_json(200, {
                "ok": True,
                "telemetry_age_s": self.server.telemetry_age_sec(),
                "telemetry": telemetry,
            })
        elif self.path == "/driver_state/latest":
            if not self._authenticated():
                self._write_json(401, {"error": "unauthorized"})
                return
            result = self.server.latest_driver_state()
            if result is None:
                self._write_json(404, {
                    "error": "driver state unavailable or stale",
                })
                return
            self._write_json(200, result)
        else:
            self._write_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in (
                "/telemetry", "/monitor", "/monitor/complete", "/control"):
            self._write_json(404, {"error": "not found"})
            return
        if not self._authenticated():
            self._write_json(401, {"error": "unauthorized"})
            return
        try:
            payload = self._read_json()
            if self.path == "/telemetry":
                self._handle_telemetry(payload)
            elif self.path == "/monitor":
                self._handle_monitor(payload)
            elif self.path == "/monitor/complete":
                self._handle_monitor_complete(payload)
            else:
                self._handle_legacy_control(payload)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            self._write_json(400, {"error": str(exc)})
        except Exception as exc:
            print("INFERENCE ERROR  %s: %s" % (
                type(exc).__name__, exc), flush=True)
            self._write_json(502, {
                "error": "model inference failed",
                "detail": str(exc)[:500],
            })

    def _authenticated(self) -> bool:
        expected_token = self.server.shared_token
        if not expected_token:
            return True
        supplied = self.headers.get("Authorization", "")
        return supplied == "Bearer " + expected_token

    def _read_json(self) -> Dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            raise ValueError("invalid Content-Length")
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request JSON must be an object")
        return payload

    def _handle_telemetry(self, payload: Dict[str, Any]) -> None:
        validate_telemetry(payload)
        self.server.store_telemetry(payload)
        self._write_json(202, {
            "ok": True,
            "request_id": payload["request_id"],
            "inference_started": False,
            "message": "telemetry cached; waiting for cabin image",
        })

    def _handle_monitor(self, payload: Dict[str, Any]) -> None:
        if "cabin_image" not in payload:
            self._write_json(422, {
                "error": "cabin_image is required; inference was not started",
                "inference_started": False,
            })
            return
        observation_id = payload.get("observation_id")
        if isinstance(observation_id, bool) or not isinstance(observation_id, int):
            raise ValueError("observation_id must be an integer")
        if int(payload.get("protocol_version", -1)) != PROTOCOL_VERSION:
            raise ValueError("unsupported protocol_version")
        if "captured_at_unix_s" in payload:
            require_number(payload["captured_at_unix_s"], "captured_at_unix_s")
        cabin_image = validate_cabin_image(payload["cabin_image"])
        # A new observation starts a new finite dataset/live-monitor session.
        # Clear an earlier completion marker before publishing its result.
        self.server.mark_monitoring_started()

        telemetry = payload.get("telemetry")
        if telemetry is None:
            telemetry = self.server.latest_telemetry()
            if telemetry is None:
                self._write_json(409, {
                    "error": "fresh telemetry is unavailable",
                    "inference_started": False,
                })
                return
        if not isinstance(telemetry, dict):
            raise ValueError("telemetry must be a JSON object")
        validate_telemetry(telemetry)
        self.server.store_telemetry(telemetry)

        if not self.server.inference_slot.acquire(blocking=False):
            self._write_json(429, {
                "error": "inference busy; submit a newer observation later",
                "inference_started": False,
            })
            return
        try:
            observation = dict(payload)
            observation["telemetry"] = telemetry
            observation["cabin_image"] = cabin_image
            started = time.monotonic()
            driver_state = self.server.engine.assess_driver(
                observation, cabin_image)
            elapsed_ms = round((time.monotonic() - started) * 1000.0, 2)
            result = {
                "protocol_version": PROTOCOL_VERSION,
                "observation_id": observation_id,
                "driver_state": driver_state,
                "inference_started": True,
                "inference_ms": elapsed_ms,
            }
            self.server.store_driver_state({
                "protocol_version": PROTOCOL_VERSION,
                "observation_id": observation_id,
                "driver_state": driver_state,
                "inference_ms": elapsed_ms,
                "received_at_unix_s": time.time(),
                "cabin_image": cabin_image,
            })
            self._write_json(200, result)
        finally:
            self.server.inference_slot.release()

    def _handle_monitor_complete(self, payload: Dict[str, Any]) -> None:
        """Publish an inference-free end-of-dataset marker to the supervisor."""
        if int(payload.get("protocol_version", -1)) != PROTOCOL_VERSION:
            raise ValueError("unsupported protocol_version")
        observation_id = payload.get("last_observation_id")
        if isinstance(observation_id, bool) or not isinstance(observation_id, int):
            raise ValueError("last_observation_id must be an integer")
        if observation_id < 0:
            raise ValueError("last_observation_id must be non-negative")
        if "completed_at_unix_s" in payload:
            require_number(payload["completed_at_unix_s"], "completed_at_unix_s")
        if not self.server.mark_monitoring_complete(observation_id):
            self._write_json(409, {
                "error": "no matching driver observation is available",
                "monitoring_complete": False,
            })
            return
        self._write_json(202, {
            "ok": True,
            "monitoring_complete": True,
            "last_observation_id": observation_id,
            "inference_started": False,
        })

    def _handle_legacy_control(self, payload: Dict[str, Any]) -> None:
        if not self.server.enable_legacy_control:
            self._write_json(410, {
                "error": "telemetry-only control inference is disabled",
                "inference_started": False,
            })
            return
        validate_telemetry(payload)
        if not self.server.inference_slot.acquire(blocking=False):
            self._write_json(429, {
                "error": "inference busy",
                "inference_started": False,
            })
            return
        try:
            started = time.monotonic()
            command = self.server.engine.decide(payload)
            elapsed_ms = round((time.monotonic() - started) * 1000.0, 2)
            self._write_json(200, {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": payload["request_id"],
                "command": command,
                "valid_for_ms": self.server.valid_for_ms,
                "inference_started": True,
                "inference_ms": elapsed_ms,
            })
        finally:
            self.server.inference_slot.release()

    def log_message(self, format_string: str, *args: Any) -> None:
        print("%s - %s" % (self.address_string(), format_string % args), flush=True)

    def _write_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


class ControlServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: Any, engine: DecisionEngine,
                 shared_token: str, valid_for_ms: int,
                 telemetry_max_age_sec: float = 5.0,
                 driver_state_max_age_sec: float = 60.0,
                 enable_legacy_control: bool = False) -> None:
        super().__init__(address, ControlRequestHandler)
        self.engine = engine
        # Tokens are hex/base64-like secrets; tolerate a pasted trailing newline
        # without ever putting that newline into an HTTP header.
        self.shared_token = shared_token.strip()
        self.valid_for_ms = valid_for_ms
        self.telemetry_max_age_sec = telemetry_max_age_sec
        self.driver_state_max_age_sec = driver_state_max_age_sec
        self.enable_legacy_control = enable_legacy_control
        self.inference_slot = threading.Lock()
        self._telemetry_lock = threading.Lock()
        self._latest_telemetry: Optional[Dict[str, Any]] = None
        self._telemetry_received_at = 0.0
        self._latest_driver_state: Optional[Dict[str, Any]] = None
        self._driver_state_received_at = 0.0
        self._monitoring_complete = False
        self._completed_observation_id = -1
        self._monitoring_completed_at = 0.0

    def store_telemetry(self, telemetry: Dict[str, Any]) -> None:
        with self._telemetry_lock:
            self._latest_telemetry = telemetry
            self._telemetry_received_at = time.monotonic()

    def telemetry_age_sec(self) -> Optional[float]:
        with self._telemetry_lock:
            if self._latest_telemetry is None:
                return None
            return round(time.monotonic() - self._telemetry_received_at, 5)

    def latest_telemetry(self) -> Optional[Dict[str, Any]]:
        with self._telemetry_lock:
            if self._latest_telemetry is None:
                return None
            age = time.monotonic() - self._telemetry_received_at
            if age > self.telemetry_max_age_sec:
                return None
            return self._latest_telemetry

    def store_driver_state(self, result: Dict[str, Any]) -> None:
        with self._telemetry_lock:
            self._latest_driver_state = result
            self._driver_state_received_at = time.monotonic()

    def mark_monitoring_started(self) -> None:
        with self._telemetry_lock:
            self._monitoring_complete = False
            self._completed_observation_id = -1
            self._monitoring_completed_at = 0.0

    def mark_monitoring_complete(self, observation_id: int) -> bool:
        with self._telemetry_lock:
            latest = self._latest_driver_state
            if latest is None:
                return False
            latest_id = latest.get("observation_id")
            if latest_id != observation_id:
                return False
            self._monitoring_complete = True
            self._completed_observation_id = observation_id
            self._monitoring_completed_at = time.time()
            return True

    def latest_driver_state(self) -> Optional[Dict[str, Any]]:
        with self._telemetry_lock:
            if self._latest_driver_state is None:
                return None
            age = time.monotonic() - self._driver_state_received_at
            if age > self.driver_state_max_age_sec:
                return None
            result = dict(self._latest_driver_state)
            result["result_age_s"] = round(age, 5)
            result["monitoring_complete"] = self._monitoring_complete
            if self._monitoring_complete:
                result["completed_observation_id"] = (
                    self._completed_observation_id)
                result["monitoring_completed_at_unix_s"] = (
                    self._monitoring_completed_at)
            return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DGX CARLA multimodal driver-monitoring bridge")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--backend", choices=("dry-run", "openai", "ollama"),
        default="dry-run")
    parser.add_argument(
        "--model", default="mistralai/Ministral-3-8B-Instruct-2512")
    parser.add_argument(
        "--llm-url",
        default="http://127.0.0.1:8001/v1/chat/completions",
        help="OpenAI chat-completions URL or Ollama /api/chat URL")
    parser.add_argument(
        "--llm-api-key", default=os.environ.get("LLM_API_KEY", ""))
    parser.add_argument(
        "--token", default=os.environ.get("LLM_BRIDGE_TOKEN", ""),
        help="shared bearer token (or set LLM_BRIDGE_TOKEN)")
    parser.add_argument("--llm-timeout", type=float, default=10.0)
    parser.add_argument("--valid-for-ms", type=int, default=1500)
    parser.add_argument(
        "--telemetry-max-age", type=float, default=5.0,
        help="maximum cached telemetry age accepted by /monitor")
    parser.add_argument(
        "--driver-state-max-age", type=float, default=60.0,
        help="maximum cached driver_state age returned by /driver_state/latest")
    parser.add_argument(
        "--enable-legacy-control", action="store_true",
        help="allow legacy telemetry-only POST /control inference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.token = args.token.strip()
    args.llm_api_key = args.llm_api_key.strip()
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be in [1, 65535]")
    if args.llm_timeout <= 0.0:
        raise ValueError("--llm-timeout must be positive")
    if args.telemetry_max_age <= 0.0:
        raise ValueError("--telemetry-max-age must be positive")
    if args.driver_state_max_age <= 0.0:
        raise ValueError("--driver-state-max-age must be positive")
    if not 100 <= args.valid_for_ms <= 5000:
        raise ValueError("--valid-for-ms must be in [100, 5000]")
    engine = DecisionEngine(
        args.backend, args.model, args.llm_url,
        args.llm_api_key, args.llm_timeout)
    server = ControlServer(
        (args.host, args.port), engine, args.token, args.valid_for_ms,
        telemetry_max_age_sec=args.telemetry_max_age,
        driver_state_max_age_sec=args.driver_state_max_age,
        enable_legacy_control=args.enable_legacy_control)
    print(
        "DGX DRIVER MONITOR READY  http://%s:%d backend=%s model=%s" % (
            args.host, args.port, args.backend, args.model),
        flush=True)
    print(
        "Telemetry cache: POST /telemetry  Multimodal inference: POST /monitor",
        flush=True)
    if args.enable_legacy_control:
        print("Legacy telemetry-only inference: POST /control", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("DGX driver monitor stopped.", flush=True)


if __name__ == "__main__":
    main()
