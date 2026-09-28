#!/usr/bin/env python3
"""Upper-body-only driver monitor.

The /monitor endpoint returns exactly five classification fields:
- eye_state
- head_pose
- upper_body_posture
- hand_on_wheel
- driver_response

This version intentionally does NOT output gaze, driver_state, risk level,
unconscious, or vehicle-control decisions.

When a validated TCN event accompanies a request, its probability and temporal
interval are provided to Qwen as auxiliary context.  The visual prompt and
three-class decision rules remain authoritative.
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
MAX_REQUEST_BYTES = 32 * 1024 * 1024
MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_VIDEO_BYTES = 20 * 1024 * 1024
MAX_TCN_EVENTS = 64
EVENT_RESPONSE_CONTINUITY_SEC = 4.0
# Event clips overlap, but they are not a dense frame-by-frame stream. Two
# consecutive, independently recovered observations are enough to clear a
# stale no-visible-response latch.
EVENT_ACTIVE_RECOVERY_CONFIRMATIONS = 2

EYE_STATES = frozenset(("open", "closed", "uncertain"))
HEAD_POSES = frozenset(("upright", "dropped"))
BODY_POSTURES = frozenset(("upright", "forward_lean", "side_lean"))
HAND_CONTACT = frozenset(("confirmed", "not_confirmed"))
VOLUNTARY_MOTION = frozenset(("clear", "weak", "none", "uncertain"))
RECOVERY_STATUS = frozenset(
    ("not_needed", "clear", "partial", "none", "uncertain")
)
DRIVER_RESPONSES = frozenset(
    ("active", "reduced", "no_visible_response")
)

# The production prompt below is the single source of truth.
FAST_DRIVER_MONITOR_PROMPT = """Classify only the driver. The supplied media may be a full driver view, an enlarged driver evidence view, a video, or a 2x2 temporal storyboard image. A driver bounding-box crop can still contain both occupants. In every view, classify only the red-clothed person seated on the right as the driver; the brown-clothed person seated on the left is the passenger and must not contribute evidence to any driver field. In a storyboard, read all four panels in timestamp order: top-left, top-right, bottom-left, bottom-right. Use full-cabin panels only to establish wheel/control visibility and passenger external contact. Passenger touching the wheel does not mean the driver is active.
Analyze every available video frame or storyboard panel, not just the last or clearest-looking panel. A normal-looking panel before or after a collapse must not erase the collapse. Return exactly one JSON object with exactly these seven keys and no other text:
{"eye_state":"open|closed|uncertain","head_pose":"upright|dropped","upper_body_posture":"upright|forward_lean|side_lean","hand_on_wheel":"confirmed|not_confirmed","voluntary_motion":"clear|weak|none|uncertain","recovery_status":"not_needed|clear|partial|none|uncertain","driver_response":"active|reduced|no_visible_response"}

Report eye_state, head_pose, and upper_body_posture from the END of the clip. Passenger touching, holding, lifting, shaking, or repositioning is external support, not an action or recovery by the driver.
- eye_state: open, closed, or uncertain. Use closed only when the driver's eyes are visibly closed; narrow or dark eyes alone are not closed. Use uncertain when occlusion, lighting, image quality, or viewing angle prevents judging whether the driver's eyes are open or closed, including when the driver's head is turned and their eyes are not visible.
- head_pose: upright or dropped. Use upright only when the driver's head is normally self-supported and roughly vertical. Use dropped for ANY non-upright head angle, including a forward, backward, left, or right sag, hang, or tilt. A brief supported glance or turn that keeps the head upright is not dropped.
- upper_body_posture: upright when supported; forward_lean or side_lean only for substantial collapse or slump. A mild controlled lean is not collapse.
- hand_on_wheel: use confirmed when at least one driver's hand is visibly gripping, touching, or actively operating the steering wheel or controls, OR when the driver's wrist or forearm visibly extends continuously into the wheel/control area and the hand endpoint is occluded there with no visible gap. Use not_confirmed when the driver's hand or arm is visibly away from the wheel/control area, or when the hand endpoint, arm direction, wheel, or contact cannot be judged.
- voluntary_motion: clear only when a driver-initiated action is visibly certain across frames: the driver turns or operates a wheel/control, deliberately moves a hand, or independently repositions their own head or torso. weak when movement is visible but driver initiation is not certain; none when no driver-initiated change is visible anywhere in the clip; uncertain when the driver is too occluded to judge. Ignore gravity, camera motion, and passive movement caused by a passenger.
- recovery_status: not_needed when no collapse or slump occurs anywhere in the clip; clear when, after a collapse, the driver independently returns to supported upright posture and shows clear voluntary motion; partial when the driver independently moves toward recovery but remains unstable; none when a collapse occurs and no independent recovery is visible; uncertain only when the clip cannot establish this.

Set driver_response from your six observations in this exact priority order:
- no_visible_response when ANY one condition is true: (1) voluntary_motion is none OR recovery_status is none; or (2) at least TWO of these are true: eye_state is closed, head_pose is dropped, and upper_body_posture is forward_lean or side_lean. This label means either no driver-initiated movement is visible, no independent recovery is visible, or at least two of the configured collapse cues are present.
- active only when ALL conditions are true: eye_state is open; head_pose is upright; upper_body_posture is upright; hand_on_wheel is confirmed; voluntary_motion is clear; and recovery_status is not_needed. This label means that, across the complete clip, the driver independently supports their own head and upper body, visibly performs a clear intentional control action or movement while hand contact with the wheel or controls is confirmed, and has no collapse or recovery episode.
- reduced for every other combination, including any uncertain value, voluntary_motion weak, or recovery_status clear or partial. After the no_visible_response rule above has been checked, use reduced when hand_on_wheel is not_confirmed AND exactly ONE of these is true: eye_state is closed, head_pose is dropped, or upper_body_posture is forward_lean or side_lean. This label means the driver shows some self-initiated movement, self-support, or recovery attempt, but control is diminished, delayed, unstable, incomplete, or otherwise does not meet the active standard.

Do not diagnose consciousness or output reasons, confidence, gaze, risk, intermediate states, or extra fields."""


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
        fragment = text
    else:
        fragment = text[start:]
    try:
        parsed, _ = json.JSONDecoder().raw_decode(fragment)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # No second inference: recover the enum fields directly from a JSON
    # string that was cut before its closing quote or brace.
    recovered: Dict[str, Any] = {}
    allowed_by_field = {
        "eye_state": EYE_STATES,
        "head_pose": HEAD_POSES,
        "upper_body_posture": BODY_POSTURES,
        "hand_on_wheel": HAND_CONTACT,
        "voluntary_motion": VOLUNTARY_MOTION,
        "recovery_status": RECOVERY_STATUS,
        "driver_response": DRIVER_RESPONSES,
    }
    for name, allowed_values in allowed_by_field.items():
        match = re.search(
            r'"' + re.escape(name) + r'"\s*:\s*"([^"\r\n}]*)',
            fragment,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        raw_value = match.group(1).strip().lower().replace("-", "_")
        if raw_value in allowed_values:
            recovered[name] = raw_value
            continue
        prefix_matches = [
            value for value in allowed_values if value.startswith(raw_value)
        ]
        if raw_value and len(prefix_matches) == 1:
            recovered[name] = prefix_matches[0]

    # Validation below fills any field that could not be recovered. Returning a
    # partial object is preferable to aborting a long batch evaluation.
    return recovered


def validate_driver_assessment(assessment: Dict[str, Any]) -> Dict[str, str]:
    if not isinstance(assessment, dict):
        raise ValueError("driver assessment must be a JSON object")

    values = {
        "eye_state": str(assessment.get("eye_state", "")).strip().lower(),
        "head_pose": str(assessment.get("head_pose", "")).strip().lower(),
        "upper_body_posture": str(
            assessment.get("upper_body_posture", "")
        ).strip().lower(),
        "hand_on_wheel": str(
            assessment.get("hand_on_wheel", "")
        ).strip().lower(),
        "voluntary_motion": str(
            assessment.get("voluntary_motion", "")
        ).strip().lower(),
        "recovery_status": str(
            assessment.get("recovery_status", "")
        ).strip().lower(),
        "driver_response": str(
            assessment.get("driver_response", "")
        ).strip().lower(),
    }

    aliases = {
        "eye_state": {
            "opened": "open",
            "eyes_open": "open",
            "partially_open": "open",
            "half_open": "open",
            "closed_eyes": "closed",
            "eyes_closed": "closed",
            "partially_closed": "closed",
            "half_closed": "closed",
            "unknown": "uncertain",
            "not_visible": "uncertain",
        },
        "head_pose": {
            "normal": "upright",
            "straight": "upright",
            "raised": "upright",
            "down": "dropped",
            "downward": "dropped",
            "leaning_down": "dropped",
            "tilted": "dropped",
            "left_tilt": "dropped",
            "right_tilt": "dropped",
        },
        "upper_body_posture": {
            "normal": "upright",
            "straight": "upright",
            "leaning_forward": "forward_lean",
            "forward": "forward_lean",
            "dropped": "forward_lean",
            "slumped": "forward_lean",
            "collapsed": "forward_lean",
            "leaning_side": "side_lean",
            "left_lean": "side_lean",
            "right_lean": "side_lean",
            "tilted": "side_lean",
        },
        "hand_on_wheel": {
            "yes": "confirmed",
            "holding": "confirmed",
            "on_wheel": "confirmed",
            "gripping": "confirmed",
            "no": "not_confirmed",
            "off_wheel": "not_confirmed",
            "not_holding": "not_confirmed",
            "unknown": "not_confirmed",
            "not_visible": "not_confirmed",
        },
        "voluntary_motion": {
            "purposeful": "clear",
            "clear_motion": "clear",
            "some": "weak",
            "minimal": "weak",
            "no_motion": "none",
            "absent": "none",
            "unknown": "uncertain",
        },
        "recovery_status": {
            "not_required": "not_needed",
            "no_recovery_needed": "not_needed",
            "no_collapse": "not_needed",
            "recovered": "clear",
            "complete": "clear",
            "partial_recovery": "partial",
            "incomplete": "partial",
            "no_recovery": "none",
            "not_recovered": "none",
            "unknown": "uncertain",
        },
        "driver_response": {
            "responsive": "active",
            "normal": "active",
            "engaged": "active",
            "diminished": "reduced",
            "low": "reduced",
            "unresponsive": "no_visible_response",
            "no_response": "no_visible_response",
            "none": "no_visible_response",
        },
    }
    for name, value in list(values.items()):
        normalized = value.replace(" ", "_").replace("-", "_")
        values[name] = aliases[name].get(normalized, normalized)

    allowed = {
        "eye_state": EYE_STATES,
        "head_pose": HEAD_POSES,
        "upper_body_posture": BODY_POSTURES,
        "hand_on_wheel": HAND_CONTACT,
        "voluntary_motion": VOLUNTARY_MOTION,
        "recovery_status": RECOVERY_STATUS,
        "driver_response": DRIVER_RESPONSES,
    }

    # A missing auxiliary field must not abort a driver_response-only
    # evaluation. Keep every valid model value and fill only malformed fields;
    # this does not trigger a second inference.
    if values["eye_state"] not in EYE_STATES:
        values["eye_state"] = "uncertain"
    if values["head_pose"] not in HEAD_POSES:
        values["head_pose"] = "upright"
    if values["upper_body_posture"] not in BODY_POSTURES:
        values["upper_body_posture"] = "upright"
    if values["hand_on_wheel"] not in HAND_CONTACT:
        values["hand_on_wheel"] = "not_confirmed"
    if values["voluntary_motion"] not in VOLUNTARY_MOTION:
        values["voluntary_motion"] = "uncertain"
    if values["recovery_status"] not in RECOVERY_STATUS:
        values["recovery_status"] = "uncertain"
    if values["driver_response"] not in DRIVER_RESPONSES:
        if (
            values["eye_state"] == "closed"
            or values["head_pose"] == "dropped"
            or values["upper_body_posture"] in ("forward_lean", "side_lean")
        ):
            values["driver_response"] = "reduced"
        else:
            values["driver_response"] = "active"

    return values


def derive_driver_response(current: Dict[str, str]) -> str:
    """Derive the final state from the six VLM observation fields.

    The priority order intentionally makes only explicit, visible evidence
    eligible for ``active``.  An ambiguous observation is always ``reduced``.
    """
    collapse_cue_count = sum((
        current["eye_state"] == "closed",
        current["head_pose"] == "dropped",
        current["upper_body_posture"] in ("forward_lean", "side_lean"),
    ))
    no_visible_response = (
        current["voluntary_motion"] == "none"
        or current["recovery_status"] == "none"
        or collapse_cue_count >= 2
    )
    if no_visible_response:
        return "no_visible_response"

    active = (
        current["eye_state"] == "open"
        and current["head_pose"] == "upright"
        and current["upper_body_posture"] == "upright"
        and current["hand_on_wheel"] == "confirmed"
        and current["voluntary_motion"] == "clear"
        and current["recovery_status"] == "not_needed"
    )
    return "active" if active else "reduced"


def event_supervisor_response(
    assessment: Dict[str, Any],
    tcn_evidence: Dict[str, Any],
    current: Dict[str, str],
) -> str:
    """Apply the same deterministic visual-state rule in event mode.

    TCN evidence may guide the upstream VLM inspection prompt, but it never
    overrides the six structured visual observations downstream.
    """
    del assessment, tcn_evidence
    return derive_driver_response(current)


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
            and decoded[8:12] == b"WEBP"
        ),
    }
    if not signatures_valid[mime_type]:
        raise ValueError("cabin_image bytes do not match mime_type")

    return {"mime_type": mime_type, "data_base64": encoded}


def validate_cabin_media(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("cabin_media must be a JSON object")
    mime_type = str(value.get("mime_type", "")).lower()
    if mime_type.startswith("image/"):
        return validate_cabin_image(value)
    if mime_type != "video/mp4":
        raise ValueError("cabin_media.mime_type must be JPEG, PNG, WebP, or MP4")
    encoded = value.get("data_base64")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("cabin_media.data_base64 is required")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("cabin_media.data_base64 is invalid") from exc
    if not decoded:
        raise ValueError("cabin_media is empty")
    if len(decoded) > MAX_VIDEO_BYTES:
        raise ValueError("cabin_media exceeds %d bytes" % MAX_VIDEO_BYTES)
    if len(decoded) < 12 or decoded[4:8] != b"ftyp":
        raise ValueError("cabin_media bytes do not look like MP4")
    return {"mime_type": mime_type, "data_base64": encoded}


def validate_tcn_evidence(value: Any) -> Dict[str, Any]:
    """Validate bounded temporal evidence without turning it into a state label.

    `driver_response` remains owned by the existing VLM assessment and its
    deterministic rules. The numeric TCN trajectory is supplied as a
    complementary temporal measurement, not as a final visual label.
    """
    if not isinstance(value, dict):
        raise ValueError("tcn_evidence must be a JSON object")
    forbidden_labels = {
        "driver_response", "eye_state", "head_pose", "upper_body_posture",
        "hand_on_wheel", "voluntary_motion", "recovery_status",
    }
    supplied_labels = sorted(forbidden_labels.intersection(value))
    if supplied_labels:
        raise ValueError(
            "tcn_evidence must not contain visual driver labels: "
            + ", ".join(supplied_labels)
        )

    def number(name: str, minimum: float, maximum: float) -> float:
        raw = value.get(name)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"tcn_evidence.{name} must be a number")
        parsed = float(raw)
        if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
            raise ValueError(
                f"tcn_evidence.{name} must be in [{minimum}, {maximum}]"
            )
        return parsed

    def nonnegative_int(name: str) -> int:
        raw = value.get(name)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ValueError(f"tcn_evidence.{name} must be a non-negative integer")
        return raw

    def nested_probability(mapping: Dict[str, Any], name: str, path: str) -> float:
        raw = mapping.get(name)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"{path}.{name} must be numeric")
        parsed = float(raw)
        if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
            raise ValueError(f"{path}.{name} must be in [0, 1]")
        return round(parsed, 6)

    def nested_bool(mapping: Dict[str, Any], name: str, path: str) -> bool:
        raw = mapping.get(name)
        if not isinstance(raw, bool):
            raise ValueError(f"{path}.{name} must be boolean")
        return raw

    def validate_ensemble(mapping: Any, path: str) -> Dict[str, Any]:
        if not isinstance(mapping, dict):
            raise ValueError(f"{path} must be an object")
        member_count = mapping.get("member_count")
        if isinstance(member_count, bool) or not isinstance(member_count, int) or not 1 <= member_count <= 32:
            raise ValueError(f"{path}.member_count must be an integer in [1, 32]")
        std = nested_probability(mapping, "std", path)
        minimum = nested_probability(mapping, "minimum", path)
        maximum = nested_probability(mapping, "maximum", path)
        if minimum > maximum:
            raise ValueError(f"{path}.minimum must not exceed maximum")
        return {
            "member_count": member_count,
            "std": std,
            "minimum": minimum,
            "maximum": maximum,
        }

    def validate_history(mapping: Any, path: str) -> List[Dict[str, float]]:
        if not isinstance(mapping, list) or len(mapping) > 20:
            raise ValueError(f"{path} must be a list of at most 20 points")
        result: List[Dict[str, float]] = []
        previous_offset = -float("inf")
        for index, item in enumerate(mapping):
            item_path = f"{path}[{index}]"
            if not isinstance(item, dict):
                raise ValueError(f"{item_path} must be an object")
            offset = item.get("time_offset_sec")
            if isinstance(offset, bool) or not isinstance(offset, (int, float)):
                raise ValueError(f"{item_path}.time_offset_sec must be numeric")
            offset = float(offset)
            if not math.isfinite(offset) or not -10.0 <= offset <= 0.001:
                raise ValueError(f"{item_path}.time_offset_sec must be in [-10, 0]")
            if offset < previous_offset:
                raise ValueError(f"{path} must be chronological")
            previous_offset = offset
            result.append({
                "time_offset_sec": round(offset, 3),
                "probability": nested_probability(item, "probability", item_path),
            })
        return result

    version = value.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ValueError("unsupported tcn_evidence.schema_version")
    source_media = value.get("source_media")
    if not isinstance(source_media, str) or not source_media or len(source_media) > 512:
        raise ValueError("tcn_evidence.source_media must be a short non-empty string")
    candidate = value.get("candidate_detected")
    if not isinstance(candidate, bool):
        raise ValueError("tcn_evidence.candidate_detected must be boolean")

    events_value = value.get("events")
    if not isinstance(events_value, list) or len(events_value) > MAX_TCN_EVENTS:
        raise ValueError("tcn_evidence.events must be a bounded JSON list")
    events: List[Dict[str, Any]] = []
    for index, event in enumerate(events_value):
        if not isinstance(event, dict):
            raise ValueError(f"tcn_evidence.events[{index}] must be an object")
        required = (
            "start_sec", "end_sec", "duration_sec", "peak_probability",
            "peak_timestamp_sec", "trigger_window_count",
        )
        if any(name not in event for name in required):
            raise ValueError(f"tcn_evidence.events[{index}] is incomplete")
        start = event["start_sec"]
        end = event["end_sec"]
        duration = event["duration_sec"]
        peak_probability = event["peak_probability"]
        peak_timestamp = event["peak_timestamp_sec"]
        window_count = event["trigger_window_count"]
        if any(isinstance(item, bool) or not isinstance(item, (int, float))
               for item in (start, end, duration, peak_probability, peak_timestamp)):
            raise ValueError(f"tcn_evidence.events[{index}] has non-numeric values")
        if (not all(math.isfinite(float(item)) for item in
                    (start, end, duration, peak_probability, peak_timestamp))
                or float(start) < 0.0 or float(end) < float(start)
                or float(duration) <= 0.0 or not 0.0 <= float(peak_probability) <= 1.0
                or not float(start) <= float(peak_timestamp) <= float(end)):
            raise ValueError(f"tcn_evidence.events[{index}] has invalid ranges")
        if isinstance(window_count, bool) or not isinstance(window_count, int) or window_count < 1:
            raise ValueError(f"tcn_evidence.events[{index}].trigger_window_count is invalid")
        events.append(
            {
                "start_sec": round(float(start), 3),
                "end_sec": round(float(end), 3),
                "duration_sec": round(float(duration), 3),
                "peak_probability": round(float(peak_probability), 6),
                "peak_timestamp_sec": round(float(peak_timestamp), 3),
                "trigger_window_count": window_count,
            }
        )

    peak = value.get("peak")
    peak_probability = None
    peak_timestamp_sec = None
    if peak is not None:
        if not isinstance(peak, dict):
            raise ValueError("tcn_evidence.peak must be an object or null")
        raw_probability = peak.get("tcn_probability")
        raw_timestamp = peak.get("timestamp_sec")
        if (isinstance(raw_probability, bool) or isinstance(raw_timestamp, bool)
                or not isinstance(raw_probability, (int, float))
                or not isinstance(raw_timestamp, (int, float))):
            raise ValueError("tcn_evidence.peak is incomplete")
        peak_probability = float(raw_probability)
        peak_timestamp_sec = float(raw_timestamp)
        if (not math.isfinite(peak_probability) or not math.isfinite(peak_timestamp_sec)
                or not 0.0 <= peak_probability <= 1.0 or peak_timestamp_sec < 0.0):
            raise ValueError("tcn_evidence.peak has invalid ranges")

    trigger_source = value.get("trigger_source", "legacy_tcn")
    allowed_trigger_sources = {
        "main_tcn", "control_reduced_tcn", "fall_onset_tcn", "eye_uncertain_watchdog",
        "visual_watchdog", "legacy_tcn",
    }
    if trigger_source not in allowed_trigger_sources:
        raise ValueError("tcn_evidence.trigger_source is invalid")
    risk_trend = value.get("risk_trend", "unavailable")
    if risk_trend not in {"increasing", "stable", "decreasing", "unavailable"}:
        raise ValueError("tcn_evidence.risk_trend is invalid")
    recent_value = value.get("recent_probabilities", [])
    if not isinstance(recent_value, list) or len(recent_value) > 20:
        raise ValueError("tcn_evidence.recent_probabilities must be a bounded list")
    recent_probabilities: List[float] = []
    for raw_probability in recent_value:
        if (isinstance(raw_probability, bool)
                or not isinstance(raw_probability, (int, float))):
            raise ValueError("tcn_evidence.recent_probabilities must contain numbers")
        parsed_probability = float(raw_probability)
        if not math.isfinite(parsed_probability) or not 0.0 <= parsed_probability <= 1.0:
            raise ValueError("tcn_evidence.recent_probabilities has an invalid value")
        recent_probabilities.append(round(parsed_probability, 6))
    raw_current_probability = value.get("current_probability", peak_probability)
    current_probability = None
    if raw_current_probability is not None:
        if (isinstance(raw_current_probability, bool)
                or not isinstance(raw_current_probability, (int, float))):
            raise ValueError("tcn_evidence.current_probability must be a number")
        current_probability = float(raw_current_probability)
        if (not math.isfinite(current_probability)
                or not 0.0 <= current_probability <= 1.0):
            raise ValueError("tcn_evidence.current_probability is outside [0, 1]")
    raw_above_duration = value.get("above_threshold_duration_sec")
    above_threshold_duration_sec = None
    if raw_above_duration is not None:
        if (isinstance(raw_above_duration, bool)
                or not isinstance(raw_above_duration, (int, float))):
            raise ValueError(
                "tcn_evidence.above_threshold_duration_sec must be a number"
            )
        above_threshold_duration_sec = float(raw_above_duration)
        if (not math.isfinite(above_threshold_duration_sec)
                or not 0.0 <= above_threshold_duration_sec <= 3600.0):
            raise ValueError(
                "tcn_evidence.above_threshold_duration_sec is invalid"
            )

    event_count = nonnegative_int("event_count")
    if event_count != len(events) or candidate != bool(events):
        raise ValueError("tcn_evidence candidate/event count is inconsistent")
    validated = {
        "schema_version": version,
        "source_media": source_media,
        "threshold": round(number("threshold", 0.0, 1.0), 6),
        "window_count": nonnegative_int("window_count"),
        "candidate_count": nonnegative_int("candidate_count"),
        "candidate_detected": candidate,
        "trigger_source": trigger_source,
        "current_probability": (
            round(current_probability, 6) if current_probability is not None else None
        ),
        "risk_trend": risk_trend,
        "recent_probabilities": recent_probabilities,
        "above_threshold_duration_sec": (
            round(above_threshold_duration_sec, 3)
            if above_threshold_duration_sec is not None else None
        ),
        "event_count": event_count,
        "events": events,
        "peak_probability": (
            round(peak_probability, 6) if peak_probability is not None else None
        ),
        "peak_timestamp_sec": (
            round(peak_timestamp_sec, 3) if peak_timestamp_sec is not None else None
        ),
    }
    raw_state_probabilities = value.get("state_probabilities")
    if raw_state_probabilities is not None:
        expected_states = {"active", "reduced", "no_visible_response"}
        if not isinstance(raw_state_probabilities, dict) or set(raw_state_probabilities) != expected_states:
            raise ValueError("tcn_evidence.state_probabilities must contain exactly three states")
        parsed_states: Dict[str, float] = {}
        for state in ("active", "reduced", "no_visible_response"):
            raw = raw_state_probabilities[state]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"tcn_evidence.state_probabilities.{state} must be numeric")
            probability = float(raw)
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"tcn_evidence.state_probabilities.{state} is invalid")
            parsed_states[state] = round(probability, 6)
        if not 0.98 <= sum(parsed_states.values()) <= 1.02:
            raise ValueError("tcn_evidence.state_probabilities must sum to approximately one")
        validated["state_probabilities"] = parsed_states
    raw_transition_probability = value.get("transition_probability")
    if raw_transition_probability is not None:
        if isinstance(raw_transition_probability, bool) or not isinstance(
            raw_transition_probability, (int, float)
        ):
            raise ValueError("tcn_evidence.transition_probability must be numeric")
        transition_probability = float(raw_transition_probability)
        if not math.isfinite(transition_probability) or not 0.0 <= transition_probability <= 1.0:
            raise ValueError("tcn_evidence.transition_probability is invalid")
        horizon = value.get("transition_horizon_sec")
        if isinstance(horizon, bool) or not isinstance(horizon, (int, float)):
            raise ValueError("tcn_evidence.transition_horizon_sec must be numeric")
        horizon_sec = float(horizon)
        if not math.isfinite(horizon_sec) or not 0.0 < horizon_sec <= 60.0:
            raise ValueError("tcn_evidence.transition_horizon_sec is invalid")
        validated["transition_probability"] = round(transition_probability, 6)
        validated["transition_horizon_sec"] = round(horizon_sec, 3)

    main_tcn = value.get("main_tcn")
    if main_tcn is not None:
        if not isinstance(main_tcn, dict):
            raise ValueError("tcn_evidence.main_tcn must be an object")
        observable = nested_bool(main_tcn, "observable", "tcn_evidence.main_tcn")
        quality = nested_probability(
            main_tcn, "observation_quality", "tcn_evidence.main_tcn"
        )
        raw_argmax_state = main_tcn.get("raw_argmax_state")
        if raw_argmax_state not in {"active", "reduced", "no_visible_response"}:
            raise ValueError("tcn_evidence.main_tcn.raw_argmax_state is invalid")
        state_decision = main_tcn.get("state_decision")
        if not isinstance(state_decision, str) or len(state_decision) > 128:
            raise ValueError("tcn_evidence.main_tcn.state_decision is invalid")
        validated["main_tcn"] = {
            "nvr_probability": nested_probability(
                main_tcn, "nvr_probability", "tcn_evidence.main_tcn"
            ),
            "combined_risk": nested_probability(
                main_tcn, "combined_risk", "tcn_evidence.main_tcn"
            ),
            "combined_risk_threshold": nested_probability(
                main_tcn, "combined_risk_threshold", "tcn_evidence.main_tcn"
            ),
            "observable": observable,
            "observation_quality": quality,
            "raw_argmax_state": raw_argmax_state,
            "state_decision": state_decision,
            "reduced_head_operational": nested_bool(
                main_tcn, "reduced_head_operational", "tcn_evidence.main_tcn"
            ),
            "ensemble": validate_ensemble(
                main_tcn.get("ensemble"), "tcn_evidence.main_tcn.ensemble"
            ),
            "history": validate_history(
                main_tcn.get("history"), "tcn_evidence.main_tcn.history"
            ),
        }

    eye_evidence = value.get("eye_evidence")
    if eye_evidence is not None:
        if not isinstance(eye_evidence, dict):
            raise ValueError("tcn_evidence.eye_evidence must be an object")
        validated["eye_evidence"] = {
            "measurement_reliable": nested_bool(
                eye_evidence, "measurement_reliable", "tcn_evidence.eye_evidence"
            ),
            "visible_fraction": nested_probability(
                eye_evidence, "visible_fraction", "tcn_evidence.eye_evidence"
            ),
            "closed_probability": nested_probability(
                eye_evidence, "closed_probability", "tcn_evidence.eye_evidence"
            ),
            "unknown_probability": nested_probability(
                eye_evidence, "unknown_probability", "tcn_evidence.eye_evidence"
            ),
            "closed_sustained": nested_bool(
                eye_evidence, "closed_sustained", "tcn_evidence.eye_evidence"
            ),
        }

    control_tcn = value.get("control_tcn")
    if control_tcn is not None:
        if not isinstance(control_tcn, dict):
            raise ValueError("tcn_evidence.control_tcn must be an object")
        confirmation_windows = control_tcn.get("confirmation_windows")
        if (isinstance(confirmation_windows, bool)
                or not isinstance(confirmation_windows, int)
                or not 1 <= confirmation_windows <= 100):
            raise ValueError(
                "tcn_evidence.control_tcn.confirmation_windows must be in [1, 100]"
            )
        validated["control_tcn"] = {
            "engaged_probability": nested_probability(
                control_tcn, "engaged_probability", "tcn_evidence.control_tcn"
            ),
            "disengaged_probability": nested_probability(
                control_tcn, "disengaged_probability", "tcn_evidence.control_tcn"
            ),
            "observable": nested_bool(
                control_tcn, "observable", "tcn_evidence.control_tcn"
            ),
            "observation_quality": nested_probability(
                control_tcn, "observation_quality", "tcn_evidence.control_tcn"
            ),
            "candidate": nested_bool(
                control_tcn, "candidate", "tcn_evidence.control_tcn"
            ),
            "reduced_confirmed": nested_bool(
                control_tcn, "reduced_confirmed", "tcn_evidence.control_tcn"
            ),
            "confirmation_windows": confirmation_windows,
            "ensemble": validate_ensemble(
                control_tcn.get("ensemble"), "tcn_evidence.control_tcn.ensemble"
            ),
            "history": validate_history(
                control_tcn.get("history"), "tcn_evidence.control_tcn.history"
            ),
        }
    return validated


def summarize_tcn_vlm_alignment(
    evidence: Dict[str, Any],
    driver_state: Dict[str, str],
) -> Dict[str, Any]:
    """Expose agreement for auditing; never alter the existing state machine."""
    candidate = bool(evidence["candidate_detected"])
    response = driver_state["driver_response"]
    if candidate and response in ("reduced", "no_visible_response"):
        alignment = "temporal_candidate_visually_supported"
    elif candidate:
        alignment = "temporal_candidate_not_visually_confirmed"
    elif response in ("reduced", "no_visible_response"):
        alignment = "visual_concern_without_temporal_candidate"
    else:
        alignment = "no_temporal_or_visual_concern"
    return {
        "alignment": alignment,
        "tcn_candidate_detected": candidate,
        "visual_driver_response": response,
    }


def prompt_with_tcn_context(tcn_evidence: Optional[Dict[str, Any]]) -> str:
    """Attach validated TCN measurements as complementary temporal evidence.

    The TCN measurements remain auxiliary. The VLM reconciles their temporal
    pattern with visible driver evidence and retains the visual decision rules
    in ``FAST_DRIVER_MONITOR_PROMPT`` as the final output contract.
    """
    if tcn_evidence is None:
        return FAST_DRIVER_MONITOR_PROMPT

    current_probability = tcn_evidence.get("current_probability")
    threshold = tcn_evidence.get("threshold")
    above_duration = tcn_evidence.get("above_threshold_duration_sec")
    peak_probability = tcn_evidence.get("peak_probability")
    peak_timestamp = tcn_evidence.get("peak_timestamp_sec")
    recent_probabilities = tcn_evidence.get("recent_probabilities", [])
    state_probabilities = tcn_evidence.get("state_probabilities")
    transition_probability = tcn_evidence.get("transition_probability")
    transition_horizon_sec = tcn_evidence.get("transition_horizon_sec")
    current_probability_text = (
        f"{float(current_probability):.3f}"
        if current_probability is not None else "unavailable"
    )
    trajectory = (
        "[" + ", ".join(f"{float(value):.3f}" for value in recent_probabilities) + "]"
        if recent_probabilities else "unavailable"
    )
    main_tcn = tcn_evidence.get("main_tcn")
    eye_evidence = tcn_evidence.get("eye_evidence")
    control_tcn = tcn_evidence.get("control_tcn")

    def history_text(evidence: Dict[str, Any] | None) -> str:
        if not isinstance(evidence, dict):
            return "unavailable"
        history = evidence.get("history")
        if not isinstance(history, list) or not history:
            return "unavailable"
        return "[" + ", ".join(
            f"{float(point['time_offset_sec']):.1f}s:{float(point['probability']):.3f}"
            for point in history
        ) + "]"

    main_context = ""
    if isinstance(main_tcn, dict):
        ensemble = main_tcn["ensemble"]
        reduced_head_note = (
            "The main reduced head is operational."
            if main_tcn["reduced_head_operational"]
            else "The main reduced score is diagnostic-only and must not itself determine reduced."
        )
        main_context = (
            "- main-state TCN: "
            f"raw NVR={float(main_tcn['nvr_probability']):.3f}, "
            f"combined NVR risk={float(main_tcn['combined_risk']):.3f} "
            f"(gate={float(main_tcn['combined_risk_threshold']):.3f}), "
            f"observable={bool(main_tcn['observable'])}, "
            f"observation quality={float(main_tcn['observation_quality']):.3f}, "
            f"raw argmax={main_tcn['raw_argmax_state']}\n"
            "- main-state ensemble NVR agreement: "
            f"members={int(ensemble['member_count'])}, std={float(ensemble['std']):.3f}, "
            f"range=[{float(ensemble['minimum']):.3f}, {float(ensemble['maximum']):.3f}]. "
            "Lower std and a narrower range mean stronger agreement.\n"
            f"- combined-risk history (time offset: risk): {history_text(main_tcn)}\n"
            f"- main-state policy: {reduced_head_note}\n"
        )

    eye_context = ""
    if isinstance(eye_evidence, dict):
        eye_context = (
            "- eye-model evidence: "
            f"reliable={bool(eye_evidence['measurement_reliable'])}, "
            f"visible fraction={float(eye_evidence['visible_fraction']):.3f}, "
            f"closed={float(eye_evidence['closed_probability']):.3f}, "
            f"unknown={float(eye_evidence['unknown_probability']):.3f}, "
            f"sustained closed={bool(eye_evidence['closed_sustained'])}.\n"
        )

    control_context = ""
    if isinstance(control_tcn, dict):
        ensemble = control_tcn["ensemble"]
        control_context = (
            "- control-engagement TCN: "
            f"engaged={float(control_tcn['engaged_probability']):.3f}, "
            f"disengaged={float(control_tcn['disengaged_probability']):.3f}, "
            f"observable={bool(control_tcn['observable'])}, "
            f"observation quality={float(control_tcn['observation_quality']):.3f}, "
            f"candidate={bool(control_tcn['candidate'])}, "
            f"reduced confirmation={bool(control_tcn['reduced_confirmed'])} "
            f"after {int(control_tcn['confirmation_windows'])} windows.\n"
            "- control ensemble disengagement agreement: "
            f"members={int(ensemble['member_count'])}, std={float(ensemble['std']):.3f}, "
            f"range=[{float(ensemble['minimum']):.3f}, {float(ensemble['maximum']):.3f}].\n"
            f"- control-disengagement history (time offset: probability): {history_text(control_tcn)}\n"
        )
    context = (
        "\n\nTCN-AUGMENTED TEMPORAL EVIDENCE (auxiliary, not a final label):\n"
        f"- trigger source: {tcn_evidence.get('trigger_source', 'legacy_tcn')}\n"
        f"- current temporal risk: {current_probability_text}\n"
        f"- candidate threshold: {float(threshold):.3f}\n"
        f"- recent risk trajectory (oldest to newest): {trajectory}\n"
        f"- trajectory trend: {tcn_evidence.get('risk_trend', 'unavailable')}\n"
        + (
            f"- time maintained above threshold: {float(above_duration):.3f} s\n"
            if above_duration is not None else "- time maintained above threshold: unavailable\n"
        )
        + (
            f"- recent peak risk: {float(peak_probability):.3f} at source "
            f"t={float(peak_timestamp):.3f} s\n"
            if peak_probability is not None and peak_timestamp is not None
            else "- recent peak risk: unavailable\n"
        )
        + (
            "- learned state distribution: "
            f"active={float(state_probabilities['active']):.3f}, "
            f"reduced={float(state_probabilities['reduced']):.3f}, "
            f"no_visible_response={float(state_probabilities['no_visible_response']):.3f}\n"
            if isinstance(state_probabilities, dict)
            else "- learned state distribution: unavailable\n"
        )
        + main_context
        + eye_context
        + control_context
        + (
            f"- learned probability of NVR transition within "
            f"{float(transition_horizon_sec):.1f} s: "
            f"{float(transition_probability):.3f}\n"
            if transition_probability is not None and transition_horizon_sec is not None
            else "- impending-NVR probability: unavailable\n"
        )
        + "Use these values to check subtle temporal changes that may be hard to "
        "see in individual frames. They may contain false positives or false "
        "negatives and must not override unambiguous video evidence. If main-state "
        "or eye observability is poor, treat the temporal score as uncertain. "
        "If control-engagement evidence is present, verify from the video whether "
        "the driver's hands or arms are disengaged from the steering wheel or "
        "controls; do not infer reduced state solely from that signal. Determine "
        "all seven output fields from the video and this complementary temporal "
        "evidence together.\n"
    )
    return (
        FAST_DRIVER_MONITOR_PROMPT
        + context
        + "Apply all existing driver-only, passenger-intervention, recovery, and "
        + "three-class rules above. Do not copy the temporal score into a class. "
        + "Active still requires sustained independent self-support and clear "
        + "purposeful control throughout the relevant clip."
    )


class DecisionEngine:
    def __init__(
        self,
        backend: str,
        model: str,
        llm_url: str,
        llm_api_key: str,
        llm_timeout: float,
    ) -> None:
        self.backend = backend
        self.model = model
        self.llm_url = llm_url
        self.llm_api_key = llm_api_key
        self.llm_timeout = llm_timeout
        self._inference_lock = threading.Lock()
        self._previous_observation_id: Optional[int] = None
        self._previous_driver_response: Optional[str] = None
        self._severe_collapse_streak = 0
        self._nonresponsive_streak = 0
        self._recovery_active_streak = 0
        self._recovering_from_no_visible = False
        self._event_no_response_latched = False
        self._event_recovery_streak = 0
        self._previous_event_timestamp_sec: Optional[float] = None

    def _apply_event_response_hysteresis(
        self,
        evidence: Dict[str, Any],
        current: Dict[str, str],
    ) -> Dict[str, str]:
        """Prevent one short event clip from clearing an active safety event.

        Event requests are sparse and their clips overlap.  A post-fall frame
        can look upright even though the preceding frames show collapse, so a
        no-visible-response finding is held until several adjacent clips show
        a complete independent recovery.
        """
        result = dict(current)
        raw_timestamp = evidence.get("peak_timestamp_sec")
        timestamp = (
            float(raw_timestamp)
            if isinstance(raw_timestamp, (int, float))
            and not isinstance(raw_timestamp, bool)
            and math.isfinite(float(raw_timestamp))
            else None
        )
        contiguous = (
            timestamp is not None
            and self._previous_event_timestamp_sec is not None
            and 0.0 < timestamp - self._previous_event_timestamp_sec
            <= EVENT_RESPONSE_CONTINUITY_SEC
        )
        if not contiguous:
            self._event_no_response_latched = False
            self._event_recovery_streak = 0

        # recovery_status is often emitted as uncertain/none when the clip
        # starts during the already-completed recovery. Do not let that field
        # alone keep the latch forever; the observable end-state is stronger.
        fully_recovered_active = (
            result["driver_response"] in ("active", "reduced")
            and result["eye_state"] == "open"
            and result["head_pose"] == "upright"
            and result["upper_body_posture"] == "upright"
            and result["voluntary_motion"] == "clear"
        )
        if result["driver_response"] == "no_visible_response":
            self._event_no_response_latched = True
            self._event_recovery_streak = 0
        elif self._event_no_response_latched:
            if fully_recovered_active and contiguous:
                self._event_recovery_streak += 1
            else:
                self._event_recovery_streak = 0
            if self._event_recovery_streak < EVENT_ACTIVE_RECOVERY_CONFIRMATIONS:
                result["driver_response"] = "reduced"
            else:
                self._event_no_response_latched = False
                self._event_recovery_streak = 0

        self._previous_event_timestamp_sec = timestamp
        return result

    def _apply_persistent_severe_collapse(
        self,
        observation: Dict[str, Any],
        current: Dict[str, str],
    ) -> Dict[str, str]:
        observation_id = observation.get("observation_id")
        consecutive = (
            isinstance(observation_id, int)
            and not isinstance(observation_id, bool)
            and isinstance(self._previous_observation_id, int)
            and observation_id == self._previous_observation_id + 1
        )
        severe_collapse = (
            current["eye_state"] == "closed"
            and current["head_pose"] == "dropped"
            and current["upper_body_posture"] == "forward_lean"
        )
        if severe_collapse:
            self._severe_collapse_streak = (
                self._severe_collapse_streak + 1 if consecutive else 1
            )
        else:
            self._severe_collapse_streak = 0

        result = dict(current)
        no_voluntary_recovery = (
            current["voluntary_motion"] == "none"
            and current["recovery_status"] == "none"
        )
        if consecutive and no_voluntary_recovery:
            self._nonresponsive_streak += 1
        elif current["driver_response"] == "no_visible_response":
            self._nonresponsive_streak = 1
        else:
            self._nonresponsive_streak = 0

        # A single ambiguous clip remains reduced. If the same clip sequence
        # repeatedly shows no self-initiated motion, no recovery, and no
        # confirmed control, promote it to no_visible_response even when the
        # final posture happens to look upright.
        if (
            self._nonresponsive_streak >= 2
            and current["driver_response"] == "reduced"
        ):
            result["driver_response"] = "no_visible_response"

        fully_recovered_active = (
            current["driver_response"] == "active"
            and current["eye_state"] == "open"
            and current["head_pose"] == "upright"
            and current["upper_body_posture"] == "upright"
            and current["voluntary_motion"] == "clear"
            and current["recovery_status"] == "clear"
        )
        if self._previous_driver_response == "no_visible_response":
            self._recovering_from_no_visible = True
            self._recovery_active_streak = (
                1 if fully_recovered_active and consecutive else 0
            )
        elif self._recovering_from_no_visible:
            if fully_recovered_active and consecutive:
                self._recovery_active_streak += 1
            elif current["driver_response"] == "no_visible_response":
                self._recovery_active_streak = 0
            elif not fully_recovered_active:
                self._recovery_active_streak = 0
        elif self._previous_driver_response == "reduced":
            if fully_recovered_active and consecutive:
                self._recovery_active_streak += 1
            else:
                self._recovery_active_streak = 0
        elif not fully_recovered_active:
            self._recovery_active_streak = 0

        # After reduced, require two fully recovered observations. After a
        # single no_visible_response, require eight consecutive fully
        # recovered observations before active can return.
        required_recovery_streak = 8 if self._recovering_from_no_visible else 2
        recovery_active_not_ready = (
            self._recovery_active_streak > 0
            and self._recovery_active_streak < required_recovery_streak
        )
        if recovery_active_not_ready:
            result["driver_response"] = "reduced"

        # Recovery must pass through the intermediate state. Do not allow a
        # single clip to jump directly from no_visible_response to active.
        recovery_jump = (
            self._previous_driver_response == "no_visible_response"
            and current["driver_response"] == "active"
        )
        recovery_attempt_after_no_visible = (
            self._previous_driver_response == "no_visible_response"
            and (
                current["voluntary_motion"] in ("clear", "weak")
                or current["recovery_status"] in ("clear", "partial")
            )
        )
        if self._severe_collapse_streak >= 2:
            result["driver_response"] = "no_visible_response"
        elif recovery_jump:
            result["driver_response"] = "reduced"
        elif recovery_attempt_after_no_visible:
            result["driver_response"] = "reduced"
        elif (
            consecutive
            and self._previous_driver_response == "no_visible_response"
            and current["driver_response"] == "reduced"
            and current["head_pose"] == "dropped"
            and current["upper_body_posture"] == "forward_lean"
        ):
            result["driver_response"] = "no_visible_response"

        if (
            self._recovery_active_streak >= required_recovery_streak
            and result["driver_response"] == "active"
        ):
            self._recovering_from_no_visible = False
            self._recovery_active_streak = 0

        self._previous_observation_id = (
            observation_id
            if isinstance(observation_id, int)
            and not isinstance(observation_id, bool)
            else None
        )
        self._previous_driver_response = result["driver_response"]
        return result

    def assess_driver(
        self,
        observation: Dict[str, Any],
        cabin_image: Dict[str, str],
    ) -> Dict[str, str]:
        if self.backend == "dry-run":
            return {
                "eye_state": "open",
                "head_pose": "upright",
                "upper_body_posture": "upright",
                "hand_on_wheel": "confirmed",
                "voluntary_motion": "clear",
                "recovery_status": "not_needed",
                "driver_response": "active",
            }

        prompt = prompt_with_tcn_context(observation.get("tcn_evidence"))
        with self._inference_lock:
            if self.backend == "openai":
                mime_type = cabin_image["mime_type"]
                user_content = []
                if mime_type == "video/mp4":
                    user_content.append({
                        "type": "text",
                        "text": "Assess the complete video clip.",
                    })
                    user_content.append({
                        "type": "video_url",
                        "video_url": {
                            "url": "data:video/mp4;base64," + cabin_image["data_base64"]
                        },
                    })
                else:
                    if observation.get("event_mode") is True:
                        user_content.append({
                            "type": "text",
                            "text": (
                                "This is a 2x2 temporal storyboard. Read all four "
                                "timestamped panels in chronological order. The red "
                                "person on the right is the driver and the brown person "
                                "on the left is the passenger. Judge collapse, eye closure, "
                                "and recovery across all panels; passenger assistance is "
                                "not driver action."
                            ),
                        })
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": "data:%s;base64,%s"
                            % (mime_type, cabin_image["data_base64"])
                        },
                    })
                model_text = self._call_openai_compatible(prompt, user_content, 64)
            elif self.backend == "ollama":
                if cabin_image["mime_type"] == "video/mp4":
                    raise ValueError("video input is supported only by the openai backend")
                model_text = self._call_ollama(
                    prompt,
                    "Assess the supplied driver image.",
                    [cabin_image["data_base64"]],
                )
            else:
                raise RuntimeError("unknown backend: %s" % self.backend)

        assessment = extract_json_object(model_text)
        current = validate_driver_assessment(assessment)
        # Qwen reports the six observations and its categorical conclusion.
        # Recompute that conclusion from the same fixed rule so malformed or
        # inconsistent model output cannot change the pipeline result.
        current["driver_response"] = derive_driver_response(current)
        return current

    def _call_openai_compatible(
        self,
        system_prompt: str,
        user_content: Any,
        max_tokens: int,
    ) -> str:
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
            raise ValueError(
                "unexpected OpenAI-compatible response"
            ) from exc
        if isinstance(content, dict):
            return json.dumps(content)
        return str(content)

    def _call_ollama(
        self,
        system_prompt: str,
        user_content: str,
        images: Optional[List[str]],
    ) -> str:
        user_message: Dict[str, Any] = {
            "role": "user",
            "content": user_content,
        }
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
            "options": {"temperature": 0.0, "num_predict": 64},
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
            self.llm_url,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.llm_timeout
            ) as response:
                response_body = response.read(MAX_REQUEST_BYTES + 1)
        except urllib.error.HTTPError as exc:
            details = exc.read(2048).decode("utf-8", errors="replace")
            raise RuntimeError(
                "model endpoint HTTP %d: %s" % (exc.code, details)
            ) from exc

        if len(response_body) > MAX_REQUEST_BYTES:
            raise ValueError("model response is too large")

        parsed = json.loads(response_body.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("model endpoint response must be a JSON object")
        return parsed


class DriverMonitorRequestHandler(BaseHTTPRequestHandler):
    server_version = "CarlaUpperBodyMonitor/1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            telemetry_age = self.server.telemetry_age_sec()
            self._write_json(
                200,
                {
                    "ok": True,
                    "protocol_version": PROTOCOL_VERSION,
                    "backend": self.server.engine.backend,
                    "inference_busy": self.server.inference_slot.locked(),
                    "telemetry_available": telemetry_age is not None,
                    "telemetry_age_s": telemetry_age,
                    "time_unix_s": time.time(),
                },
            )
            return

        if self.path == "/telemetry/latest":
            if not self._authenticated():
                self._write_json(401, {"error": "unauthorized"})
                return
            telemetry = self.server.latest_telemetry()
            if telemetry is None:
                self._write_json(
                    404, {"error": "telemetry unavailable or stale"}
                )
                return
            self._write_json(
                200,
                {
                    "ok": True,
                    "telemetry_age_s": self.server.telemetry_age_sec(),
                    "telemetry": telemetry,
                },
            )
            return

        if self.path == "/driver_state/latest":
            if not self._authenticated():
                self._write_json(401, {"error": "unauthorized"})
                return
            result = self.server.latest_driver_state()
            if result is None:
                self._write_json(
                    404, {"error": "driver state unavailable or stale"}
                )
                return
            self._write_json(200, result)
            return

        self._write_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in ("/telemetry", "/monitor"):
            self._write_json(404, {"error": "not found"})
            return

        if not self._authenticated():
            self._write_json(401, {"error": "unauthorized"})
            return

        try:
            payload = self._read_json()
            if self.path == "/telemetry":
                self._handle_telemetry(payload)
            else:
                self._handle_monitor(payload)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            self._write_json(400, {"error": str(exc)})
        except Exception as exc:
            print(
                "INFERENCE ERROR  %s: %s"
                % (type(exc).__name__, exc),
                flush=True,
            )
            self._write_json(
                502,
                {
                    "error": "model inference failed",
                    "detail": str(exc)[:500],
                },
            )

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

        payload = json.loads(
            self.rfile.read(content_length).decode("utf-8")
        )
        if not isinstance(payload, dict):
            raise ValueError("request JSON must be an object")
        return payload

    def _handle_monitor(self, payload: Dict[str, Any]) -> None:
        media_value = payload.get("cabin_media", payload.get("cabin_image"))
        if media_value is None:
            raise ValueError("cabin_media is required")

        if "protocol_version" in payload:
            if int(payload["protocol_version"]) != PROTOCOL_VERSION:
                raise ValueError("unsupported protocol_version")

        observation_id = payload.get("observation_id")
        if observation_id is not None and (
            isinstance(observation_id, bool)
            or not isinstance(observation_id, int)
        ):
            raise ValueError("observation_id must be an integer")

        if "captured_at_unix_s" in payload:
            value = payload["captured_at_unix_s"]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("captured_at_unix_s must be a number")

        event_mode = payload.get("event_mode", False)
        if not isinstance(event_mode, bool):
            raise ValueError("event_mode must be boolean")
        if event_mode and "tcn_evidence" not in payload:
            raise ValueError("event_mode requires tcn_evidence")

        cabin_image = validate_cabin_media(media_value)
        tcn_evidence = None
        if "tcn_evidence" in payload:
            tcn_evidence = validate_tcn_evidence(payload["tcn_evidence"])
        telemetry = payload.get("telemetry")
        if telemetry is not None:
            if not isinstance(telemetry, dict):
                raise ValueError("telemetry must be a JSON object")
            self.server.store_telemetry(telemetry)

        if not self.server.inference_slot.acquire(blocking=False):
            self._write_json(
                429,
                {"error": "inference busy; submit a newer observation later"},
            )
            return

        try:
            started = time.monotonic()
            inference_observation = dict(payload)
            if tcn_evidence is not None:
                # Use the validated, bounded contract in the prompt rather
                # than untrusted request data.
                inference_observation["tcn_evidence"] = tcn_evidence
            driver_state = self.server.engine.assess_driver(
                inference_observation, cabin_image
            )
            elapsed_ms = round((time.monotonic() - started) * 1000.0, 2)
            result: Dict[str, Any] = {
                "protocol_version": PROTOCOL_VERSION,
                "observation_id": observation_id,
                "driver_state": driver_state,
                "inference_ms": elapsed_ms,
            }
            if event_mode:
                result["event_mode"] = True
            if tcn_evidence is not None:
                result["tcn_evidence"] = tcn_evidence
            self.server.store_driver_state(result)
            self._write_json(200, result)
        finally:
            self.server.inference_slot.release()

    def _handle_telemetry(self, payload: Dict[str, Any]) -> None:
        if "protocol_version" in payload:
            if int(payload["protocol_version"]) != PROTOCOL_VERSION:
                raise ValueError("unsupported protocol_version")
        self.server.store_telemetry(payload)
        self._write_json(
            202,
            {
                "ok": True,
                "request_id": payload.get("request_id"),
                "inference_started": False,
                "message": "telemetry cached; waiting for cabin image",
            },
        )

    def log_message(self, format_string: str, *args: Any) -> None:
        print(
            "%s - %s"
            % (self.address_string(), format_string % args),
            flush=True,
        )

    def _write_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header(
                "Content-Type", "application/json; charset=utf-8"
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


class DriverMonitorServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: Any,
        engine: DecisionEngine,
        shared_token: str,
        driver_state_max_age_sec: float,
        telemetry_max_age_sec: float,
    ) -> None:
        super().__init__(address, DriverMonitorRequestHandler)
        self.engine = engine
        self.shared_token = shared_token.strip()
        self.driver_state_max_age_sec = driver_state_max_age_sec
        self.telemetry_max_age_sec = telemetry_max_age_sec
        self.inference_slot = threading.Lock()
        self._state_lock = threading.Lock()
        self._latest_telemetry: Optional[Dict[str, Any]] = None
        self._telemetry_received_at = 0.0
        self._latest_driver_state: Optional[Dict[str, Any]] = None
        self._driver_state_received_at = 0.0

    def store_driver_state(self, result: Dict[str, Any]) -> None:
        with self._state_lock:
            self._latest_driver_state = dict(result)
            self._driver_state_received_at = time.monotonic()

    def store_telemetry(self, telemetry: Dict[str, Any]) -> None:
        with self._state_lock:
            self._latest_telemetry = dict(telemetry)
            self._telemetry_received_at = time.monotonic()

    def telemetry_age_sec(self) -> Optional[float]:
        with self._state_lock:
            if self._latest_telemetry is None:
                return None
            return round(time.monotonic() - self._telemetry_received_at, 5)

    def latest_telemetry(self) -> Optional[Dict[str, Any]]:
        with self._state_lock:
            if self._latest_telemetry is None:
                return None
            age = time.monotonic() - self._telemetry_received_at
            if age > self.telemetry_max_age_sec:
                return None
            return dict(self._latest_telemetry)

    def latest_driver_state(self) -> Optional[Dict[str, Any]]:
        with self._state_lock:
            if self._latest_driver_state is None:
                return None
            age = time.monotonic() - self._driver_state_received_at
            if age > self.driver_state_max_age_sec:
                return None
            return dict(self._latest_driver_state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upper-body-only CARLA driver monitor"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--backend",
        choices=("dry-run", "openai", "ollama"),
        default="dry-run",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-2B-Instruct",
    )
    parser.add_argument(
        "--llm-url",
        default="http://127.0.0.1:8001/v1/chat/completions",
        help="OpenAI chat-completions URL or Ollama /api/chat URL",
    )
    parser.add_argument(
        "--llm-api-key",
        default=os.environ.get("LLM_API_KEY", ""),
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("LLM_BRIDGE_TOKEN", ""),
        help="shared bearer token (or set LLM_BRIDGE_TOKEN)",
    )
    parser.add_argument("--llm-timeout", type=float, default=10.0)
    parser.add_argument(
        "--driver-state-max-age",
        type=float,
        default=60.0,
    )

    # Accepted only for compatibility with the previous launch command.
    parser.add_argument("--safety-prompt-file", default=None)
    parser.add_argument(
        "--safety-supervisor",
        choices=("deterministic", "llm", "off"),
        default="off",
    )
    parser.add_argument("--safety-max-tokens", type=int, default=180)
    parser.add_argument("--valid-for-ms", type=int, default=1500)
    parser.add_argument("--telemetry-max-age", type=float, default=5.0)
    parser.add_argument("--enable-legacy-control", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.token = args.token.strip()
    args.llm_api_key = args.llm_api_key.strip()

    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be in [1, 65535]")
    if args.llm_timeout <= 0.0:
        raise ValueError("--llm-timeout must be positive")
    if args.driver_state_max_age <= 0.0:
        raise ValueError("--driver-state-max-age must be positive")
    if args.telemetry_max_age <= 0.0:
        raise ValueError("--telemetry-max-age must be positive")

    engine = DecisionEngine(
        backend=args.backend,
        model=args.model,
        llm_url=args.llm_url,
        llm_api_key=args.llm_api_key,
        llm_timeout=args.llm_timeout,
    )
    server = DriverMonitorServer(
        (args.host, args.port),
        engine,
        args.token,
        args.driver_state_max_age,
        args.telemetry_max_age,
    )

    print(
        "Upper-body driver monitor listening on %s:%d"
        % (args.host, args.port),
        flush=True,
    )
    print(
        "Output fields: eye_state, head_pose, upper_body_posture, "
        "hand_on_wheel, voluntary_motion, recovery_status, driver_response",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
