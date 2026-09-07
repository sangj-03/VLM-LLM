#!/usr/bin/env python3
"""Real-time in-cabin driver-response viewer for the local PC screen.

Fetches latest driver_state and cabin_image from DGX bridge endpoint
and renders OpenCV GUI window on local screen.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import sys
import time
import urllib.error
import urllib.request
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


KOREAN_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
DRIVER_RESPONSES = {
    "active", "reduced", "no_visible_response",
}
CONTENT_TOP = 104
CONTENT_BOTTOM_HEIGHT = 130
CONTENT_MARGIN = 12


def korean_font(size: int):
    return ImageFont.truetype(KOREAN_FONT, size)


def draw_steering_wheel_icon(draw: ImageDraw.ImageDraw,
                             center: tuple[int, int], radius: int,
                             color: tuple[int, int, int, int]) -> None:
    """Draw a font-independent steering-wheel warning icon."""
    cx, cy = center
    inner_radius = max(4, radius // 3)
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        outline=color, width=max(3, radius // 6))
    draw.ellipse(
        (cx - inner_radius, cy - inner_radius,
         cx + inner_radius, cy + inner_radius),
        outline=color, width=max(2, radius // 8))
    draw.line((cx, cy - inner_radius, cx, cy - radius + 3),
              fill=color, width=max(2, radius // 8))
    draw.line((cx - inner_radius, cy, cx - radius + 3, cy),
              fill=color, width=max(2, radius // 8))
    draw.line((cx + inner_radius, cy, cx + radius - 3, cy),
              fill=color, width=max(2, radius // 8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display DGX in-cabin frame and driver response on local PC screen")
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8000/driver_state/latest",
        help="DGX driver_state endpoint URL")
    parser.add_argument(
        "--token",
        default=os.environ.get("LLM_BRIDGE_TOKEN", "").strip(),
        help="Shared bearer token")
    parser.add_argument(
        "--poll-hz",
        type=float,
        default=5.0,
        help="Polling frequency in Hz")
    parser.add_argument(
        "--window-name",
        default="In-Cabin Driver Response Monitor",
        help="OpenCV display window title")
    parser.add_argument(
        "--window-width", type=int, default=1440,
        help="display window width in pixels")
    parser.add_argument(
        "--window-height", type=int, default=900,
        help="display window height in pixels")
    return parser.parse_args()


def fetch_latest_driver_state(server_url: str, token: str) -> dict | None:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(server_url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            body = response.read()
            return json.loads(body.decode("utf-8"))
    except Exception:
        return None


def center_cabin_frame(image: np.ndarray, canvas_width: int,
                       canvas_height: int) -> np.ndarray:
    """Letterbox one inference frame in the center without stretching it."""
    if image is None or image.ndim < 2 or image.size == 0:
        raise ValueError("cabin frame must be a non-empty image")
    if canvas_width <= 2 * CONTENT_MARGIN:
        raise ValueError("display width is too small")

    content_top = min(CONTENT_TOP, max(0, canvas_height - 1))
    content_bottom = max(
        content_top + 1, canvas_height - CONTENT_BOTTOM_HEIGHT)
    available_width = canvas_width - 2 * CONTENT_MARGIN
    available_height = content_bottom - content_top
    if available_height <= 0:
        raise ValueError("display height is too small")

    frame_height, frame_width = image.shape[:2]
    scale = min(
        available_width / float(frame_width),
        available_height / float(frame_height))
    resized_width = max(1, int(round(frame_width * scale)))
    resized_height = max(1, int(round(frame_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image, (resized_width, resized_height),
        interpolation=interpolation)

    canvas = np.empty((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[:] = (18, 22, 28)
    x = (canvas_width - resized_width) // 2
    y = content_top + (available_height - resized_height) // 2
    canvas[y:y + resized_height, x:x + resized_width] = resized
    return canvas


def draw_gui_overlay(image: np.ndarray, data: dict) -> np.ndarray:
    h, w = image.shape[:2]
    state = data.get("driver_state", {})
    if not isinstance(state, dict):
        state = {}

    response = str(state.get(
        "driver_response", data.get("driver_response", ""))) \
        .strip().lower().replace("-", "_").replace(" ", "_")
    if response not in DRIVER_RESPONSES:
        response = None

    response_labels = {
        "active": "명확한 자발적 반응",
        "reduced": "반응 저하",
        "no_visible_response": "명확한 자발적 반응 없음",
    }
    # RGB colors used by PIL: accent, top bar, translucent alert panel.
    colors = {
        "active": ((45, 210, 75), (22, 115, 45), (18, 105, 42, 225)),
        "reduced": ((255, 175, 35), (175, 95, 0), (180, 100, 0, 235)),
        "no_visible_response": ((245, 55, 55), (135, 15, 15), (155, 0, 0, 240)),
    }
    status_messages = {
        "active": "정상 감시 중 · 차량 안전 제어 개입 없음",
        "reduced": "운전자 반응이 약함 · 핸들을 잡으십시오",
        "no_visible_response": "운전자 무반응 · 갓길 안전 정차 수행",
    }
    if response is None:
        response_label = "결과 없음"
        status_message = "유효한 운전자 반응 결과를 기다리는 중"
        accent, bar_color = (145, 145, 145), (65, 65, 65)
        alert_color = (65, 65, 65, 235)
    else:
        response_label = response_labels[response]
        status_message = status_messages[response]
        accent, bar_color, alert_color = colors[response]

    try:
        confidence = float(state["confidence"])
        confidence_text = "  신뢰도: %.0f%%" % (confidence * 100.0)
    except (KeyError, TypeError, ValueError):
        confidence_text = ""

    obs_id = data.get("observation_id", "-")
    try:
        inf_ms = float(data.get("inference_ms", 0.0))
    except (TypeError, ValueError):
        inf_ms = 0.0
    try:
        age_s = float(data.get("result_age_s", 0.0))
    except (TypeError, ValueError):
        age_s = 0.0

    value_labels = {
        "open": "뜸", "closed": "감음", "upright": "정상",
        "dropped": "떨어짐", "tilted": "기울어짐",
        "forward_lean": "앞으로 기울어짐", "side_lean": "옆으로 기울어짐",
        "forward": "전방", "away": "전방 아님", "unavailable": "확인 불가",
        "uncertain": "확인 불가",
    }

    def translated(field: str) -> str:
        value = str(state.get(field, "uncertain")).strip().lower()
        return value_labels.get(value, value or "확인 불가")

    pil = Image.fromarray(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil, "RGBA")
    draw.rectangle((0, 0, w, 64), fill=bar_color + (255,))
    draw.rectangle((0, 64, w, 104), fill=(25, 25, 25, 245))
    draw.rectangle((0, h - 130, w, h), fill=(12, 12, 12, 225))
    draw.rectangle((0, 0, w - 1, h - 1), outline=accent + (255,), width=5)

    draw.text(
        (18, 12),
        "운전자 반응: %s%s" % (response_label, confidence_text),
        font=korean_font(27), fill=(255, 255, 255, 255))
    draw.text((18, 72), status_message,
              font=korean_font(19), fill=(235, 235, 235, 255))
    draw.text(
        (18, h - 112),
        "눈: %s   머리: %s   상체: %s   시선: %s" % (
            translated("eye_state"), translated("head_pose"),
            translated("upper_body_posture"), translated("gaze")),
        font=korean_font(18), fill=(255, 255, 255, 255))
    action = str(state.get("driver_action", "확인 불가"))
    draw.text(
        (18, h - 72), "운전자 행동: %s" % action,
        font=korean_font(17), fill=(205, 225, 255, 255))
    draw.text(
        (18, h - 38),
        "추론 번호: %s   추론 시간: %.0fms   결과 나이: %.1fs" % (
            obs_id, inf_ms, age_s),
        font=korean_font(15), fill=(185, 205, 225, 255))

    cabin_image = data.get("cabin_image")
    if (isinstance(cabin_image, dict)
            and cabin_image.get("data_base64")):
        frame_label = "현재 판정 영상 · 추론 번호 %s" % obs_id
        frame_font = korean_font(15)
        frame_bbox = draw.textbbox((0, 0), frame_label, font=frame_font)
        label_width = frame_bbox[2] - frame_bbox[0]
        label_x = max(12, (w - label_width) // 2)
        draw.rounded_rectangle(
            (label_x - 10, 110, label_x + label_width + 10, 140),
            radius=8, fill=(10, 10, 10, 190))
        draw.text(
            (label_x, 113), frame_label, font=frame_font,
            fill=(245, 245, 245, 255))

    if response in ("reduced", "no_visible_response"):
        alert_text = {
            "reduced": (
                "주의", "운전자 반응 저하",
                "핸들을 잡고 주행에 집중하십시오"),
            "no_visible_response": (
                "긴급 경고", "운전자 무반응",
                "차량이 갓길 안전 정차를 수행합니다"),
        }[response]
        box_h = min(205, max(170, h // 4))
        top = max(108, (h - box_h) // 2)
        draw.rectangle((0, top, w, top + box_h), fill=alert_color)
        if response == "reduced":
            draw_steering_wheel_icon(
                draw, (max(38, min(54, w // 8)), top + 48), 25,
                (255, 255, 255, 255))
        for text_value, font_size, y in (
                (alert_text[0], 50, top + 14),
                (alert_text[1], 30, top + 82),
                (alert_text[2], 21, top + 132)):
            font = korean_font(font_size)
            bbox = draw.textbbox((0, 0), text_value, font=font)
            x = max(12, (w - (bbox[2] - bbox[0])) // 2)
            draw.text((x, y), text_value, font=font,
                      fill=(255, 255, 255, 255))

    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def main() -> None:
    args = parse_args()

    stop_requested = False

    def request_stop(_signum: int, _frame: any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    print("==========================================================", flush=True)
    print("IN-CABIN LIVE MONITOR DISPLAY READY", flush=True)
    print(f"Connecting to DGX endpoint: {args.server_url}", flush=True)
    print(f"Polling rate: {args.poll_hz} Hz", flush=True)
    print("Press 'q' or Esc on the window (or Ctrl+C in terminal) to exit.", flush=True)
    print("==========================================================", flush=True)

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(args.window_name, args.window_width, args.window_height)
    # Keep a state-triggered warning visible above the CARLA/Unreal window when
    # the current OpenCV GUI backend supports the topmost property.
    if hasattr(cv2, "WND_PROP_TOPMOST"):
        try:
            cv2.setWindowProperty(
                args.window_name, cv2.WND_PROP_TOPMOST, 1.0)
        except cv2.error:
            pass

    period = 1.0 / args.poll_hz
    last_obs_id = -1

    try:
        while not stop_requested:
            start_t = time.monotonic()
            data = fetch_latest_driver_state(args.server_url, args.token)

            if data and "cabin_image" in data and "data_base64" in data["cabin_image"]:
                obs_id = data.get("observation_id", -1)
                if obs_id != last_obs_id:
                    last_obs_id = obs_id

                img_b64 = data["cabin_image"]["data_base64"]
                img_bytes = base64.b64decode(img_b64)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is not None:
                    centered_frame = center_cabin_frame(
                        frame, args.window_width, args.window_height)
                    display_frame = draw_gui_overlay(centered_frame, data)
                    cv2.imshow(args.window_name, display_frame)
            elif data:
                # Keep the display useful even when the latest API response
                # has no image yet or the image is temporarily unavailable.
                dashboard = np.zeros(
                    (args.window_height, args.window_width, 3), dtype=np.uint8)
                dashboard[:] = (18, 22, 28)
                cv2.imshow(args.window_name, draw_gui_overlay(dashboard, data))
            else:
                dashboard = np.zeros(
                    (args.window_height, args.window_width, 3), dtype=np.uint8)
                dashboard[:] = (18, 22, 28)
                cv2.putText(
                    dashboard, "DRIVER DISPLAY / CONNECTION WAITING",
                    (48, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (220, 220, 220), 2, cv2.LINE_AA)
                cv2.imshow(args.window_name, dashboard)

            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord('q')):  # Esc or 'q'
                break

            elapsed = time.monotonic() - start_t
            time.sleep(max(0.0, period - elapsed))
    finally:
        cv2.destroyAllWindows()
        print("\nLive monitor display closed.", flush=True)


if __name__ == "__main__":
    main()
