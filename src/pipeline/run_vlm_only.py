#!/usr/bin/env python3
"""DGX-side driver monitor that publishes observations to the bridge.

The legacy path sends files/clips to ``POST /monitor``.  ``--video-path``
sends the fixed driver-side ROI from every source frame to the bridge VLM.
The bridge owns eye-state and final driver-state inference.
"""

from __future__ import annotations

import argparse
import base64
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import csv
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Repository layout: src/pipeline/<runner>.py, with helper modules in the
# sibling src/features and src/evaluation directories.
SRC_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SRC_DIR.parent
WEIGHTS_DIR = REPO_ROOT / "assets" / "weights"
EVALUATION_DIR = SRC_DIR / "evaluation"
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))
from driver_state_metrics import evaluate_jsonl

PROTOCOL_VERSION = 1
DEFAULT_FRAMES = REPO_ROOT / "assets" / "frames"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".mp4"}
MEDIA_SUFFIXES = IMAGE_SUFFIXES | VIDEO_SUFFIXES
# OpenCV's MPEG-4 writer overshoots low bitrate targets. 700k here produces
# approximately 0.9 Mbps output, matching the pre-generated YOLO clips.
YOLO_VIDEO_BITRATE = "700k"
MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
}
DRIVER_RESPONSE_LABELS = ("active", "reduced", "no_visible_response")
# This is the historical driver-side training ROI, normalized for every video
# resolution. It is now used directly as the VLM input region.
DEFAULT_FALLBACK_DRIVER_ROI = (765 / 1728, 108 / 960, 1.0, 1.0)
# In this right-driver cabin view, prevent the event VLM crop from expanding
# into the passenger seat when YOLO's body box is broad or merged.
DEFAULT_VLM_DRIVER_SEAT_LEFT_RATIO = 0.57
DEFAULT_DRIVER_YOLO_MODEL = WEIGHTS_DIR / "yolo26n.pt"
DRIVER_SELECTOR_PATH = SRC_DIR / "features" / "driver_selector.py"
class RequestError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code

def normalized_roi_to_box(
    roi: tuple[float, float, float, float] | list[float],
    frame_shape: tuple[int, ...],
) -> tuple[int, int, int, int] | None:
    """Convert a normalized fallback driver ROI into a valid image crop."""
    height, width = frame_shape[:2]
    x1_ratio, y1_ratio, x2_ratio, y2_ratio = (float(value) for value in roi)
    x1 = max(0, min(width, int(round(x1_ratio * width))))
    y1 = max(0, min(height, int(round(y1_ratio * height))))
    x2 = max(0, min(width, int(round(x2_ratio * width))))
    y2 = max(0, min(height, int(round(y2_ratio * height))))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def load_driver_selector_module() -> Any:
    """Load the same YOLO person-tracking policy used by TCN–VLM–YOLO."""
    if not DRIVER_SELECTOR_PATH.is_file():
        raise FileNotFoundError(
            f"TCN–VLM–YOLO 운전자 selector가 없습니다: {DRIVER_SELECTOR_PATH}"
        )
    spec = importlib.util.spec_from_file_location(
        "vlm_only_driver_selector", DRIVER_SELECTOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("TCN–VLM–YOLO 운전자 selector를 불러올 수 없습니다.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send DGX cabin images and latest CARLA telemetry to bridge /monitor. "
            "The bridge runs VLM inference and updates /driver_state/latest."
        )
    )
    parser.add_argument(
        "--bridge-url",
        default=os.getenv("LLM_BRIDGE_URL", "http://127.0.0.1:8000"),
        help="DGX bridge base URL, not the vLLM 8001 URL",
    )
    parser.add_argument(
        "--bridge-token",
        default=os.getenv("LLM_BRIDGE_TOKEN", ""),
        help="shared bridge bearer token, or set LLM_BRIDGE_TOKEN",
    )
    parser.add_argument("--bridge-timeout", type=float, default=120.0)
    parser.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help="single/latest cabin image path; use with --watch for live operation",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=DEFAULT_FRAMES,
        help="directory of image frames or MP4 clips when --image-path is not set",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="keep watching --image-path or --frames-dir and process new images",
    )
    parser.add_argument(
        "--resend-same-image",
        action="store_true",
        help=(
            "with --watch, resend the unchanged latest image every "
            "--monitor-period-sec"
        ),
    )
    parser.add_argument(
        "--poll-sec",
        type=float,
        default=0.5,
        help="filesystem polling interval for --watch",
    )
    parser.add_argument(
        "--monitor-period-sec",
        type=float,
        default=0.5,
        help="minimum interval between successful monitor submissions",
    )
    parser.add_argument(
        "--max-telemetry-age-sec",
        type=float,
        default=5.0,
        help="skip inference when cached telemetry is older than this",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="maximum images to submit; 0 means unlimited",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="first sorted image to process, using a 1-based index",
    )
    parser.add_argument(
        "--start-observation-id",
        type=int,
        default=1,
        help="first observation_id sent to /monitor",
    )
    parser.add_argument(
        "--no-inline-telemetry",
        action="store_true",
        help="let /monitor read cached telemetry instead of sending it inline",
    )
    parser.add_argument(
        "--vlm-only",
        action="store_true",
        help=(
            "run image/video VLM inference without querying or sending CARLA "
            "telemetry"
        ),
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="print full /monitor response JSON instead of one-line status",
    )
    parser.add_argument(
        "--show-image",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "display each processed image in an OpenCV window "
            "(default: disabled; use --show-image to enable)"
        ),
    )
    parser.add_argument(
        "--result-display-sec",
        type=float,
        default=0.5,
        help="seconds to keep each completed image visible before continuing",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=None,
        help=("optional: append variables, risk judgment, and VLM inference "
              "as JSON Lines; omitted means no file is written"),
    )
    parser.add_argument(
        "--evaluate-results-file",
        type=Path,
        default=None,
        help=(
            "create an accuracy plot from an existing JSONL results file "
            "without submitting media to the bridge"
        ),
    )
    parser.add_argument(
        "--evaluate-last-records",
        type=int,
        default=0,
        help=(
            "when used with --evaluate-results-file, evaluate only the last "
            "N JSONL records; 0 means all records"
        ),
    )
    parser.add_argument(
        "--evaluate-event-results-file",
        type=Path,
        default=None,
        help=(
            "evaluate Qwen responses in a full-video events.jsonl file without "
            "submitting media to the bridge"
        ),
    )
    parser.add_argument(
        "--evaluate-last-event-records",
        type=int,
        default=0,
        help=(
            "when used with --evaluate-event-results-file, evaluate only the "
            "last N Qwen event records; 0 means all records"
        ),
    )
    parser.add_argument(
        "--event-accuracy-plot-file",
        type=Path,
        default=Path("qwen_event_accuracy.png"),
        help="output PNG for event-only Qwen driver_response accuracy",
    )
    parser.add_argument(
        "--event-matched-file",
        type=Path,
        default=None,
        help="optional CSV containing each Qwen event prediction and matched GT",
    )
    parser.add_argument(
        "--ground-truth-file",
        type=Path,
        default=None,
        help="CSV containing clip_name and driver_response ground truth",
    )
    parser.add_argument(
        "--accuracy-plot-file",
        type=Path,
        default=Path("driver_state_accuracy.png"),
        help="output PNG for per-field accuracy (default: driver_state_accuracy.png)",
    )
    parser.add_argument(
        "--show-accuracy-plot",
        action="store_true",
        help="show the final per-field accuracy graph after inference",
    )
    parser.add_argument(
        "--allow-partial-evaluation",
        action="store_true",
        help=(
            "evaluate only the processed subset instead of requiring every "
            "ground-truth item"
        ),
    )
    parser.add_argument(
        "--yolo-face-crop",
        action="store_true",
        help="YOLO-face로 운전자 얼굴을 검출해 crop한 미디어를 전송",
    )
    parser.add_argument(
        "--yolo-dynamic-track",
        action="store_true",
        default=True,
        help="영상 전체에서 YOLO 얼굴 위치를 재추적 (기본값: 활성화)",
    )
    parser.add_argument(
        "--yolo-static-crop",
        action="store_true",
        help="첫 운전자 얼굴 위치를 영상 전체에서 고정",
    )
    parser.add_argument(
        "--yolo-face-model",
        type=Path,
        default=WEIGHTS_DIR / "yolov8n-face.pt",
        help="YOLO 얼굴 검출 가중치",
    )
    parser.add_argument(
        "--yolo-eye-box",
        action="store_true",
        help="YOLO Pose 눈 keypoint 주변에 참고용 눈 바운딩박스를 표시",
    )
    parser.add_argument(
        "--yolo-eye-model",
        type=Path,
        default=WEIGHTS_DIR / "yolov8n-pose.pt",
        help="YOLO Pose 가중치 (눈 keypoint용)",
    )
    parser.add_argument(
        "--yolo-eye-interval",
        type=int,
        default=5,
        help="눈 YOLO 검출 주기(프레임). 기본값 5, 중간 프레임은 이전 박스 유지",
    )
    parser.add_argument(
        "--driver-side",
        choices=("left", "right"),
        default="right",
        help="여러 얼굴 중 운전자가 위치한 화면 방향",
    )
    parser.add_argument("--face-confidence", type=float, default=0.25)
    parser.add_argument(
        "--face-padding",
        type=float,
        default=0.50,
        help="검출 얼굴 박스 주변 여백 비율",
    )
    parser.add_argument(
        "--face-crop-size", type=int, default=384,
        help="VLM 운전자 얼굴 evidence 한 변 해상도 (기본 384)",
    )
    parser.add_argument(
        "--qwen-face-padding", type=float, default=0.30,
        help=(
            "전체 영상 VLM evidence에만 적용할 얼굴 주변 여백 비율. "
            "기존 --face-padding/TNC crop에는 영향을 주지 않습니다."
        ),
    )
    parser.add_argument(
        "--vlm-fixed-driver-box-padding", type=float, default=0.20,
        help=(
            "이벤트 VLM의 고정 운전자 box에 추가할 여백 비율. 기본 0.20은 "
            "쓰러질 때 머리·어깨가 프레임 밖으로 잘리는 것을 방지합니다."
        ),
    )
    parser.add_argument(
        "--vlm-driver-seat-left-ratio",
        type=float,
        default=DEFAULT_VLM_DRIVER_SEAT_LEFT_RATIO,
        help=(
            "이벤트 VLM crop의 최소 x 비율. 오른쪽 운전석 영상에서는 이 경계 "
            "왼쪽의 동승석 방향을 VLM evidence에서 제외합니다."
        ),
    )
    parser.add_argument(
        "--driver-roi-ratio",
        type=float,
        default=0.5,
        help="운전자 얼굴 후보를 제한할 화면 측 영역 비율 (기본값: 0.5)",
    )
    parser.add_argument(
        "--face-track-max-shift",
        type=float,
        default=0.12,
        help="이전 운전자 얼굴에서 허용할 최대 중심 이동 거리/화면 너비",
    )
    parser.add_argument(
        "--face-track-max-missed-frames",
        type=int,
        default=152,
        help="이 프레임 수까지 검출 실패 시 이전 운전자 위치 유지",
    )
    parser.add_argument(
        "--face-track-reacquire-after",
        type=int,
        default=10,
        help="연속 미검출 후 오른쪽 ROI 안에서 운전자 얼굴 재획득 허용",
    )
    parser.add_argument(
        "--qwen-face-track-max-shift",
        type=float,
        default=0.18,
        help=(
            "전체 영상 Qwen 얼굴 crop에서 이전 운전자 얼굴로부터 허용할 "
            "최대 중심 이동 거리/화면 너비"
        ),
    )
    parser.add_argument(
        "--qwen-face-track-max-missed-frames",
        type=int,
        default=12,
        help=(
            "전체 영상 Qwen 얼굴 crop에서 운전자 얼굴 미검출 시 "
            "마지막 얼굴 위치를 유지할 최대 샘플 수"
        ),
    )
    parser.add_argument(
        "--qwen-driver-guide-fallback-delay-sec",
        type=float,
        default=4.0,
        help=(
            "전체 영상 Qwen 얼굴 crop에서 시작 직후 driver-guide fallback을 "
            "허용하지 않는 시간(초)"
        ),
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=None,
        help=(
            "전체 원본 영상을 연속된 고정 운전자 ROI MP4 클립으로 나누어 VLM에 전송합니다."
        ),
    )
    parser.add_argument(
        "--fallback-driver-roi",
        type=float,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=DEFAULT_FALLBACK_DRIVER_ROI,
        help=(
            "VLM에 사용할 정규화된 고정 운전자 ROI. "
            "기본값은 기존 학습 ROI를 현재 해상도에 맞게 환산합니다."
        ),
    )
    parser.add_argument(
        "--yolo-driver-crop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "TCN-VLM-YOLO와 같은 YOLO person 추적기로 운전자를 선택해 "
            "VLM 클립을 crop합니다. 각 클립 안에서는 첫 box를 고정합니다."
        ),
    )
    parser.add_argument(
        "--driver-yolo-model",
        type=Path,
        default=DEFAULT_DRIVER_YOLO_MODEL,
        help="TCN-VLM-YOLO와 공유하는 운전자 YOLO person 가중치",
    )
    parser.add_argument("--driver-tracker", default="bytetrack.yaml")
    parser.add_argument(
        "--driver-yolo-confidence", type=float, default=0.15,
        help="운전자 YOLO person 검출 최소 confidence (기본: 0.15)",
    )
    parser.add_argument(
        "--driver-yolo-interval", type=int, default=1,
        help="운전자 YOLO 실행 간격(프레임). 중간 프레임은 마지막 box를 유지합니다.",
    )
    parser.add_argument(
        "--driver-min-x-ratio", type=float, default=0.45,
        help="운전자 후보를 허용할 최소 화면 x 중심 비율 (기본: 0.45)",
    )
    parser.add_argument(
        "--driver-box-hold-sec", type=float, default=2.0,
        help="YOLO가 잠시 끊겼을 때 마지막 운전자 box를 유지할 시간(초)",
    )
    parser.add_argument("--driver-anchor-x", type=float, default=1260.0)
    parser.add_argument("--driver-anchor-y", type=float, default=550.0)
    parser.add_argument(
        "--draw-driver-box",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "실시간 미리보기에서 고정 운전자 ROI를 표시합니다 "
            "(기본: 활성화; VLM 입력 원본에는 그리지 않음)"
        ),
    )
    parser.add_argument(
        "--video-start-sec", type=float, default=0.0,
        help="전체 영상 분석 시작 시각(초)",
    )
    parser.add_argument(
        "--video-max-sec", type=float, default=None,
        help="전체 영상에서 분석할 최대 길이(초)",
    )
    parser.add_argument(
        "--vlm-clip-seconds", type=float, default=1.5,
        help="각 VLM 호출에 포함할 비중첩 운전자 ROI MP4 길이(초, 기본 1.5)",
    )
    parser.add_argument(
        "--vlm-clip-max-width", type=int, default=768,
        help="전송 MP4의 최대 가로 해상도. 0이면 ROI 원본 해상도 유지",
    )
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="VLM이 원본 FPS보다 빠를 때만 원본 시간에 맞춰 대기합니다 (기본: 활성화)",
    )
    parser.add_argument(
        "--video-preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="전체 입력 영상과 프레임별 VLM 상태를 오버레이합니다",
    )
    parser.add_argument(
        "--event-output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "vlm_frame_results",
        help="프레임별 VLM 결과 JSONL 저장 디렉터리",
    )
    parser.add_argument("--api-key", help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--vehicle-state-source", help=argparse.SUPPRESS)
    parser.add_argument("--fps", help=argparse.SUPPRESS)
    parser.add_argument("--observation-window-sec", help=argparse.SUPPRESS)
    parser.add_argument("--temporal-window-sec", help=argparse.SUPPRESS)
    parser.add_argument("--behavior-confidence-threshold", help=argparse.SUPPRESS)
    parser.add_argument("--vehicle-speed-kmh", help=argparse.SUPPRESS)
    parser.add_argument("--traffic-signal", help=argparse.SUPPRESS)
    parser.add_argument("--throttle", help=argparse.SUPPRESS)
    parser.add_argument("--brake", help=argparse.SUPPRESS)
    parser.add_argument("--steer", help=argparse.SUPPRESS)
    parser.add_argument("--driving-context", help=argparse.SUPPRESS)
    parser.add_argument("--max-tokens-driver", help=argparse.SUPPRESS)
    parser.add_argument("--max-tokens-safety", help=argparse.SUPPRESS)
    parser.add_argument("--timeout", help=argparse.SUPPRESS)
    parser.add_argument("--retries", help=argparse.SUPPRESS)
    return parser.parse_args()


def frame_number(path: Path) -> int:
    match = re.search(r"_frame_(\d+)$", path.stem)
    if not match:
        match = re.search(r"F(\d+)$", path.stem, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0


def image_sort_key(path: Path) -> tuple[int, str]:
    return (frame_number(path), path.name)


def bridge_headers(args: argparse.Namespace) -> dict[str, str]:
    token = args.bridge_token.strip()
    if not token:
        raise ValueError("LLM_BRIDGE_TOKEN 또는 --bridge-token이 필요합니다.")
    return {
        "Accept": "application/json",
        "Authorization": "Bearer " + token,
    }


def request_json(
    method: str,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any] | None,
    timeout: float,
) -> dict[str, Any]:
    body = None
    final_headers = dict(headers)
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        final_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=final_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read(2048).decode("utf-8", errors="replace")
        raise RequestError(
            f"{method} {url} failed HTTP {exc.code}: {detail}",
            status_code=exc.code,
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RequestError(f"{method} {url} failed: {exc}") from exc
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{method} {url} returned non-JSON: {data[:200]!r}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{method} {url} returned non-object JSON")
    return parsed


def latest_telemetry(args: argparse.Namespace, headers: dict[str, str]) -> tuple[dict[str, Any], float | None]:
    url = args.bridge_url.rstrip("/") + "/telemetry/latest"
    payload = request_json("GET", url, headers, None, args.bridge_timeout)
    telemetry = payload.get("telemetry")
    if not isinstance(telemetry, dict):
        raise RuntimeError("/telemetry/latest 응답에 telemetry 객체가 없습니다.")
    age = payload.get("telemetry_age_s")
    age_s = float(age) if isinstance(age, (int, float)) else None
    if age_s is not None and age_s > args.max_telemetry_age_sec:
        raise RuntimeError(
            f"telemetry가 너무 오래됨: {age_s:.2f}s > {args.max_telemetry_age_sec:.2f}s"
        )
    return telemetry, age_s


_FACE_MODEL: Any = None
_POSE_MODEL: Any = None


def _load_face_model(model_path: Path) -> Any:
    global _FACE_MODEL
    if _FACE_MODEL is None:
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO 얼굴 가중치가 없습니다: {model_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("YOLO crop에는 ultralytics 패키지가 필요합니다.") from exc
        _FACE_MODEL = YOLO(str(model_path))
    return _FACE_MODEL


def _load_pose_model(model_path: Path) -> Any:
    global _POSE_MODEL
    if _POSE_MODEL is None:
        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO Pose 가중치가 없습니다: {model_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("눈 박스에는 ultralytics 패키지가 필요합니다.") from exc
        _POSE_MODEL = YOLO(str(model_path))
    return _POSE_MODEL


def _detect_eye_box(frame: Any, pose_model: Any) -> tuple[int, int, int, int] | None:
    result = pose_model.predict(frame, conf=0.25, verbose=False)[0]
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
        return None
    # The face crop is centered on the selected driver. If a passenger's
    # shoulder/face is still visible in the padded crop, choose the pose whose
    # face/body center is closest to the crop center instead of blindly using
    # pose result 0.
    all_points = keypoints.xy.detach().cpu().tolist()
    h, w = frame.shape[:2]
    crop_center = (w / 2.0, h / 2.0)
    def pose_distance(candidate: list[list[float]]) -> float:
        face = [candidate[i] for i in (0, 1, 2, 3, 4)
                if i < len(candidate) and 0 <= candidate[i][0] < w
                and 0 <= candidate[i][1] < h]
        pts = face or [p for p in candidate if 0 <= p[0] < w and 0 <= p[1] < h]
        if not pts:
            return float("inf")
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        return (cx - crop_center[0]) ** 2 + (cy - crop_center[1]) ** 2
    selected_index = min(range(len(all_points)), key=lambda i: pose_distance(all_points[i]))
    points = all_points[selected_index]
    confidences = None
    if getattr(keypoints, "conf", None) is not None:
        confidences = keypoints.conf[0].detach().cpu().tolist()
    eye_points = []
    for index in (1, 2):
        if index >= len(points):
            continue
        if confidences is not None and confidences[index] < 0.20:
            continue
        x, y = points[index]
        if not (0 <= x < w and 0 <= y < h):
            continue
        eye_points.append((x, y))
    if not eye_points:
        return None
    min_x = min(x for x, _ in eye_points)
    max_x = max(x for x, _ in eye_points)
    min_y = min(y for _, y in eye_points)
    max_y = max(y for _, y in eye_points)
    pad_x = max(14, int(w * 0.10))
    pad_y = max(10, int(h * 0.07))
    min_box_w = max(48, int(w * 0.22))
    min_box_h = max(28, int(h * 0.12))
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    box_w = max(max_x - min_x + 2 * pad_x, min_box_w)
    box_h = max(max_y - min_y + 2 * pad_y, min_box_h)
    x1 = max(0, int(center_x - box_w / 2))
    y1 = max(0, int(center_y - box_h / 2))
    x2 = min(w - 1, int(center_x + box_w / 2))
    y2 = min(h - 1, int(center_y + box_h / 2))
    return x1, y1, x2, y2


def _draw_eye_box(frame: Any, box: tuple[int, int, int, int] | None) -> Any:
    import cv2
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, "eyes", (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return frame


def _draw_eye_boxes(frame: Any, pose_model: Any) -> Any:
    return _draw_eye_box(frame, _detect_eye_box(frame, pose_model))


def _draw_driver_component_boxes(frame: Any, pose_model: Any,
                                 target_center: tuple[float, float] | None = None,
                                 driver_box: tuple[float, float, float, float] | None = None,
                                 cached_boxes: list[Any] | None = None,
                                 detect: bool = True) -> Any:
    """Draw coarse component boxes for the pose nearest the selected driver.

    The driver face detector supplies the target center, preventing a passenger
    pose from being selected. COCO
    pose indices are used: nose 0, eyes 1/2, ears 3/4, shoulders 5/6, hips
    11/12. Missing keypoints simply disable the corresponding box.
    """
    import cv2

    def draw_boxes(boxes: list[Any] | tuple[Any, ...]) -> Any:
        for label, box, color in boxes:
            if box is None:
                continue
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(14, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        return frame

    if not detect:
        return draw_boxes(cached_boxes or [])

    result = pose_model.predict(frame, conf=0.25, verbose=False)[0]
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
        if cached_boxes is not None:
            cached_boxes.clear()
        return frame
    all_points = keypoints.xy.detach().cpu().tolist()
    h, w = frame.shape[:2]
    wanted = target_center or (w / 2.0, h / 2.0)
    def distance(candidate: list[list[float]]) -> float:
        pts = [candidate[i] for i in (0, 1, 2, 3, 4)
               if i < len(candidate) and 0 <= candidate[i][0] < w
               and 0 <= candidate[i][1] < h]
        if not pts:
            pts = [p for p in candidate if 0 <= p[0] < w and 0 <= p[1] < h]
        if not pts:
            return float("inf")
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        return (cx - wanted[0]) ** 2 + (cy - wanted[1]) ** 2
    selected = min(all_points, key=distance)
    points = selected
    confidences = None
    if getattr(keypoints, "conf", None) is not None:
        confidences = keypoints.conf[0].detach().cpu().tolist()

    def valid(index: int) -> tuple[float, float] | None:
        if index >= len(points):
            return None
        if confidences is not None and confidences[index] < 0.20:
            return None
        x, y = points[index]
        return (x, y) if 0 <= x < w and 0 <= y < h else None

    def make_box(indices: tuple[int, ...], pad_x: float, pad_y: float,
                 top_extra: float = 0.0) -> tuple[int, int, int, int] | None:
        pts = [point for index in indices if (point := valid(index)) is not None]
        if not pts:
            return None
        min_x, max_x = min(p[0] for p in pts), max(p[0] for p in pts)
        min_y, max_y = min(p[1] for p in pts), max(p[1] for p in pts)
        bw = max(max_x - min_x, w * 0.10)
        bh = max(max_y - min_y, h * 0.08)
        x1 = max(0, int(min_x - max(bw * pad_x, 8)))
        x2 = min(w - 1, int(max_x + max(bw * pad_x, 8)))
        y1 = max(0, int(min_y - max(bh * pad_y, 8) - bw * top_extra))
        y2 = min(h - 1, int(max_y + max(bh * pad_y, 8)))
        return x1, y1, x2, y2

    # Eyes: tight box around the two eye keypoints. Do not use the generous
    # face padding used by the old preview implementation.
    eyes_box = make_box((1, 2), 0.22, 0.45)

    # Head: use the face detector box when available, with only a small margin
    # for hair and chin. This is more stable than expanding keypoints heavily.
    if driver_box is not None:
        fx1, fy1, fx2, fy2 = driver_box
        fw, fh = fx2 - fx1, fy2 - fy1
        head_box = (
            max(0, int(fx1 - fw * 0.12)),
            max(0, int(fy1 - fh * 0.28)),
            min(w - 1, int(fx2 + fw * 0.12)),
            min(h - 1, int(fy2 + fh * 0.12)),
        )
    else:
        head_box = make_box((0, 1, 2, 3, 4), 0.22, 0.22, 0.25)

    # Upper body: requested definition is from the top of the driver's head
    # through the visible end of the driver's body, not only shoulder/hip
    # keypoints. Keep it centered on the selected driver.
    head_points = [point for index in (0, 1, 2, 3, 4)
                   if (point := valid(index)) is not None]
    shoulder_points = [point for index in (5, 6)
                       if (point := valid(index)) is not None]
    body_points = head_points + shoulder_points
    if body_points:
        min_x = min(p[0] for p in body_points)
        max_x = max(p[0] for p in body_points)
        top_y = head_box[1] if head_box is not None else min(
            p[1] for p in head_points or body_points
        )
        bw = max(max_x - min_x, w * 0.18)
        upper_body_box = (
            # Include the driver's shoulders and arms, not only the narrow
            # span between the two shoulder keypoints.
            max(0, int(min_x - bw * 0.30)),
            max(0, int(top_y - max(10, h * 0.04))),
            min(w - 1, int(max_x + bw * 0.30)),
            h - 1,
        )
    else:
        upper_body_box = None

    boxes = (
        ("eyes", eyes_box, (0, 255, 255)),
        ("head", head_box, (255, 180, 0)),
        ("upper_body", upper_body_box, (0, 180, 0)),
    )
    if cached_boxes is not None:
        cached_boxes[:] = list(boxes)
    return draw_boxes(boxes)


def _select_driver_face(
    result: Any,
    side: str,
    frame_width: int,
    previous_box: tuple[float, float, float, float] | None,
    roi_ratio: float,
    max_shift_ratio: float,
    guide_box: tuple[int, int, int, int] | None = None,
) -> tuple[float, float, float, float] | None:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None
    candidates = boxes.xyxy.detach().cpu().tolist()

    def center(box: list[float] | tuple[float, ...]) -> tuple[float, float]:
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    # Reject passenger faces before choosing a driver. For a right-hand driver,
    # only the rightmost roi_ratio portion of the image is a valid driver area.
    boundary = frame_width * (1.0 - roi_ratio if side == "right" else roi_ratio)
    candidates = [
        box for box in candidates
        if (center(box)[0] >= boundary if side == "right" else center(box)[0] <= boundary)
    ]
    # A face detector can occasionally return a passenger/torso-sized box
    # when the driver is occluded. Reject those boxes instead of expanding
    # them into a cabin-wide crop.
    candidates = [
        box for box in candidates
        if (box[2] - box[0]) <= frame_width * 0.30
        and (box[3] - box[1]) <= frame_width * 0.20
    ]
    if guide_box is not None:
        # Use the body tracker as the driver's identity anchor. This prevents
        # face reacquisition from switching to the passenger after an
        # occlusion or a missed face detection.
        gx1, gy1, gx2, gy2 = guide_box
        gw, gh = max(1, gx2 - gx1), max(1, gy2 - gy1)
        gx1 -= 0.12 * gw
        gx2 += 0.12 * gw
        gy1 -= 0.12 * gh
        gy2 += 0.12 * gh

        def inside_driver_body(box: list[float] | tuple[float, ...]) -> bool:
            cx, cy = center(box)
            return gx1 <= cx <= gx2 and gy1 <= cy <= gy2

        candidates = [box for box in candidates if inside_driver_body(box)]
    if not candidates:
        return None

    if previous_box is None:
        # The driver is camera-calibrated near 66% of the frame width (34% for
        # a left-side driver). Choosing the rightmost detection used to let a
        # raised hand false-positive replace the actual driver's face.
        expected_x = frame_width * (0.66 if side == "right" else 0.34)
        selected = min(candidates, key=lambda box: abs(center(box)[0] - expected_x))
    else:
        previous_center = center(previous_box)

        def distance(box: list[float]) -> float:
            box_center = center(box)
            return (
                (box_center[0] - previous_center[0]) ** 2
                + (box_center[1] - previous_center[1]) ** 2
            ) ** 0.5

        selected = min(candidates, key=distance)
        # The body guide can be a merged driver/passenger box, so it is not
        # sufficient as an identity constraint by itself. Keep the face near
        # the last confirmed driver face and reject a passenger that appears
        # inside the broad guide.
        if distance(selected) > frame_width * max_shift_ratio:
            return None
    return tuple(float(value) for value in selected)


def _expanded_square(
    box: tuple[float, float, float, float], width: int, height: int, padding: float
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    size = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * padding)
    size = min(size, float(width), float(height))
    left = max(0, min(width - int(size), int(round(cx - size / 2.0))))
    top = max(0, min(height - int(size), int(round(cy - size / 2.0))))
    return left, top, left + int(size), top + int(size)


def _default_driver_face_box(
    width: int, height: int, side: str
) -> tuple[float, float, float, float]:
    """Camera-calibrated face box used when YOLO temporarily loses the driver."""
    center_x = width * (0.66 if side == "right" else 0.34)
    center_y = height * 0.46
    box_width = width * 0.11
    box_height = height * 0.22
    return (
        center_x - box_width / 2.0,
        center_y - box_height / 2.0,
        center_x + box_width / 2.0,
        center_y + box_height / 2.0,
    )


def _driver_guide_face_box(
    guide_box: tuple[int, int, int, int], side: str
) -> tuple[float, float, float, float]:
    """Estimate the driver's head area from a right/left seat body guide.

    The person detector can merge the passenger's reaching arm with the
    driver into one wide box. In that case the driver's head is still near the
    outside edge of the driver's seat, so use that edge rather than the merged
    box center as the temporary face-crop anchor.
    """
    gx1, gy1, gx2, gy2 = guide_box
    gw = max(1.0, float(gx2 - gx1))
    gh = max(1.0, float(gy2 - gy1))
    if side == "right":
        center_x = float(gx2) - 0.08 * gw
    else:
        center_x = float(gx1) + 0.08 * gw
    center_y = float(gy1) + 0.55 * gh
    box_width = max(48.0, 0.24 * gw)
    box_height = max(64.0, 0.26 * gh)
    return (
        center_x - box_width / 2.0,
        center_y - box_height / 2.0,
        center_x + box_width / 2.0,
        center_y + box_height / 2.0,
    )


def _crop_frame(
    frame: Any,
    model: Any,
    args: argparse.Namespace,
    tracker: dict[str, Any],
    guide_box: tuple[int, int, int, int] | None = None,
) -> tuple[Any, dict[str, Any]]:
    import cv2

    sample_index = int(tracker.get("sample_index", 0))
    tracker["sample_index"] = sample_index + 1
    fallback_delay_sec = float(getattr(args, "driver_guide_fallback_delay_sec", 0.0))
    fallback_allowed = sample_index / 10.0 >= fallback_delay_sec

    def resize_box_crop(box: tuple[int, int, int, int]) -> Any:
        bx1, by1, bx2, by2 = box
        crop = frame[max(0, by1):min(frame.shape[0], by2),
                     max(0, bx1):min(frame.shape[1], bx2)]
        if crop.size == 0:
            return cv2.resize(
                frame, (args.face_crop_size, args.face_crop_size),
                interpolation=cv2.INTER_AREA,
            )
        return cv2.resize(
            crop, (args.face_crop_size, args.face_crop_size),
            interpolation=cv2.INTER_AREA,
        )

    # In a fixed cabin camera, keep the first accepted driver face locked when
    # --yolo-static-crop is enabled. Re-running face selection after a missed
    # detection can otherwise reacquire the passenger.
    if getattr(args, "yolo_static_crop", False) and tracker.get("box") is not None:
        box = tracker["box"]
        x1, y1, x2, y2 = _expanded_square(
            box, frame.shape[1], frame.shape[0], args.face_padding
        )
        crop = resize_box_crop((x1, y1, x2, y2))
        return crop, tracker

    result = model.predict(frame, conf=args.face_confidence, verbose=False)[0]
    height, width = frame.shape[:2]
    previous_box = tracker.get("box")
    previous_guide_box = tracker.get("guide_box")

    missed_frames = int(tracker.get("missed_frames", 0))
    last_confirmed_box = tracker.get("last_confirmed_box")
    tracking_reference = (
        last_confirmed_box
        if last_confirmed_box is not None
        else previous_box
    )
    detected = _select_driver_face(
        result,
        args.driver_side,
        width,
        tracking_reference,
        args.driver_roi_ratio,
        args.face_track_max_shift,
        guide_box,
    )
    if detected is not None and previous_box is not None:
        # EMA smoothing prevents the video crop from visibly jittering.
        detected = tuple(0.7 * old + 0.3 * new for old, new in zip(previous_box, detected))
    if detected is not None:
        tracker["box"] = detected
        tracker["last_confirmed_box"] = detected
        tracker["missed_frames"] = 0
        tracker["crop_source"] = "face"
        if guide_box is not None and tracker.get("face_guide_offset") is None:
            gx1, gy1, gx2, gy2 = guide_box
            gw, gh = max(1.0, gx2 - gx1), max(1.0, gy2 - gy1)
            gcx, gcy = (gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0
            fcx = (detected[0] + detected[2]) / 2.0
            fcy = (detected[1] + detected[3]) / 2.0
            tracker["face_guide_offset"] = ((fcx - gcx) / gw, (fcy - gcy) / gh)
            tracker["face_guide_size"] = ((detected[2] - detected[0]) / gw,
                                           (detected[3] - detected[1]) / gh)
    else:
        tracker["missed_frames"] = int(tracker.get("missed_frames", 0)) + 1
        if guide_box is not None and tracker.get("box") is not None:
            # A single missed face detection is common while the passenger's
            # arm crosses the image. Keep the last confirmed driver face for a
            # few samples so a visible driver face does not jump to the
            # steering wheel immediately. Only then use the driver-edge
            # fallback for a sustained occlusion.
            short_hold_frames = max(1, min(3, args.face_track_max_missed_frames))
            if (
                tracker["missed_frames"] <= short_hold_frames
                or not fallback_allowed
            ):
                tracker["box"] = previous_box
                tracker["crop_source"] = "face_short_hold"
            else:
                tracker["box"] = _driver_guide_face_box(guide_box, args.driver_side)
                tracker["crop_source"] = "driver_guide_face_fallback"
        if (
            tracker["missed_frames"] > args.face_track_max_missed_frames
            and guide_box is None
        ):
            # Only use the camera-calibrated fallback when there is no body
            # guide.  While the selected driver's body is still tracked, keep
            # the last confirmed driver-face crop indefinitely; otherwise a
            # long occlusion can make the face detector reacquire the passenger
            # or a window/background at the default location.
            tracker["box"] = _default_driver_face_box(
                width, height, args.driver_side
            )
    if guide_box is not None:
        tracker["guide_box"] = guide_box
    box = tracker.get("box")
    if box is None:
        tracker["box"] = _default_driver_face_box(
            width, height, args.driver_side
        )
        box = tracker["box"]
    x1, y1, x2, y2 = _expanded_square(box, width, height, args.face_padding)
    crop = resize_box_crop((x1, y1, x2, y2))
    return crop, tracker


def yolo_face_crop_bytes(path: Path, args: argparse.Namespace) -> tuple[str, bytes]:
    import cv2

    model = _load_face_model(args.yolo_face_model)
    pose_model = _load_pose_model(args.yolo_eye_model) if args.yolo_eye_box else None
    if path.suffix.lower() in IMAGE_SUFFIXES:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"이미지를 읽을 수 없습니다: {path}")
        height, width = frame.shape[:2]
        crop, _ = _crop_frame(frame, model, args, {
            "box": _default_driver_face_box(width, height, args.driver_side),
            "missed_frames": 0,
        })
        if pose_model is not None:
            crop = _draw_eye_boxes(crop, pose_model)
        ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError(f"얼굴 crop JPEG 인코딩 실패: {path}")
        return "image/jpeg", encoded.tobytes()

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    # Find the earliest valid driver face before writing frame 0. This avoids
    # showing a default/passenger-adjacent crop for the first few frames and
    # then visibly jumping to the driver after tracking starts.
    initial_box: tuple[float, float, float, float] | None = None
    while True:
        ok, scan_frame = capture.read()
        if not ok:
            break
        scan_height, scan_width = scan_frame.shape[:2]
        scan_result = model.predict(
            scan_frame, conf=args.face_confidence, verbose=False
        )[0]
        initial_box = _select_driver_face(
            scan_result,
            args.driver_side,
            scan_width,
            None,
            args.driver_roi_ratio,
            args.face_track_max_shift,
        )
        if initial_box is not None:
            break
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    temp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_path = Path(temp.name)
    temp.close()
    encoded_path: Path | None = None
    writer = cv2.VideoWriter(
        str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (args.face_crop_size, args.face_crop_size),
    )
    tracker: dict[str, Any] | None = None
    frames_written = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if tracker is None:
                height, width = frame.shape[:2]
                tracker = {
                    "box": initial_box or _default_driver_face_box(
                        width, height, args.driver_side),
                    "missed_frames": 0,
                }
            if getattr(args, "yolo_dynamic_track", True) and not getattr(
                args, "yolo_static_crop", False
            ):
                crop, tracker = _crop_frame(frame, model, args, tracker)
            else:
                x1, y1, x2, y2 = _expanded_square(
                    tracker["box"], width, height, args.face_padding
                )
                crop = frame[y1:y2, x1:x2]
                crop = cv2.resize(
                    crop,
                    (args.face_crop_size, args.face_crop_size),
                    interpolation=cv2.INTER_AREA,
                )
            if pose_model is not None:
                crop = _draw_eye_boxes(crop, pose_model)
            writer.write(crop)
            frames_written += 1
        capture.release()
        writer.release()
        if frames_written == 0 or not temp_path.is_file():
            raise RuntimeError(f"얼굴 crop 영상 생성 실패: {path}")
        output_path = Path(tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ).name)
        encoded_path = output_path
        if shutil.which("ffmpeg") is not None:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", str(temp_path),
                    "-an",
                    "-c:v", "mpeg4",
                    "-b:v", YOLO_VIDEO_BITRATE,
                    "-r", str(fps),
                    str(output_path),
                ],
                check=True,
            )
            return "video/mp4", output_path.read_bytes()
        return "video/mp4", temp_path.read_bytes()
    finally:
        capture.release()
        writer.release()
        temp_path.unlink(missing_ok=True)
        if encoded_path is not None:
            encoded_path.unlink(missing_ok=True)


def yolo_driver_boxes_bytes(path: Path, args: argparse.Namespace) -> tuple[str, bytes]:
    """Keep the original media and draw boxes only on the selected driver."""
    import cv2
    face_model = _load_face_model(args.yolo_face_model)
    pose_model = _load_pose_model(args.yolo_eye_model)

    def annotate(frame: Any, box_cache: list[Any]) -> Any:
        height, width = frame.shape[:2]
        result = face_model.predict(frame, conf=args.face_confidence, verbose=False)[0]
        driver = _select_driver_face(
            result, args.driver_side, width, None,
            args.driver_roi_ratio, args.face_track_max_shift,
        )
        if driver is None:
            box_cache.clear()
            return frame
        center = ((driver[0] + driver[2]) / 2.0, (driver[1] + driver[3]) / 2.0)
        return _draw_driver_component_boxes(
            frame, pose_model, center, driver, box_cache, detect=True
        )

    if path.suffix.lower() in IMAGE_SUFFIXES:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"이미지를 읽을 수 없습니다: {path}")
        frame = annotate(frame, [])
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError(f"운전자 박스 JPEG 인코딩 실패: {path}")
        return "image/jpeg", encoded.tobytes()

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    temp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_path = Path(temp.name)
    temp.close()
    writer = cv2.VideoWriter(
        str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    frames_written = 0
    frame_index = 0
    box_cache: list[Any] = []
    interval = max(1, int(getattr(args, "yolo_driver_box_interval", 5)))
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % interval == 0:
                annotated = annotate(frame, box_cache)
            else:
                annotated = _draw_driver_component_boxes(
                    frame, pose_model, cached_boxes=box_cache, detect=False
                )
            writer.write(annotated)
            frames_written += 1
            frame_index += 1
        capture.release()
        writer.release()
        if frames_written == 0:
            raise RuntimeError(f"운전자 박스 영상 생성 실패: {path}")
        return "video/mp4", temp_path.read_bytes()
    finally:
        capture.release()
        writer.release()
        temp_path.unlink(missing_ok=True)


def yolo_eye_box_bytes(path: Path, args: argparse.Namespace) -> tuple[str, bytes]:
    """Annotate an existing video/image without changing its crop."""
    import cv2

    pose_model = _load_pose_model(args.yolo_eye_model)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"이미지를 읽을 수 없습니다: {path}")
        frame = _draw_eye_boxes(frame, pose_model)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError(f"눈 박스 JPEG 인코딩 실패: {path}")
        return "image/jpeg", encoded.tobytes()

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    temp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_path = Path(temp.name)
    temp.close()
    output_path = Path(tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name)
    writer = cv2.VideoWriter(
        str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    frames_written = 0
    try:
        eye_box = None
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % max(1, args.yolo_eye_interval) == 0:
                detected_box = _detect_eye_box(frame, pose_model)
                if detected_box is not None:
                    eye_box = detected_box
            writer.write(_draw_eye_box(frame, eye_box))
            frames_written += 1
            frame_index += 1
        capture.release()
        writer.release()
        if frames_written == 0:
            raise RuntimeError(f"눈 박스 영상 생성 실패: {path}")
        if shutil.which("ffmpeg") is not None:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(temp_path),
                 "-an", "-c:v", "mpeg4", "-b:v", YOLO_VIDEO_BITRATE,
                 "-r", str(fps), str(output_path)], check=True
            )
            return "video/mp4", output_path.read_bytes()
        return "video/mp4", temp_path.read_bytes()
    finally:
        capture.release()
        writer.release()
        temp_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


def cabin_image_payload(path: Path, args: argparse.Namespace) -> dict[str, str]:
    suffix = path.suffix.lower()
    if suffix not in MIME_TYPES:
        raise ValueError(f"지원하지 않는 미디어 확장자입니다: {path}")
    if args.yolo_face_crop:
        mime_type, media_bytes = yolo_face_crop_bytes(path, args)
    elif args.yolo_eye_box:
        mime_type, media_bytes = yolo_eye_box_bytes(path, args)
    else:
        mime_type, media_bytes = MIME_TYPES[suffix], path.read_bytes()
    return {
        "mime_type": mime_type,
        "data_base64": base64.b64encode(media_bytes).decode("ascii"),
    }


def driving_summary(telemetry: dict[str, Any]) -> dict[str, Any]:
    vehicle = telemetry.get("vehicle", {})
    speed = float(vehicle.get("speed_kph", 0.0) or 0.0)
    return {
        "motion_state": "stationary" if abs(speed) < 0.1 else "moving",
        "current_speed_kph": round(speed, 3),
    }


def build_monitor_payload(
    args: argparse.Namespace,
    image_path: Path,
    observation_id: int,
    headers: dict[str, str],
) -> dict[str, Any]:
    telemetry: dict[str, Any] | None = None
    telemetry_age_s: float | None = None
    if not args.no_inline_telemetry and not args.vlm_only:
        telemetry, telemetry_age_s = latest_telemetry(args, headers)

    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "observation_id": observation_id,
        "captured_at_unix_s": time.time(),
        "cabin_media": cabin_image_payload(image_path, args),
    }
    if telemetry is not None:
        payload["telemetry"] = telemetry
        payload["driving_summary"] = driving_summary(telemetry)
    if telemetry_age_s is not None:
        payload["telemetry_age_s"] = telemetry_age_s
    return payload


def post_monitor(
    args: argparse.Namespace,
    image_path: Path,
    observation_id: int,
    headers: dict[str, str],
) -> dict[str, Any]:
    payload = build_monitor_payload(args, image_path, observation_id, headers)
    url = args.bridge_url.rstrip("/") + "/monitor"
    return request_json("POST", url, headers, payload, args.bridge_timeout)


def compose_event_evidence_frame(source_frame: Any, driver_crop: Any, cv2: Any) -> Any:
    """Preserve cabin/passenger context while keeping the driver easy to inspect."""
    # Keep the event image below a quarter megapixel so Qwen can respond in
    # roughly the same time as a single-image request, while retaining both
    # the cabin context and a legible driver close-up.
    panel_height, context_width, driver_width = 288, 512, 336

    def fit(frame: Any, width: int) -> Any:
        height, source_width = frame.shape[:2]
        scale = min(width / max(source_width, 1), panel_height / max(height, 1))
        resized = cv2.resize(
            frame,
            (max(1, int(round(source_width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        import numpy as np
        canvas = np.zeros((panel_height, width, 3), dtype=np.uint8)
        y = (panel_height - resized.shape[0]) // 2
        x = (width - resized.shape[1]) // 2
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        return canvas

    context = fit(source_frame, context_width)
    driver = fit(driver_crop, driver_width)
    cv2.putText(context, "CABIN CONTEXT", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(driver, "DRIVER CLOSE-UP", (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return cv2.hconcat([context, driver])


def select_event_video_frames(
    frames: deque[tuple[float, Any]], end_time: float, seconds: float, fps: float
) -> list[tuple[float, Any]]:
    """Uniformly sample a short temporal event clip from the live frame buffer."""
    start_time = max(0.0, end_time - seconds)
    available = [(t, image) for t, image in frames if start_time <= t <= end_time]
    if not available:
        raise ValueError("이벤트 영상에 사용할 프레임이 없습니다.")
    count = max(2, int(math.ceil(seconds * fps)))
    targets = [
        start_time + (end_time - start_time) * index / (count - 1)
        for index in range(count)
    ]
    return [
        min(available, key=lambda item: abs(item[0] - target))
        for target in targets
    ]


def select_fixed_driver_box_event_frames(
    source_video: Path,
    end_time: float,
    seconds: float,
    frame_count: int,
    reference_box: tuple[int, int, int, int],
    padding: float,
    driver_seat_left_ratio: float,
) -> list[tuple[float, Any]]:
    """Read event frames using one stationary driver crop for the full event.

    A crop that follows a person detector re-centres a falling driver in every
    panel and hides the very displacement the VLM must judge.  The reference
    box is chosen before the candidate interval, then reused at every sampled
    timestamp from the original video.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("고정 VLM 이벤트 크롭에는 opencv-python이 필요합니다.") from exc
    if frame_count < 2:
        raise ValueError("event frame count must be at least two")
    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise RuntimeError(f"이벤트 원본 영상을 열 수 없습니다: {source_video}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        x1, y1, x2, y2 = reference_box
        box_width, box_height = max(1, x2 - x1), max(1, y2 - y1)
        # The left side contains the passenger seat. Limit left-side padding
        # so a broad/merged YOLO person box cannot make the passenger the
        # dominant face in Qwen's evidence.
        seat_left = int(round(width * driver_seat_left_ratio))
        crop_x1 = max(seat_left, int(round(x1 - 0.25 * padding * box_width)))
        crop_x2 = min(
            width, max(crop_x1 + 1, int(round(x2 + padding * box_width)))
        )
        crop_y1 = max(0, int(round(y1 - padding * box_height)))
        crop_y2 = min(
            height, max(crop_y1 + 1, int(round(y2 + padding * box_height)))
        )
        start_time = max(0.0, end_time - seconds)
        targets = [
            start_time + (end_time - start_time) * index / (frame_count - 1)
            for index in range(frame_count)
        ]
        selected: list[tuple[float, Any]] = []
        for timestamp in targets:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"이벤트 프레임을 읽지 못했습니다: {timestamp:.3f}s")
            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size == 0:
                raise RuntimeError("고정 운전자 crop이 비어 있습니다.")
            selected.append((timestamp, crop))
        return selected
    finally:
        capture.release()


def crop_fixed_driver_box(
    frame: Any,
    reference_box: tuple[int, int, int, int],
    padding: float,
    driver_seat_left_ratio: float,
) -> tuple[Any, tuple[int, int, int, int]]:
    """Apply the fusion pipeline's stationary VLM crop to one source frame."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = reference_box
    box_width, box_height = max(1, x2 - x1), max(1, y2 - y1)
    seat_left = int(round(width * driver_seat_left_ratio))
    crop_x1 = max(seat_left, int(round(x1 - 0.25 * padding * box_width)))
    crop_x2 = min(width, max(crop_x1 + 1, int(round(x2 + padding * box_width))))
    crop_y1 = max(0, int(round(y1 - padding * box_height)))
    crop_y2 = min(height, max(crop_y1 + 1, int(round(y2 + padding * box_height))))
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        raise RuntimeError("고정 운전자 crop이 비어 있습니다.")
    return crop, (crop_x1, crop_y1, crop_x2, crop_y2)


def encode_event_video(
    frames: list[tuple[float, Any]], output_path: Path, fps: float,
    max_width: int = 0,
) -> Path:
    """Write the recent driver-only buffer as a compact MP4 for one Qwen event."""
    if not frames:
        raise ValueError("Qwen 이벤트용 프레임이 없습니다.")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("이벤트 MP4 생성에는 opencv-python이 필요합니다.") from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first = frames[0][1]
    source_height, source_width = first.shape[:2]
    if max_width > 0 and source_width > max_width:
        scale = float(max_width) / float(source_width)
        width = max(2, int(round(source_width * scale)))
        height = max(2, int(round(source_height * scale)))
    else:
        width, height = source_width, source_height
    width, height = max(2, width - width % 2), max(2, height - height % 2)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"이벤트 MP4를 만들 수 없습니다: {output_path}")
    try:
        for _, frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(
                    frame, (width, height), interpolation=cv2.INTER_AREA
                )
            writer.write(frame)
    finally:
        writer.release()
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"이벤트 MP4 생성에 실패했습니다: {output_path}")
    if output_path.stat().st_size > 20 * 1024 * 1024:
        raise RuntimeError(f"이벤트 MP4가 20 MiB를 초과합니다: {output_path}")
    return output_path


def encode_event_storyboard(
    frames: list[tuple[float, Any]], output_path: Path, panel_size: int = 256,
) -> Path:
    """Write four timestamped temporal panels for reliable VLM inspection."""
    if not frames:
        raise ValueError("Qwen 스토리보드용 프레임이 없습니다.")
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Qwen 스토리보드 생성에는 opencv-python과 numpy가 필요합니다.") from exc

    # Square face crops must remain square.  The previous 640x220 resize
    # compressed eyelids vertically, exactly the visual detail this evidence
    # needs to preserve.
    if not 160 <= panel_size <= 512:
        raise ValueError("event storyboard panel size must be in [160, 512]")
    panel_width = panel_height = int(panel_size)
    panel_count = min(4, len(frames))
    indices = np.linspace(0, len(frames) - 1, panel_count).round().astype(int)
    panels: list[Any] = []
    for index in indices:
        timestamp, frame = frames[int(index)]
        source_height, source_width = frame.shape[:2]
        scale = min(
            panel_width / max(source_width, 1),
            panel_height / max(source_height, 1),
        )
        resized = cv2.resize(
            frame,
            (
                max(1, int(round(source_width * scale))),
                max(1, int(round(source_height * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        top = (panel_height - resized.shape[0]) // 2
        left = (panel_width - resized.shape[1]) // 2
        panel[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
        cv2.rectangle(panel, (0, 0), (220, 34), (0, 0, 0), -1)
        cv2.putText(
            panel, f"TIME {timestamp:.2f}s", (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA,
        )
        panels.append(panel)
    while len(panels) < 4:
        panels.append(np.zeros_like(panels[0]))
    gap = np.zeros((panel_height, 8, 3), dtype=np.uint8)
    top = cv2.hconcat([panels[0], gap, panels[1]])
    bottom = cv2.hconcat([panels[2], gap, panels[3]])
    canvas = cv2.vconcat([top, np.zeros((8, top.shape[1], 3), dtype=np.uint8), bottom])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise RuntimeError(f"Qwen 스토리보드 저장에 실패했습니다: {output_path}")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Qwen 스토리보드 생성에 실패했습니다: {output_path}")
    if output_path.stat().st_size > 2 * 1024 * 1024:
        raise RuntimeError(f"Qwen 스토리보드가 2 MiB를 초과합니다: {output_path}")
    return output_path


def print_qwen_event_result(
    bridge_result: dict[str, Any], metadata: dict[str, Any], print_json: bool,
) -> str:
    """Print Qwen's final fields plus model-only and end-to-end latency."""
    state = bridge_result.get("driver_state", {})
    if not isinstance(state, dict):
        state = {}
    response = str(state.get("driver_response", "unknown"))
    model_ms = bridge_result.get("inference_ms")
    model_sec = float(model_ms) / 1000.0 if isinstance(model_ms, (int, float)) else None
    end_to_end_sec = time.monotonic() - float(metadata["submitted_monotonic"])
    print(
        f"Qwen event 완료: event={metadata['event_id']} "
        f"qwen_video_start_t={float(metadata['event_video_start_sec']):.1f}s "
        f"response={response}",
        flush=True,
    )
    print(
        "Qwen 판정: " + ", ".join(
            f"{name}={state.get(name, 'unknown')}" for name in (
                "eye_state", "head_pose", "upper_body_posture", "hand_on_wheel",
                "voluntary_motion", "recovery_status", "driver_response",
            )
        ),
        flush=True,
    )
    print(
        "Qwen 추론 시간: "
        + (f"{model_sec:.2f}s (브리지 모델 호출)" if model_sec is not None else "unknown")
        + f" | 전체 이벤트 시간: {end_to_end_sec:.2f}s",
        flush=True,
    )
    if print_json:
        print(json.dumps(bridge_result, ensure_ascii=False, indent=2), flush=True)
    return response


def run_full_video_realtime(
    args: argparse.Namespace,
    headers: dict[str, str],
    preview: "ImagePreview",
) -> int:
    """Run one synchronous VLM inference per non-overlapping ROI video clip.

    The input remains one full source MP4, but the VLM receives consecutive
    fixed-duration driver-side ROI clips. This avoids resubmitting a nearly
    identical rolling clip for every source frame.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("--video-path 모드에는 opencv-python이 필요합니다.") from exc

    source_video = args.video_path.resolve()
    if not source_video.is_file():
        raise FileNotFoundError(f"입력 영상이 없습니다: {source_video}")

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {source_video}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if src_fps <= 0.0:
        cap.release()
        raise RuntimeError("입력 영상 FPS를 읽을 수 없습니다.")

    duration = total_frames / src_fps
    start_sec = max(0.0, float(args.video_start_sec))
    end_sec = (
        duration
        if args.video_max_sec is None
        else min(duration, start_sec + args.video_max_sec)
    )
    if start_sec >= end_sec:
        cap.release()
        raise ValueError("--video-start-sec/--video-max-sec 범위가 올바르지 않습니다.")
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
    frame_index = int(round(start_sec * src_fps))

    selector_module: Any | None = None
    selector: Any | None = None
    last_driver_box: tuple[int, int, int, int] | None = None
    last_driver_box_time = -float("inf")
    last_driver_info: dict[str, Any] | None = None
    driver_yolo_frame_count = 0
    if args.yolo_driver_crop:
        selector_module = load_driver_selector_module()
        selector = selector_module.DriverSelector(
            model_path=str(args.driver_yolo_model), tracker=args.driver_tracker,
            anchor_x=args.driver_anchor_x, anchor_y=args.driver_anchor_y, fps=src_fps,
            confidence=args.driver_yolo_confidence,
            min_driver_x_ratio=args.driver_min_x_ratio,
        )

    output_dir = (args.event_output_dir / source_video.stem).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "vlm_clips.jsonl"
    observation_id = args.start_observation_id
    processed = 0
    successful = 0
    wall_start = time.perf_counter()
    clip_frames: list[tuple[float, Any]] = []
    clip_reference_box: tuple[int, int, int, int] | None = None
    clip_crop_box: tuple[int, int, int, int] | None = None
    clip_crop_source = "fixed_roi"

    print(
        f"VLM 클립 분석 시작: {source_video.name} ({duration:.1f}s), "
        f"{args.vlm_clip_seconds:.2f}초 비중첩 "
        f"{'YOLO 운전자 고정-box' if args.yolo_driver_crop else '고정 운전자 ROI'} MP4를 전송합니다.",
        flush=True,
    )

    def submit_clip() -> None:
        """Encode and submit the buffered sequential clip once."""
        nonlocal observation_id, processed, successful
        if not clip_frames:
            return
        clip_start_time = clip_frames[0][0]
        clip_end_time = clip_frames[-1][0] + 1.0 / src_fps
        clip_path = output_dir / f".vlm_clip_{observation_id:08d}.mp4"
        try:
            encode_event_video(
                clip_frames, clip_path, src_fps, args.vlm_clip_max_width
            )
            media_bytes = clip_path.read_bytes()
        except Exception:
            clip_path.unlink(missing_ok=True)
            raise
        payload: dict[str, Any] = {
            "protocol_version": PROTOCOL_VERSION,
            "observation_id": observation_id,
            "captured_at_unix_s": time.time(),
            "cabin_media": {
                "mime_type": "video/mp4",
                "data_base64": base64.b64encode(media_bytes).decode("ascii"),
            },
            "source_video": source_video.name,
            "video_time_sec": round(clip_end_time, 6),
            "clip_start_time_sec": round(clip_start_time, 6),
            "clip_end_time_sec": round(clip_end_time, 6),
        }
        if not args.no_inline_telemetry and not args.vlm_only:
            telemetry, telemetry_age_s = latest_telemetry(args, headers)
            if telemetry is not None:
                payload["telemetry"] = telemetry
                payload["driving_summary"] = driving_summary(telemetry)
            if telemetry_age_s is not None:
                payload["telemetry_age_s"] = telemetry_age_s

        submitted = time.monotonic()
        try:
            preview.set_vlm_inference_running(clip_end_time)
            result = request_json(
                "POST", args.bridge_url.rstrip("/") + "/monitor", headers,
                payload, args.bridge_timeout,
            )
            state = result.get("driver_state", {})
            if not isinstance(state, dict):
                state = {}
            response = str(state.get("driver_response", "unknown"))
            preview.set_vlm_result(response)
            print(
                "VLM 클립 판정: "
                f"t={clip_start_time:.3f}-{clip_end_time:.3f}s "
                f"eye_state={state.get('eye_state', 'unknown')} "
                f"head_pose={state.get('head_pose', 'unknown')} "
                f"upper_body_posture={state.get('upper_body_posture', 'unknown')} "
                f"driver_response={response} "
                f"inference_ms={result.get('inference_ms', 'unknown')}",
                flush=True,
            )
            if args.print_json:
                print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            record: dict[str, Any] = {
                "source_video": str(source_video),
                "frame_index": frame_index - 1,
                "video_time": round(clip_end_time, 6),
                "event_video_start_sec": round(clip_start_time, 6),
                "event_video_end_sec": round(clip_end_time, 6),
                "trigger": "vlm_clip",
                "driver_crop_mode": "yolo_fixed_box" if args.yolo_driver_crop else "fixed_roi",
                "driver_crop_source": clip_crop_source,
                "driver_reference_box": list(clip_reference_box) if clip_reference_box else None,
                "driver_crop_box": list(clip_crop_box) if clip_crop_box else None,
                "qwen_model_inference_ms": result.get("inference_ms"),
                "event_end_to_end_sec": round(time.monotonic() - submitted, 4),
                "bridge_result": result,
            }
            successful += 1
        except Exception as exc:
            preview.set_vlm_error()
            print(
                f"VLM 클립 추론 실패: t={clip_start_time:.3f}-{clip_end_time:.3f}s: {exc}",
                file=sys.stderr, flush=True,
            )
            record = {
                "source_video": str(source_video),
                "frame_index": frame_index - 1,
                "video_time": round(clip_end_time, 6),
                "event_video_start_sec": round(clip_start_time, 6),
                "event_video_end_sec": round(clip_end_time, 6),
                "trigger": "vlm_clip",
                "driver_crop_mode": "yolo_fixed_box" if args.yolo_driver_crop else "fixed_roi",
                "driver_crop_source": clip_crop_source,
                "driver_reference_box": list(clip_reference_box) if clip_reference_box else None,
                "driver_crop_box": list(clip_crop_box) if clip_crop_box else None,
                "error": str(exc),
            }
        with results_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        clip_path.unlink(missing_ok=True)
        observation_id += 1
        processed += 1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_time = frame_index / src_fps
            frame_index += 1
            if frame_time > end_sec + 1e-9:
                break

            driver_box_source = "fallback_roi"
            if args.yolo_driver_crop:
                assert selector is not None and selector_module is not None
                run_driver_yolo = (
                    driver_yolo_frame_count % args.driver_yolo_interval == 0
                )
                driver_yolo_frame_count += 1
                if run_driver_yolo:
                    last_driver_info = selector.update(frame)
                    driver_info = last_driver_info
                elif last_driver_info is None:
                    driver_info = None
                else:
                    driver_info = {**last_driver_info, "held": True}

                driver_box = None
                if driver_info is not None:
                    driver_box = selector_module.expand_driver_box(
                        driver_info["box"], frame.shape
                    )
                if driver_box is not None:
                    last_driver_box = driver_box
                    last_driver_box_time = frame_time
                    driver_box_source = (
                        "held" if driver_info.get("held", False) else "tracked"
                    )
                elif (
                    last_driver_box is not None
                    and frame_time - last_driver_box_time <= args.driver_box_hold_sec
                ):
                    driver_box = last_driver_box
                    driver_box_source = "held"
                else:
                    driver_box = normalized_roi_to_box(args.fallback_driver_roi, frame.shape)
                    driver_box_source = "fallback_roi"

                if driver_box is None:
                    raise RuntimeError("운전자 ROI를 프레임 크기에 맞게 만들 수 없습니다.")
                if clip_reference_box is None:
                    clip_reference_box = driver_box
                    clip_crop_source = driver_box_source
                driver_crop, crop_box = crop_fixed_driver_box(
                    frame, clip_reference_box, args.vlm_fixed_driver_box_padding,
                    args.vlm_driver_seat_left_ratio,
                )
            else:
                driver_box = normalized_roi_to_box(args.fallback_driver_roi, frame.shape)
                if driver_box is None:
                    raise RuntimeError("고정 운전자 ROI를 프레임 크기에 맞게 만들 수 없습니다.")
                x1, y1, x2, y2 = driver_box
                driver_crop = frame[y1:y2, x1:x2]
                if driver_crop.size == 0:
                    raise RuntimeError("고정 운전자 ROI crop이 비어 있습니다.")
                crop_box = driver_box
                if clip_reference_box is None:
                    clip_reference_box = driver_box
                    clip_crop_source = "fixed_roi"
            clip_crop_box = crop_box
            x1, y1, x2, y2 = crop_box

            display_frame = frame.copy()
            if args.draw_driver_box:
                # The displayed box is fixed for the full VLM clip, so label
                # it by the source that established that fixed box rather
                # than a later detector update within the same clip.
                is_fallback = clip_crop_source in ("fallback_roi", "fixed_roi")
                box_color = (0, 180, 255) if is_fallback else (40, 220, 40)
                box_label = "DRIVER (FALLBACK ROI)" if is_fallback else "DRIVER (YOLO)"
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 3)
                cv2.putText(
                    display_frame,
                    box_label,
                    (x1, max(24, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2, cv2.LINE_AA,
                )
            if preview.show_frame(display_frame, frame_time):
                break

            clip_frames.append((frame_time, driver_crop.copy()))
            clip_duration = frame_time - clip_frames[0][0] + 1.0 / src_fps
            if clip_duration + 1e-9 >= args.vlm_clip_seconds:
                submit_clip()
                clip_frames.clear()
                clip_reference_box = None
                clip_crop_box = None
                clip_crop_source = "fixed_roi"

            if args.realtime:
                target_elapsed = frame_time - start_sec
                sleep_seconds = target_elapsed - (time.perf_counter() - wall_start)
                if sleep_seconds > 0:
                    preview.wait(sleep_seconds)
        # Preserve a final shorter tail clip instead of silently dropping it.
        if clip_frames:
            submit_clip()
    finally:
        cap.release()

    print(f"VLM 클립 결과 저장: {results_path} ({processed} clips)", flush=True)
    if args.ground_truth_file is not None and successful == processed and processed:
        evaluate_qwen_event_results(
            args,
            results_path,
            processed,
            match_video_time=True,
            plot_path_override=args.accuracy_plot_file,
        )
        evaluate_jsonl(
            results_path,
            args.ground_truth_file,
            output_dir,
            source="vlm",
            title="VLM-only",
            last_records=processed,
            observation_duration_sec=end_sec - start_sec,
            coverage_type="continuous_timeline",
        )
    elif args.ground_truth_file is not None and processed:
        print("VLM 실패 프레임이 있어 이번 실행의 정확도 평가는 건너뜁니다.", flush=True)
    return 0
def list_images(frames_dir: Path) -> list[Path]:
    if not frames_dir.exists():
        raise FileNotFoundError(f"이미지 디렉터리가 없습니다: {frames_dir}")
    images = [
        path for path in frames_dir.iterdir()
        if path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES
    ]
    return sorted(images, key=image_sort_key)


def newest_image(args: argparse.Namespace) -> Path | None:
    if args.image_path is not None:
        return args.image_path if args.image_path.exists() else None
    images = list_images(args.frames_dir)
    return images[-1] if images else None


def image_signature(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return (str(path), int(stat.st_mtime_ns), int(stat.st_size))


class ImagePreview:
    """Optional OpenCV preview of media currently sent to the bridge."""

    WINDOW_NAME = "DGX Driver Monitor - Current Frame"

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.cv2: Any = None
        if not enabled:
            return
        try:
            system_font_dir = "/usr/share/fonts/opentype/urw-base35"
            os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
            if os.path.isdir(system_font_dir):
                os.environ["QT_QPA_FONTDIR"] = system_font_dir
            import cv2

            # The OpenCV wheel overwrites QT_QPA_FONTDIR while importing.
            # Restore a real system font directory before Qt initializes.
            if os.path.isdir(system_font_dir):
                os.environ["QT_QPA_FONTDIR"] = system_font_dir

            self.cv2 = cv2
            self.vlm_inference_active = False
            self.vlm_live_result: str | None = None
            self.vlm_result_visible_until = 0.0
            self.vlm_events: deque[dict[str, Any]] = deque(maxlen=120)
            self._latest_video_frame: Any | None = None
            self._latest_video_time = 0.0
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
            # The VLM-only view keeps the VLM response timeline, but never
            # renders the TCN risk chart used by the fusion runner.
            cv2.resizeWindow(self.WINDOW_NAME, 960, 1000)
            cv2.startWindowThread()
        except Exception as exc:
            self.enabled = False
            self.cv2 = None
            print(
                f"이미지 미리보기 비활성화: OpenCV 창을 열 수 없습니다: {exc}",
                file=sys.stderr,
                flush=True,
            )

    def show(
        self, image_path: Path, label: str, result: dict[str, Any] | None = None
    ) -> None:
        if not self.enabled or self.cv2 is None:
            return
        if image_path.suffix.lower() in VIDEO_SUFFIXES:
            # Video input is still sent to the monitor, but is not rendered
            # or played in the local preview window.
            return

        image = self.cv2.imread(str(image_path), self.cv2.IMREAD_COLOR)
        if image is None:
            print(
                f"미디어 미리보기 실패: {image_path}",
                file=sys.stderr,
                flush=True,
            )
            return
        image = self._prepare_frame(image)
        self.cv2.imshow(self.WINDOW_NAME, image)
        self.cv2.waitKey(1)

    def show_frame(self, frame: Any, video_time: float) -> bool:
        """Render video time, driver box, and the live VLM state only."""
        if not self.enabled or self.cv2 is None:
            return False
        self._latest_video_frame = frame.copy()
        self._latest_video_time = float(video_time)
        return self._present_latest_frame()

    def set_vlm_inference_running(self, video_time: float) -> None:
        if not self.enabled:
            return
        self.vlm_inference_active = True
        self.vlm_live_result = None
        self.vlm_result_visible_until = 0.0
        self.vlm_events.append({"video_time": float(video_time), "response": None})
        self._present_latest_frame()

    def set_vlm_result(self, response: str) -> None:
        if not self.enabled:
            return
        self.vlm_inference_active = False
        if response in DRIVER_RESPONSE_LABELS:
            self.vlm_live_result = response
            self.vlm_result_visible_until = time.monotonic() + 2.0
            for event in reversed(self.vlm_events):
                if event["response"] is None:
                    event["response"] = response
                    break
        else:
            self.vlm_live_result = None
            self.vlm_result_visible_until = 0.0
        self._present_latest_frame()

    def set_vlm_error(self) -> None:
        if not self.enabled:
            return
        self.vlm_inference_active = False
        self.vlm_live_result = "error"
        self.vlm_result_visible_until = time.monotonic() + 2.0
        self._present_latest_frame()

    def _prepare_frame(self, image: Any) -> Any:
        """Resize a source frame without adding the legacy VLM-only header."""
        height, width = image.shape[:2]
        scale = min(960.0 / width, 700.0 / height, 1.0)
        if scale < 1.0:
            image = self.cv2.resize(
                image,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=self.cv2.INTER_AREA,
            )
        return image

    def _present_latest_frame(self) -> bool:
        """Present the most recent video frame with no legacy status overlay."""
        if (
            not self.enabled or self.cv2 is None
            or self._latest_video_frame is None
        ):
            return False
        import numpy as np

        cv2 = self.cv2
        image = self._prepare_frame(self._latest_video_frame.copy())
        video_time_label = f"Video time: {self._latest_video_time:.1f}s"
        text_size, baseline = cv2.getTextSize(
            video_time_label, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2
        )
        left, top = 8, 8
        cv2.rectangle(
            image, (left - 4, top - 4),
            (left + text_size[0] + 4, top + text_size[1] + baseline + 4),
            (0, 0, 0), -1,
        )
        cv2.putText(
            image, video_time_label, (left, top + text_size[1]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA,
        )

        panel = self._build_vlm_timeline_panel(image.shape[1], np)
        self.cv2.imshow(self.WINDOW_NAME, cv2.vconcat([image, panel]))
        return (self.cv2.waitKey(1) & 0xFF) == ord("q")

    def _build_vlm_timeline_panel(self, width: int, np: Any) -> Any:
        """Render VLM requests and event-specific driver responses only."""
        cv2 = self.cv2
        panel = np.full((254, width, 3), (20, 20, 20), dtype=np.uint8)
        left, right = 112, max(114, width - 14)
        lane_top, lane_bottom = 58, 161
        chart_width = max(1, right - left)
        now_time = self._latest_video_time
        start_time = now_time - 12.0
        colors = {
            "active": (70, 190, 70),
            "reduced": (0, 210, 240),
            "no_visible_response": (45, 45, 230),
        }
        lane_centers = {
            "no_visible_response": 75,
            "reduced": 109,
            "active": 143,
        }
        lane_labels = {
            "no_visible_response": "NO VISIBLE RESPONSE",
            "reduced": "REDUCED",
            "active": "ACTIVE",
        }
        cv2.putText(panel, "VLM INFERENCE", (left, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (230, 230, 230), 2, cv2.LINE_AA)
        legend_x = left + 172
        for color, label in (
            (colors["active"], "Active"),
            (colors["reduced"], "Reduced"),
            (colors["no_visible_response"], "No visible response"),
        ):
            cv2.rectangle(panel, (legend_x, 14), (legend_x + 8, 22), color, -1)
            cv2.putText(panel, label, (legend_x + 13, 23), cv2.FONT_HERSHEY_SIMPLEX,
                        0.36, (220, 220, 220), 1, cv2.LINE_AA)
            label_width = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1
            )[0][0]
            legend_x += label_width + 29

        for tick_index in range(5):
            fraction = tick_index / 4.0
            x = left + int(round(fraction * chart_width))
            cv2.line(panel, (x, lane_top), (x, lane_bottom), (55, 55, 55), 1, cv2.LINE_AA)
            label = f"{start_time + fraction * 12.0:.1f}s"
            label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)[0][0]
            cv2.putText(panel, label, (x - label_width // 2, 182),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (185, 185, 185), 1, cv2.LINE_AA)
        for state, lane_y in lane_centers.items():
            cv2.rectangle(panel, (left, lane_y - 9), (right, lane_y + 9), (34, 34, 34), -1)
            cv2.line(panel, (left, lane_y + 10), (right, lane_y + 10),
                     (70, 70, 70), 1, cv2.LINE_AA)
            label = lane_labels[state]
            if state == "no_visible_response":
                cv2.putText(panel, "NO VISIBLE", (28, lane_y - 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.32, colors[state], 1, cv2.LINE_AA)
                cv2.putText(panel, "RESPONSE", (28, lane_y + 9), cv2.FONT_HERSHEY_SIMPLEX,
                            0.32, colors[state], 1, cv2.LINE_AA)
            else:
                cv2.putText(panel, label, (28, lane_y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.32, colors[state], 1, cv2.LINE_AA)

        visible_events = [
            event for event in self.vlm_events
            if float(event["video_time"]) >= start_time
        ]
        # Neutral request markers show the VLM inference cadence.  There is no
        # TCN curve or TCN-trigger colour in this VLM-only panel.
        for event in visible_events:
            event_time = float(event["video_time"])
            x = left + int(round((event_time - start_time) / 12.0 * chart_width))
            x = max(left, min(right, x))
            cv2.line(panel, (x, lane_top), (x, lane_bottom), (155, 155, 155), 1, cv2.LINE_AA)
        # Draw verdict squares after all request markers so the outcome always
        # remains in the foreground.
        for event_index, event in enumerate(visible_events):
            if event["response"] not in colors:
                continue
            event_time = float(event["video_time"])
            x = left + int(round((event_time - start_time) / 12.0 * chart_width))
            x = max(left, min(right, x))
            state = str(event["response"])
            lane_y = lane_centers[state]
            cv2.rectangle(panel, (x - 6, lane_y - 6), (x + 6, lane_y + 6),
                          colors[state], -1)
            timestamp = f"{event_time:.1f}s"
            label_width = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)[0][0]
            label_x = max(left, min(right - label_width, x - label_width // 2))
            label_y = 198 if event_index % 2 == 0 else 211
            cv2.putText(panel, timestamp, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, (210, 210, 210), 1, cv2.LINE_AA)

        result_labels = {
            "active": "ACTIVE", "reduced": "REDUCED",
            "no_visible_response": "NO VISIBLE RESPONSE", "error": "ERROR",
        }
        if self.vlm_inference_active:
            status_label, status_color = "VLM INFERENCE RUNNING", (255, 210, 80)
        elif (
            self.vlm_live_result in result_labels
            and time.monotonic() < self.vlm_result_visible_until
        ):
            result = str(self.vlm_live_result)
            status_label = f"VLM RESULT: {result_labels[result]}"
            status_color = colors.get(result, (45, 45, 230))
        else:
            status_label, status_color = "VLM WAITING - NOT INFERENCING", (190, 190, 190)
        cv2.rectangle(panel, (left, 218), (right, 244), (42, 42, 42), -1)
        cv2.rectangle(panel, (left, 218), (right, 244), (85, 85, 85), 1)
        cv2.putText(panel, status_label, (left + 10, 238), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, status_color, 2, cv2.LINE_AA)
        return panel

    def pump(self, delay_ms: int = 30) -> None:
        """Keep the native window responsive while model inference is running."""
        if not self.enabled or self.cv2 is None:
            return
        self.cv2.waitKey(max(1, delay_ms))

    def wait(self, seconds: float) -> None:
        """Wait without freezing the preview window."""
        if not self.enabled or self.cv2 is None:
            time.sleep(max(0.0, seconds))
            return
        deadline = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < deadline:
            remaining_ms = int((deadline - time.monotonic()) * 1000.0)
            self.pump(min(50, max(1, remaining_ms)))

    def close(self) -> None:
        if self.cv2 is None:
            return
        try:
            self.cv2.destroyWindow(self.WINDOW_NAME)
            self.cv2.waitKey(1)
        except Exception:
            pass


def post_monitor_responsive(
    args: argparse.Namespace,
    image_path: Path,
    observation_id: int,
    headers: dict[str, str],
    preview: ImagePreview,
) -> dict[str, Any]:
    """Run blocking HTTP inference off the GUI thread when preview is enabled."""
    if not preview.enabled:
        return post_monitor(args, image_path, observation_id, headers)

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="monitor-http")
    future = executor.submit(
        post_monitor, args, image_path, observation_id, headers)
    try:
        while not future.done():
            preview.pump(30)
        return future.result()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def post_monitor_when_available(
    args: argparse.Namespace,
    image_path: Path,
    observation_id: int,
    headers: dict[str, str],
    preview: ImagePreview,
) -> dict[str, Any]:
    """Retry the same frame when another inference still owns the bridge."""
    busy_reported = False
    while True:
        try:
            return post_monitor_responsive(
                args, image_path, observation_id, headers, preview)
        except RuntimeError as exc:
            status_code = getattr(exc, "status_code", None)
            is_busy = status_code == 429 or "HTTP 429" in str(exc)
            if not is_busy:
                raise
            if not busy_reported:
                print(
                    "모델이 이전 요청을 처리 중입니다. 현재 프레임을 유지하고 "
                    "1초마다 자동 재시도합니다.",
                    flush=True,
                )
                busy_reported = True
            preview.wait(1.0)


def print_result(result: dict[str, Any], image_path: Path, print_json: bool) -> None:
    if print_json:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return

    state = result.get("driver_state", {})
    if not isinstance(state, dict):
        state = {}
    if "driver_response" in state and "risk_level" not in state:
        print(
            "DRIVER MONITOR result "
            f"image={image_path.name} "
            f"eye_state={state.get('eye_state')} "
            f"head_pose={state.get('head_pose')} "
            f"upper_body_posture={state.get('upper_body_posture')} "
            f"driver_response={state.get('driver_response')} "
            f"inference_ms={result.get('inference_ms')}",
            flush=True,
        )
        return
    supervisor = result.get("safety_supervisor", {})
    if not isinstance(supervisor, dict):
        supervisor = {}
    intervention = supervisor.get("selected_intervention", {})
    if not isinstance(intervention, dict):
        intervention = {}
    print(
        "DRIVER MONITOR result "
        f"image={image_path.name} "
        f"risk={state.get('risk_level')} "
        f"eye_state={state.get('eye_state')} "
        f"head_pose={state.get('head_pose')} "
        f"upper_body_posture={state.get('upper_body_posture')} "
        f"gaze={state.get('gaze')} "
        f"steering_wheel_contact={state.get('steering_wheel_contact')} "
        f"driver_state={state.get('driver_state')} "
        f"unconscious={state.get('unconscious')} "
        f"confidence={state.get('confidence')} "
        f"intervention={intervention.get('intervention_type')} "
        f"inference_ms={result.get('inference_ms')} "
        f"risk_reason={state.get('risk_reason', '')}",
        flush=True,
    )


def save_result(
    result: dict[str, Any], image_path: Path, output: Path | None
) -> None:
    if output is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {"image_path": str(image_path), **result}
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_evaluation_records(
    path: Path, last_records: int = 0
) -> list[tuple[Path, dict[str, Any]]]:
    """Load saved monitor responses for plotting without another inference run."""
    if not path.is_file():
        raise FileNotFoundError(f"결과 JSONL 파일이 없습니다: {path}")
    records: list[tuple[Path, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"결과 JSONL {line_number}행 JSON 형식 오류: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"결과 JSONL {line_number}행이 객체가 아닙니다.")
            image_path = record.pop("image_path", None)
            if not isinstance(image_path, str) or not image_path:
                raise ValueError(
                    f"결과 JSONL {line_number}행 image_path가 없습니다."
                )
            records.append((Path(image_path), record))
    if last_records > 0:
        records = records[-last_records:]
    if not records:
        raise ValueError("채점할 저장 추론 결과가 없습니다.")
    return records


def load_ground_truth(path: Path) -> dict[str, str]:
    """Load and strictly validate filename-keyed driver-response annotations."""
    if not path.is_file():
        raise FileNotFoundError(f"정답 CSV가 없습니다: {path}")
    annotations: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"clip_name", "driver_response"}
        missing_columns = required - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                "정답 CSV 필수 열 누락: " + ", ".join(sorted(missing_columns))
            )
        for line_number, row in enumerate(reader, 2):
            clip_name = Path((row.get("clip_name") or "").strip()).name
            if not clip_name:
                raise ValueError(f"정답 CSV {line_number}행 clip_name이 비어 있습니다.")
            if clip_name in annotations:
                raise ValueError(f"정답 CSV에 중복 clip_name: {clip_name}")
            value = (row.get("driver_response") or "").strip().lower()
            if value not in DRIVER_RESPONSE_LABELS:
                allowed = ", ".join(DRIVER_RESPONSE_LABELS)
                raise ValueError(
                    f"정답 CSV {line_number}행 driver_response={value!r} 오류 "
                    f"(허용값: {allowed})"
                )
            annotations[clip_name] = value
    if not annotations:
        raise ValueError(f"정답 CSV에 데이터가 없습니다: {path}")
    return annotations


def select_matching_ground_truth(
    records: list[tuple[Path, dict[str, Any]]], args: argparse.Namespace
) -> dict[str, str]:
    """Prefer a sibling CSV whose clip set exactly matches this evaluation run."""
    if args.ground_truth_file is None:
        return {}
    requested = load_ground_truth(args.ground_truth_file)
    record_names = {image_path.name for image_path, _ in records}
    if set(requested) == record_names:
        return requested

    candidates = sorted(args.ground_truth_file.parent.glob("*ground_truth*.csv"))
    for candidate in candidates:
        if candidate == args.ground_truth_file:
            continue
        try:
            candidate_truth = load_ground_truth(candidate)
        except (OSError, ValueError):
            continue
        if set(candidate_truth) == record_names:
            print(f"부분 정답 CSV 자동 선택: {candidate}", flush=True)
            return candidate_truth
    return requested


def evaluate_and_plot(
    records: list[tuple[Path, dict[str, Any]]], args: argparse.Namespace
) -> None:
    """Plot per-class and overall exact-match driver-response accuracy."""
    if args.ground_truth_file is None:
        return
    if not records:
        raise ValueError("채점할 추론 결과가 없습니다.")

    ground_truth = select_matching_ground_truth(records, args)
    seen: set[str] = set()
    class_total = {label: 0 for label in DRIVER_RESPONSE_LABELS}
    class_correct = {label: 0 for label in DRIVER_RESPONSE_LABELS}
    overall_correct = 0
    for image_path, result in records:
        clip_name = image_path.name
        if clip_name in seen:
            raise ValueError(f"추론 결과에 중복 영상이 있습니다: {clip_name}")
        seen.add(clip_name)
        truth = ground_truth.get(clip_name)
        if truth is None:
            raise ValueError(f"정답 CSV에 영상이 없습니다: {clip_name}")
        state = result.get("driver_state")
        if not isinstance(state, dict):
            raise ValueError(f"driver_state가 없는 추론 결과: {clip_name}")
        prediction = str(state.get("driver_response", "")).strip().lower()
        if prediction not in DRIVER_RESPONSE_LABELS:
            raise ValueError(
                f"{clip_name}의 VLM driver_response 예측값이 유효하지 않습니다: "
                f"{prediction!r}"
            )
        matched = prediction == truth
        class_total[truth] += 1
        class_correct[truth] += int(matched)
        overall_correct += int(matched)
    missing_predictions = set(ground_truth) - seen
    if missing_predictions and not args.allow_partial_evaluation:
        example = sorted(missing_predictions)[0]
        raise ValueError(
            f"정답 CSV의 {len(missing_predictions)}개 영상을 처리하지 않았습니다. "
            f"예: {example}"
        )

    total = len(records)
    accuracy = {
        label: (
            class_correct[label] * 100.0 / class_total[label]
            if class_total[label] > 0 else None
        )
        for label in DRIVER_RESPONSE_LABELS
    }
    accuracy["overall"] = overall_correct * 100.0 / total
    inference_times = [
        float(result["inference_ms"])
        for _, result in records
        if isinstance(result.get("inference_ms"), (int, float))
    ]
    average_inference_ms = (
        sum(inference_times) / len(inference_times)
        if inference_times else None
    )
    # Preserve previous evaluation plots. If the requested filename already
    # exists, create a numbered sibling instead of overwriting it.
    plot_path = args.accuracy_plot_file
    if plot_path.exists():
        suffix = plot_path.suffix
        stem = plot_path.stem
        number = 1
        while True:
            candidate = plot_path.with_name(f"{stem}_{number}{suffix}")
            if not candidate.exists():
                plot_path = candidate
                break
            number += 1
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    if not args.show_accuracy_plot:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_labels = (*DRIVER_RESPONSE_LABELS, "overall")
    display_names = ("Active", "Reduced", "No Visible\nResponse", "Overall")
    # A class with no ground-truth samples is displayed as N/A rather than
    # aborting evaluation or misleadingly reporting 0% accuracy.
    values = [accuracy[label] if accuracy[label] is not None else 0.0
              for label in plot_labels]
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(display_names, values, marker="o", linewidth=2.2, markersize=8)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Accuracy (%)")
    axis.set_title(f"Driver Response Accuracy (n={total})")
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    for index, label in enumerate(plot_labels):
        point_correct = overall_correct if label == "overall" else class_correct[label]
        point_total = total if label == "overall" else class_total[label]
        if accuracy[label] is None:
            annotation = "N/A (0/0)"
        else:
            annotation = f"{accuracy[label]:.1f}% ({point_correct}/{point_total})"
        axis.annotate(
            annotation,
            (index, values[index]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
        )
    figure.tight_layout(rect=(0, 0, 0.78, 1))
    if average_inference_ms is None:
        inference_text = "Average inference time\nN/A"
    else:
        inference_text = (
            "Average inference time\n"
            f"{average_inference_ms:.2f} ms\n"
            f"(n={len(inference_times)})"
        )
    figure.text(
        0.98,
        0.5,
        inference_text,
        ha="right",
        va="center",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f2f2f2", "edgecolor": "#999999"},
    )
    figure.savefig(plot_path, dpi=160)
    print(f"정확도 그래프 저장: {plot_path}", flush=True)
    if args.show_accuracy_plot:
        plt.show()
    plt.close(figure)


def parse_ground_truth_clip_interval(clip_name: str) -> tuple[float, float] | None:
    """Return the [start, end) seconds encoded in a standard GT clip name."""
    match = re.fullmatch(r"clip_(\d+)s_to_(\d+)s\.mp4", Path(clip_name).name)
    if match is None:
        return None
    start_sec, end_sec = (float(value) for value in match.groups())
    return (start_sec, end_sec) if end_sec > start_sec else None


def event_evidence_start_sec(record: dict[str, Any]) -> float:
    """Get the start of the actual Qwen video, including old result files."""
    value = record.get("event_video_start_sec")
    if isinstance(value, (int, float)):
        return float(value)
    video_time = record.get("video_time")
    if isinstance(video_time, (int, float)):
        return float(video_time)
    raise ValueError("event_video_start_sec 또는 video_time이 없습니다.")


def load_event_evaluation_records(path: Path, last_records: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Qwen 이벤트 JSONL 파일이 없습니다: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"이벤트 JSONL {line_number}행 JSON 형식 오류: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"이벤트 JSONL {line_number}행이 객체가 아닙니다.")
            records.append(record)
    if last_records > 0:
        records = records[-last_records:]
    if not records:
        raise ValueError("채점할 Qwen 이벤트 결과가 없습니다.")
    return records


def evaluate_qwen_event_results(
    args: argparse.Namespace,
    results_path: Path | None = None,
    last_records: int | None = None,
    *,
    match_video_time: bool = False,
    plot_path_override: Path | None = None,
) -> None:
    """Score recorded Qwen responses against time-indexed GT clip intervals."""
    source_path = results_path or args.evaluate_event_results_file
    if source_path is None or args.ground_truth_file is None:
        raise ValueError("이벤트 Qwen 평가에는 결과 JSONL과 --ground-truth-file이 필요합니다.")
    annotations = load_ground_truth(args.ground_truth_file)
    timed_truth: list[tuple[float, float, str, str]] = []
    for clip_name, label in annotations.items():
        interval = parse_ground_truth_clip_interval(clip_name)
        if interval is not None:
            timed_truth.append((interval[0], interval[1], clip_name, label))
    if not timed_truth:
        raise ValueError("이벤트 평가는 clip_000000s_to_000004s.mp4 형식의 GT가 필요합니다.")
    timed_truth.sort()

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(
        load_event_evaluation_records(
            source_path,
            args.evaluate_last_event_records if last_records is None else last_records,
        ), 1,
    ):
        result = record.get("bridge_result")
        state = result.get("driver_state") if isinstance(result, dict) else None
        prediction = state.get("driver_response") if isinstance(state, dict) else None
        if not isinstance(prediction, str) or prediction.strip().lower() not in DRIVER_RESPONSE_LABELS:
            raise ValueError(f"이벤트 {index}에 유효한 Qwen driver_response가 없습니다.")
        evidence_start = event_evidence_start_sec(record)
        video_time = record.get("video_time")
        if match_video_time and isinstance(video_time, (int, float)):
            match_time = float(video_time)
            match_basis = "video_time"
        else:
            match_time = evidence_start
            match_basis = "event_video_start_sec"
        candidates = [item for item in timed_truth if item[0] <= match_time < item[1]]
        if not candidates:
            earlier = [item for item in timed_truth if item[0] <= match_time]
            candidates = [max(earlier, key=lambda item: item[0])] if earlier else [timed_truth[0]]
        _, _, gt_clip, truth = max(candidates, key=lambda item: item[0])
        inference_ms = record.get("qwen_model_inference_ms")
        if not isinstance(inference_ms, (int, float)) and isinstance(result, dict):
            inference_ms = result.get("inference_ms")
        rows.append({
            "event_row": index,
            "event_id": record.get("event_id"),
            "video_time_sec": round(float(video_time), 3) if isinstance(video_time, (int, float)) else "",
            "gt_match_time_sec": round(match_time, 3),
            "gt_match_basis": match_basis,
            "qwen_video_start_sec": round(evidence_start, 3),
            "ground_truth_clip": gt_clip,
            "ground_truth": truth,
            "qwen_prediction": prediction.strip().lower(),
            "correct": int(prediction.strip().lower() == truth),
            "qwen_inference_ms": inference_ms if isinstance(inference_ms, (int, float)) else "",
        })

    total = len(rows)
    class_total = {label: 0 for label in DRIVER_RESPONSE_LABELS}
    class_correct = {label: 0 for label in DRIVER_RESPONSE_LABELS}
    for row in rows:
        truth = str(row["ground_truth"])
        class_total[truth] += 1
        class_correct[truth] += int(row["correct"])
    overall_correct = sum(int(row["correct"]) for row in rows)
    accuracy = {
        label: (100.0 * class_correct[label] / class_total[label]
                if class_total[label] else None)
        for label in DRIVER_RESPONSE_LABELS
    }
    accuracy["overall"] = 100.0 * overall_correct / total
    latencies = [float(row["qwen_inference_ms"]) for row in rows
                 if isinstance(row["qwen_inference_ms"], (int, float))]

    plot_path = plot_path_override or args.event_accuracy_plot_file
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    matched_path = args.event_matched_file or plot_path.with_name(
        plot_path.stem + "_matched.csv"
    )
    matched_path.parent.mkdir(parents=True, exist_ok=True)
    with matched_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = (*DRIVER_RESPONSE_LABELS, "overall")
    display_names = ("Active", "Reduced", "No Visible\nResponse", "Overall")
    values = [accuracy[label] if accuracy[label] is not None else 0.0 for label in labels]
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(display_names, values, marker="o", linewidth=2.2, markersize=8)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Qwen accuracy (%)")
    axis.set_title(f"Event-only Qwen Driver Response Accuracy (n={total})")
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    for position, label in enumerate(labels):
        correct = overall_correct if label == "overall" else class_correct[label]
        count = total if label == "overall" else class_total[label]
        note = "N/A (0/0)" if accuracy[label] is None else f"{accuracy[label]:.1f}% ({correct}/{count})"
        axis.annotate(note, (position, values[position]), xytext=(0, 10),
                      textcoords="offset points", ha="center")
    mean_latency = sum(latencies) / len(latencies) if latencies else None
    latency_note = "Qwen model latency\nN/A" if mean_latency is None else (
        f"Qwen model latency\n{mean_latency:.1f} ms\n(n={len(latencies)})"
    )
    figure.text(0.98, 0.5, latency_note, ha="right", va="center", fontsize=11,
                bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f2f2f2", "edgecolor": "#999999"})
    figure.tight_layout(rect=(0, 0, 0.78, 1))
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)

    print(
        f"Qwen 이벤트 정확도: {accuracy['overall']:.1f}% ({overall_correct}/{total}) "
        f"| 평균 모델 추론: "
        + (f"{mean_latency:.1f} ms" if mean_latency is not None else "N/A"),
        flush=True,
    )
    for label in DRIVER_RESPONSE_LABELS:
        if accuracy[label] is not None:
            print(f"  {label}: {accuracy[label]:.1f}% ({class_correct[label]}/{class_total[label]})", flush=True)
        else:
            print(f"  {label}: N/A (0/0)", flush=True)
    incorrect_rows = [row for row in rows if not int(row["correct"])]
    print(f"틀린 이벤트 시간 스탬프 ({len(incorrect_rows)}건):", flush=True)
    if incorrect_rows:
        for row in incorrect_rows:
            print(
                f"  {float(row['gt_match_time_sec']):.3f}s | "
                f"정답={row['ground_truth']} | Qwen={row['qwen_prediction']}",
                flush=True,
            )
    else:
        print("  없음", flush=True)
    print(f"Qwen 이벤트 매칭 CSV 저장: {matched_path}", flush=True)
    print(f"Qwen 이벤트 정확도 그래프 저장: {plot_path}", flush=True)


def validate_args(args: argparse.Namespace) -> None:
    if args.bridge_timeout <= 0:
        raise ValueError("--bridge-timeout은 0보다 커야 합니다.")
    if args.poll_sec <= 0:
        raise ValueError("--poll-sec은 0보다 커야 합니다.")
    if args.monitor_period_sec <= 0:
        raise ValueError("--monitor-period-sec은 0보다 커야 합니다.")
    if args.max_telemetry_age_sec <= 0:
        raise ValueError("--max-telemetry-age-sec은 0보다 커야 합니다.")
    if args.limit < 0:
        raise ValueError("--limit은 0 이상이어야 합니다.")
    if args.start_index < 1:
        raise ValueError("--start-index는 1 이상이어야 합니다.")
    if args.result_display_sec < 0:
        raise ValueError("--result-display-sec은 0 이상이어야 합니다.")
    if args.start_observation_id < 0:
        raise ValueError("--start-observation-id는 0 이상이어야 합니다.")
    if args.ground_truth_file is not None and args.watch:
        raise ValueError("정확도 평가는 종료 시점이 있는 비-watch 모드에서만 가능합니다.")
    if args.show_accuracy_plot and args.ground_truth_file is None:
        raise ValueError("--show-accuracy-plot에는 --ground-truth-file이 필요합니다.")
    if args.evaluate_last_records < 0:
        raise ValueError("--evaluate-last-records는 0 이상이어야 합니다.")
    if args.evaluate_last_event_records < 0:
        raise ValueError("--evaluate-last-event-records는 0 이상이어야 합니다.")
    if args.evaluate_results_file is not None and args.evaluate_event_results_file is not None:
        raise ValueError("한 번에 하나의 저장 결과 평가만 지정하세요.")
    if args.evaluate_results_file is not None:
        if args.watch:
            raise ValueError("저장 결과 평가는 --watch와 함께 사용할 수 없습니다.")
        if args.ground_truth_file is None:
            raise ValueError(
                "--evaluate-results-file에는 --ground-truth-file이 필요합니다."
            )
        if not args.evaluate_results_file.is_file():
            raise FileNotFoundError(
                f"결과 JSONL 파일이 없습니다: {args.evaluate_results_file}"
            )
    if args.evaluate_event_results_file is not None:
        if args.watch:
            raise ValueError("이벤트 저장 결과 평가는 --watch와 함께 사용할 수 없습니다.")
        if args.ground_truth_file is None:
            raise ValueError(
                "--evaluate-event-results-file에는 --ground-truth-file이 필요합니다."
            )
        if not args.evaluate_event_results_file.is_file():
            raise FileNotFoundError(
                f"Qwen 이벤트 JSONL 파일이 없습니다: {args.evaluate_event_results_file}"
            )
    if args.ground_truth_file is not None:
        load_ground_truth(args.ground_truth_file)
    if not 0.0 < args.face_confidence <= 1.0:
        raise ValueError("--face-confidence는 0 초과 1 이하여야 합니다.")
    if args.face_padding < 0:
        raise ValueError("--face-padding은 0 이상이어야 합니다.")
    if args.qwen_face_padding < 0:
        raise ValueError("--qwen-face-padding은 0 이상이어야 합니다.")
    if not 0.0 <= args.vlm_fixed_driver_box_padding <= 0.5:
        raise ValueError("--vlm-fixed-driver-box-padding은 0.0~0.5여야 합니다.")
    if not 0.0 <= args.vlm_driver_seat_left_ratio < 1.0:
        raise ValueError("--vlm-driver-seat-left-ratio는 0.0 이상 1.0 미만이어야 합니다.")
    if args.face_crop_size < 32:
        raise ValueError("--face-crop-size는 32 이상이어야 합니다.")
    if not 0.0 < args.driver_roi_ratio <= 1.0:
        raise ValueError("--driver-roi-ratio는 0 초과 1 이하여야 합니다.")
    if not 0.0 < args.face_track_max_shift <= 1.0:
        raise ValueError("--face-track-max-shift는 0 초과 1 이하여야 합니다.")
    if args.face_track_max_missed_frames < 0:
        raise ValueError("--face-track-max-missed-frames는 0 이상이어야 합니다.")
    if args.face_track_reacquire_after < 1:
        raise ValueError("--face-track-reacquire-after는 1 이상이어야 합니다.")
    if not 0.0 < args.qwen_face_track_max_shift <= 1.0:
        raise ValueError("--qwen-face-track-max-shift는 0 초과 1 이하여야 합니다.")
    if args.qwen_face_track_max_missed_frames < 0:
        raise ValueError(
            "--qwen-face-track-max-missed-frames는 0 이상이어야 합니다."
        )
    if args.qwen_driver_guide_fallback_delay_sec < 0.0:
        raise ValueError(
            "--qwen-driver-guide-fallback-delay-sec는 0 이상이어야 합니다."
        )
    fallback_roi = args.fallback_driver_roi
    if not all(0.0 <= value <= 1.0 for value in fallback_roi) or not (
        fallback_roi[0] < fallback_roi[2]
        and fallback_roi[1] < fallback_roi[3]
    ):
        raise ValueError("--fallback-driver-roi는 0~1 범위의 X1 Y1 X2 Y2여야 합니다.")
    if args.yolo_face_crop and not args.yolo_face_model.is_file():
        raise FileNotFoundError(f"YOLO 얼굴 가중치가 없습니다: {args.yolo_face_model}")
    if args.yolo_driver_crop:
        if not DRIVER_SELECTOR_PATH.is_file():
            raise FileNotFoundError(
                f"TCN-VLM-YOLO 운전자 selector가 없습니다: {DRIVER_SELECTOR_PATH}"
            )
        if not args.driver_yolo_model.is_file():
            raise FileNotFoundError(f"운전자 YOLO 가중치가 없습니다: {args.driver_yolo_model}")
        if args.driver_box_hold_sec < 0:
            raise ValueError("--driver-box-hold-sec은 0 이상이어야 합니다.")
        if not 0.0 < args.driver_yolo_confidence <= 1.0:
            raise ValueError("--driver-yolo-confidence는 0 초과 1 이하여야 합니다.")
        if args.driver_yolo_interval < 1:
            raise ValueError("--driver-yolo-interval은 1 이상이어야 합니다.")
        if not 0.0 <= args.driver_min_x_ratio < 1.0:
            raise ValueError("--driver-min-x-ratio는 0 이상 1 미만이어야 합니다.")
    if args.yolo_eye_box and not args.yolo_eye_model.is_file():
        raise FileNotFoundError(f"YOLO Pose 눈 가중치가 없습니다: {args.yolo_eye_model}")
    if args.yolo_eye_interval < 1:
        raise ValueError("--yolo-eye-interval은 1 이상이어야 합니다.")
    if args.image_path is not None:
        if args.image_path.suffix.lower() not in MEDIA_SUFFIXES:
            raise ValueError(f"지원하지 않는 미디어 확장자입니다: {args.image_path}")
    elif (args.video_path is None and not args.watch
          and args.evaluate_results_file is None
          and args.evaluate_event_results_file is None):
        images = list_images(args.frames_dir)
        if not images:
            raise ValueError(f"이미지가 없습니다: {args.frames_dir}")
    if args.video_path is not None:
        if args.video_path.suffix.lower() != ".mp4":
            raise ValueError("--video-path에는 MP4 전체 영상을 지정하세요.")
        if args.image_path is not None or args.watch:
            raise ValueError("--video-path는 --image-path 또는 --watch와 함께 사용할 수 없습니다.")
        if (
            args.video_start_sec < 0
            or args.video_max_sec is not None and args.video_max_sec <= 0
        ):
            raise ValueError("--video-start-sec/--video-max-sec 값이 올바르지 않습니다.")
        if args.vlm_clip_seconds <= 0.0:
            raise ValueError("--vlm-clip-seconds는 0보다 커야 합니다.")
        if args.vlm_clip_max_width < 0:
            raise ValueError("--vlm-clip-max-width는 0 이상이어야 합니다.")

def run_once_or_directory(
    args: argparse.Namespace,
    headers: dict[str, str],
    preview: ImagePreview,
    evaluation_records: list[tuple[Path, dict[str, Any]]] | None = None,
) -> int:
    if args.image_path is not None:
        images = [args.image_path]
    else:
        images = list_images(args.frames_dir)
    images = images[args.start_index - 1:]
    if args.limit > 0:
        images = images[:args.limit]
    if not images:
        raise ValueError(
            f"--start-index {args.start_index}부터 처리할 이미지가 없습니다.")

    observation_id = args.start_observation_id
    for index, image_path in enumerate(images, 1):
        started = time.monotonic()
        print(f"[{index}/{len(images)}] POST /monitor {image_path}", flush=True)
        result = post_monitor_when_available(
            args, image_path, observation_id, headers, preview)
        print_result(result, image_path, args.print_json)
        save_result(result, image_path, args.results_file)
        if evaluation_records is not None:
            evaluation_records.append((image_path, result))
        preview.show(image_path, f"Result {index}/{len(images)}", result)
        preview.wait(args.result_display_sec)
        print(f"처리 완료 ({time.monotonic() - started:.1f}초)", flush=True)
        observation_id += 1
    return 0


def run_watch(
    args: argparse.Namespace,
    headers: dict[str, str],
    preview: ImagePreview,
) -> int:
    observation_id = args.start_observation_id
    processed = 0
    last_signature: tuple[str, int, int] | None = None
    last_submit = 0.0
    print("watch mode 시작: 새 이미지가 들어오면 /monitor로 보냅니다.", flush=True)
    while args.limit == 0 or processed < args.limit:
        image_path = newest_image(args)
        if image_path is None:
            preview.wait(args.poll_sec)
            continue

        try:
            signature = image_signature(image_path)
        except FileNotFoundError:
            preview.wait(args.poll_sec)
            continue
        now = time.monotonic()
        unchanged_and_disabled = (
            signature == last_signature and not args.resend_same_image
        )
        period_not_elapsed = now - last_submit < args.monitor_period_sec
        if unchanged_and_disabled or period_not_elapsed:
            preview.wait(args.poll_sec)
            continue

        started = time.monotonic()
        print(f"[{processed + 1}] POST /monitor {image_path}", flush=True)
        try:
            result = post_monitor_when_available(
                args, image_path, observation_id, headers, preview)
        except RuntimeError as exc:
            print(f"SKIP: {exc}", file=sys.stderr, flush=True)
            preview.wait(args.poll_sec)
            continue

        print_result(result, image_path, args.print_json)
        save_result(result, image_path, args.results_file)
        preview.show(image_path, f"Result {processed + 1}", result)
        preview.wait(args.result_display_sec)
        print(f"처리 완료 ({time.monotonic() - started:.1f}초)", flush=True)
        last_signature = signature
        last_submit = time.monotonic()
        observation_id += 1
        processed += 1
    return 0


def main() -> int:
    args = parse_args()
    preview = ImagePreview(args.show_image or (
        args.video_path is not None and args.video_preview
    ))
    try:
        validate_args(args)
        if args.evaluate_results_file is not None:
            evaluation_records = load_evaluation_records(
                args.evaluate_results_file, args.evaluate_last_records
            )
            evaluate_and_plot(evaluation_records, args)
            return 0
        if args.evaluate_event_results_file is not None:
            evaluate_qwen_event_results(args)
            return 0
        headers = bridge_headers(args)
        if args.video_path is not None:
            return run_full_video_realtime(args, headers, preview)
        if args.watch:
            return run_watch(args, headers, preview)
        evaluation_records: list[tuple[Path, dict[str, Any]]] = []
        exit_code = run_once_or_directory(
            args, headers, preview, evaluation_records
        )
        evaluate_and_plot(evaluation_records, args)
        return exit_code
    except KeyboardInterrupt:
        print("중단됨", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    finally:
        preview.close()


if __name__ == "__main__":
    raise SystemExit(main())
