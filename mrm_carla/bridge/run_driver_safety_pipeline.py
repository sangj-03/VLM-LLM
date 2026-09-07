#!/usr/bin/env python3
"""DGX-side cabin image monitor that publishes observations to the bridge.

This script does not call vLLM directly.  It sends each cabin image together
with the latest CARLA telemetry to the bridge POST /monitor endpoint.  The
bridge owns model inference and updates GET /driver_state/latest for the CARLA
safety supervisor.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import signal
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = 1
DEFAULT_FRAMES = Path("/home/seame/Downloads/Day-RGB1/RGB1/S1/AC10")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


class RequestError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


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
        help="directory of image frames when --image-path is not set",
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
        default=1.0,
        help="minimum interval between successful monitor submissions",
    )
    parser.add_argument(
        "--max-telemetry-age-sec",
        type=float,
        default=2.0,
        help="skip inference when cached telemetry is older than this",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="maximum images to submit; 0 means unlimited",
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
        "--print-json",
        action="store_true",
        help="print full /monitor response JSON instead of one-line status",
    )
    parser.add_argument(
        "--no-notify-supervisor-on-complete",
        action="store_true",
        help=(
            "do not POST /monitor/complete after a finite dataset/limited "
            "watch run finishes successfully"
        ),
    )
    parser.add_argument(
        "--show-image",
        action="store_true",
        help="display the image currently being processed in an OpenCV window",
    )

    # Deprecated arguments accepted so old shell commands fail less abruptly.
    parser.add_argument("--role-prompt", help=argparse.SUPPRESS)
    parser.add_argument("--safety-prompt", help=argparse.SUPPRESS)
    parser.add_argument("--base-url", help=argparse.SUPPRESS)
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


def cabin_image_payload(path: Path) -> dict[str, str]:
    suffix = path.suffix.lower()
    if suffix not in MIME_TYPES:
        raise ValueError(f"지원하지 않는 이미지 확장자입니다: {path}")
    return {
        "mime_type": MIME_TYPES[suffix],
        "data_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
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
    if not args.no_inline_telemetry:
        telemetry, telemetry_age_s = latest_telemetry(args, headers)

    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "observation_id": observation_id,
        "captured_at_unix_s": time.time(),
        "cabin_image": cabin_image_payload(image_path),
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


def notify_monitoring_complete(
    args: argparse.Namespace,
    observation_id: int,
    headers: dict[str, str],
) -> None:
    """Tell the CARLA supervisor that a finite dataset has been exhausted."""
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "last_observation_id": observation_id,
        "completed_at_unix_s": time.time(),
    }
    url = args.bridge_url.rstrip("/") + "/monitor/complete"
    # The bridge stores the final inference before answering /monitor, but a
    # finite-run worker can still race the store/HTTP response boundary. Retry
    # a short 409 window so completion reliably reaches the CARLA supervisor.
    result = None
    for attempt in range(4):
        try:
            result = request_json(
                "POST", url, headers, payload, args.bridge_timeout)
            break
        except RequestError as exc:
            if exc.status_code != 409 or attempt == 3:
                raise
            time.sleep(0.25 * (attempt + 1))
    if result is None:
        raise RuntimeError("bridge 완료 응답이 없습니다.")
    if result.get("monitoring_complete") is not True:
        raise RuntimeError("bridge가 dataset 완료 신호를 확인하지 않았습니다.")
    print(
        "DATASET COMPLETE: supervisor 종료 신호 전송 완료 "
        f"(last_observation_id={observation_id})",
        flush=True,
    )


def list_images(frames_dir: Path) -> list[Path]:
    if not frames_dir.exists():
        raise FileNotFoundError(f"이미지 디렉터리가 없습니다: {frames_dir}")
    images = [
        path for path in frames_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
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
    """Optional OpenCV preview of the image currently sent to the bridge."""

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
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WINDOW_NAME, 960, 720)
            cv2.startWindowThread()
        except Exception as exc:
            self.enabled = False
            self.cv2 = None
            print(
                f"이미지 미리보기 비활성화: OpenCV 창을 열 수 없습니다: {exc}",
                file=sys.stderr,
                flush=True,
            )

    def show(self, image_path: Path, label: str) -> None:
        if not self.enabled or self.cv2 is None:
            return
        image = self.cv2.imread(str(image_path), self.cv2.IMREAD_COLOR)
        if image is None:
            print(
                f"이미지 미리보기 실패: {image_path}",
                file=sys.stderr,
                flush=True,
            )
            return
        height, width = image.shape[:2]
        scale = min(960.0 / width, 720.0 / height, 1.0)
        if scale < 1.0:
            image = self.cv2.resize(
                image,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=self.cv2.INTER_AREA,
            )
        caption = f"{label}  {image_path.name}"
        self.cv2.rectangle(image, (0, 0), (image.shape[1], 42), (0, 0, 0), -1)
        self.cv2.putText(
            image,
            caption,
            (12, 29),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            self.cv2.LINE_AA,
        )
        self.cv2.imshow(self.WINDOW_NAME, image)
        self.cv2.waitKey(1)

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
    supervisor = result.get("safety_supervisor", {})
    if not isinstance(supervisor, dict):
        supervisor = {}
    overall = supervisor.get("overall_risk", {})
    if not isinstance(overall, dict):
        overall = {}
    intervention = supervisor.get("selected_intervention", {})
    if not isinstance(intervention, dict):
        intervention = {}
    print(
        "DRIVER MONITOR result "
        f"image={image_path.name} "
        f"observation={result.get('observation_id')} "
        f"risk={state.get('risk_level')} "
        f"forward={state.get('looking_forward')} "
        f"unsafe={state.get('unsafe_behavior')} "
        f"confidence={state.get('confidence')} "
        f"overall_risk={overall.get('risk_level')} "
        f"intervention={intervention.get('intervention_type')} "
        f"inference_ms={result.get('inference_ms')} "
        f"reason={state.get('reason', '')}",
        flush=True,
    )


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
    if args.start_observation_id < 0:
        raise ValueError("--start-observation-id는 0 이상이어야 합니다.")
    if args.image_path is not None:
        if args.image_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"지원하지 않는 이미지 확장자입니다: {args.image_path}")
    elif not args.watch:
        images = list_images(args.frames_dir)
        if not images:
            raise ValueError(f"이미지가 없습니다: {args.frames_dir}")


def run_once_or_directory(
    args: argparse.Namespace,
    headers: dict[str, str],
    preview: ImagePreview,
) -> int:
    if args.image_path is not None:
        images = [args.image_path]
    else:
        images = list_images(args.frames_dir)
    if args.limit > 0:
        images = images[:args.limit]

    observation_id = args.start_observation_id
    last_success = 0.0
    for index, image_path in enumerate(images, 1):
        if last_success > 0.0:
            remaining = args.monitor_period_sec - (time.monotonic() - last_success)
            if remaining > 0.0:
                print(f"다음 모니터링까지 {remaining:.1f}초 대기", flush=True)
                preview.wait(remaining)
        started = time.monotonic()
        print(f"[{index}/{len(images)}] POST /monitor {image_path}", flush=True)
        preview.show(image_path, f"Processing {index}/{len(images)}")
        result = post_monitor_when_available(
            args, image_path, observation_id, headers, preview)
        last_success = time.monotonic()
        args.last_successful_observation_id = observation_id
        print_result(result, image_path, args.print_json)
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
        preview.show(image_path, f"Processing {processed + 1}")
        try:
            result = post_monitor_when_available(
                args, image_path, observation_id, headers, preview)
        except RuntimeError as exc:
            print(f"SKIP: {exc}", file=sys.stderr, flush=True)
            preview.wait(args.poll_sec)
            continue

        print_result(result, image_path, args.print_json)
        print(f"처리 완료 ({time.monotonic() - started:.1f}초)", flush=True)
        last_signature = signature
        last_submit = time.monotonic()
        args.last_successful_observation_id = observation_id
        observation_id += 1
        processed += 1
    return 0


def main() -> int:
    args = parse_args()
    args.last_successful_observation_id = None
    preview = ImagePreview(args.show_image)
    headers = None
    completion_sent = False
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def stop_on_sigterm(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_sigterm)
    try:
        validate_args(args)
        headers = bridge_headers(args)
        if args.watch:
            result = run_watch(args, headers, preview)
        else:
            result = run_once_or_directory(args, headers, preview)
        finite_run = not args.watch or args.limit > 0
        if (result == 0 and finite_run
                and not args.no_notify_supervisor_on_complete
                and args.last_successful_observation_id is not None):
            notify_monitoring_complete(
                args, args.last_successful_observation_id, headers)
            completion_sent = True
        return result
    except KeyboardInterrupt:
        print("중단됨", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    finally:
        if (headers is not None
                and not completion_sent
                and not args.no_notify_supervisor_on_complete
                and args.last_successful_observation_id is not None):
            try:
                notify_monitoring_complete(
                    args, args.last_successful_observation_id, headers)
            except (OSError, RuntimeError, RequestError) as exc:
                print(
                    f"SUPERVISOR COMPLETE NOTIFY FAILED: {exc}",
                    file=sys.stderr, flush=True)
        preview.close()
        signal.signal(signal.SIGTERM, previous_sigterm_handler)


if __name__ == "__main__":
    raise SystemExit(main())
