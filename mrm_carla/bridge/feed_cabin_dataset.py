#!/usr/bin/env python3
"""Sequentially feed in-cabin dataset frames into latest.jpg and send to DGX monitor server.

Can be run on DGX SPARK or on local PC. Displays real-time preview via OpenCV.
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import shutil
import time
import urllib.error
import urllib.request
import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feed in-cabin dataset frames to DGX monitor and preview on screen")
    parser.add_argument(
        "--dataset-dir",
        default="/home/seame/Downloads/inner_mirror/extracted_frames/vp1",
        help="Directory containing frame images (.jpg, .png, .webp)")
    parser.add_argument(
        "--target-image-path",
        default="/home/seame/cabin/latest.jpg",
        help="Path where latest.jpg will be updated (atomic rename)")
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8000/monitor",
        help="DGX bridge server endpoint")
    parser.add_argument(
        "--token",
        default=os.environ.get("LLM_BRIDGE_TOKEN", "").strip(),
        help="Shared bearer token")
    parser.add_argument(
        "--period",
        type=float,
        default=4.0,
        help="Interval in seconds between frames")
    parser.add_argument(
        "--loop",
        action="store_true",
        default=True,
        help="Loop dataset infinitely")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV window display")
    return parser.parse_args()


def get_image_files(dataset_dir: str) -> list[str]:
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.PNG")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(dataset_dir, ext)))
    files = sorted(list(set(files)))
    return files


def send_monitor_request(server_url: str, token: str, image_bytes: bytes,
                         mime_type: str, frame_name: str) -> dict | None:
    payload = {
        "protocol_version": 1,
        "observation_id": int(time.time()),
        "captured_at_unix_s": time.time(),
        "telemetry": {
            "protocol_version": 1,
            "vehicle": {"speed_kph": 30.0},
        },
        "driving_summary": {
            "motion_state": "moving",
            "current_speed_kph": 30.0,
        },
        "cabin_image": {
            "mime_type": mime_type,
            "data_base64": base64.b64encode(image_bytes).decode("ascii"),
        },
    }

    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if token:
        headers["Authorization"] = "Bearer " + token

    request = urllib.request.Request(server_url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=10.0) as response:
            res_body = response.read()
            return json.loads(res_body.decode("utf-8"))
    except Exception as exc:
        print(f"[{frame_name}] Monitor request failed: {exc}", flush=True)
        return None


def main() -> None:
    args = parse_args()
    image_files = get_image_files(args.dataset_dir)
    if not image_files:
        print(f"Error: No image files found in {args.dataset_dir}")
        return

    print(f"Found {len(image_files)} dataset frames in {args.dataset_dir}")
    print(f"Feeding to: {args.target_image_path} every {args.period}s")
    print(f"Server URL: {args.server_url}")

    os.makedirs(os.path.dirname(os.path.abspath(args.target_image_path)), exist_ok=True)

    index = 0
    try:
        while True:
            frame_path = image_files[index]
            frame_name = os.path.basename(frame_path)

            # Atomic update target_image_path
            tmp_path = args.target_image_path + ".tmp"
            shutil.copy(frame_path, tmp_path)
            os.replace(tmp_path, args.target_image_path)

            # Read image for display & server
            with open(frame_path, "rb") as f:
                image_bytes = f.read()

            suffix = os.path.splitext(frame_path)[1].lower()
            mime = "image/png" if suffix == ".png" else "image/jpeg"

            print(f"\n[{index+1}/{len(image_files)}] Displaying & sending frame: {frame_name}")

            # Send to model server
            res = send_monitor_request(args.server_url, args.token, image_bytes, mime, frame_name)

            # Display on local screen using OpenCV
            if not args.no_display:
                img = cv2.imread(frame_path)
                if img is not None:
                    overlay_text = f"Frame: {frame_name}"
                    if res and "driver_state" in res:
                        st = res["driver_state"]
                        risk = st.get("risk_level", "unknown").upper()
                        reason = st.get("reason", "")
                        overlay_text += f" | Risk: {risk} ({reason})"

                        # Color coding based on risk
                        color = (0, 255, 0) if risk == "NORMAL" else ((0, 165, 255) if risk == "CAUTION" else (0, 0, 255))
                    else:
                        color = (255, 255, 255)

                    cv2.putText(img, overlay_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                    cv2.putText(img, overlay_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.imshow("In-Cabin Dataset Real-Time Display", img)
                    cv2.waitKey(1)

            index = (index + 1) % len(image_files)
            if index == 0 and not args.loop:
                break

            time.sleep(args.period)
    except KeyboardInterrupt:
        print("\nStopped dataset feeding.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
