#!/usr/bin/env python3
"""
Real-time-style streaming driver-state runtime.

Pipeline:
MP4 / camera-like stream
 -> YOLO + ByteTrack driver selection on every source frame
 -> MediaPipe Holistic at 10 Hz
 -> incremental 22-D feature vector
 -> rolling [100, 22] buffer
 -> 5-seed Binary TCN ensemble every 0.2 s
 -> threshold 0.48 candidate detection
 -> asynchronous Qwen3-VL-2B atomic + posture verification
 -> provisional Safety Supervisor
 -> terminal pop-out:
      ACTIVE / REDUCED / NO VISIBLE RESPONSE

This is a streaming prototype for the current development pipeline.
It does NOT use ground-truth labels at runtime.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Some existing scripts use project-relative paths.
os.chdir(ROOT)

from common import FEATURE_NAMES, clip_finite, load_config
from driver_selector import DriverSelector, expand_driver_box


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Reuse the exact feature math and exact TCN architecture already used
# by this project instead of duplicating them.
feature_mod = load_module(
    "driver_feature_module",
    SRC / "03_extract_features.py",
)
binary_mod = load_module(
    "binary_tcn_module_realtime",
    SRC / "17_train_binary_tcn.py",
)

BinaryDriverTCN = binary_mod.BinaryDriverTCN


# ============================================================
# TCN ensemble
# ============================================================

class TCNEnsemble:
    def __init__(self, config_path: Path, device: torch.device):
        self.device = device

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.threshold = float(config["selected_threshold"])
        self.members = []

        model_paths = [Path(x) for x in config["model_paths"]]

        for p in model_paths:
            if not p.is_absolute():
                p = ROOT / p

            ckpt = torch.load(
                p,
                map_location="cpu",
                weights_only=False,
            )

            feature_names = [
                str(x)
                for x in ckpt["feature_names"]
            ]

            if feature_names != list(FEATURE_NAMES):
                raise RuntimeError(
                    f"Feature order mismatch in {p}\n"
                    f"checkpoint={feature_names}\n"
                    f"runtime={list(FEATURE_NAMES)}"
                )

            model = BinaryDriverTCN(
                feature_dim=int(ckpt["feature_dim"]),
                hidden_dim=int(ckpt["hidden_dim"]),
                dropout=float(ckpt["dropout"]),
            ).to(device)

            model.load_state_dict(
                ckpt["model_state_dict"]
            )
            model.eval()

            mean = torch.as_tensor(
                np.asarray(
                    ckpt["normalization_mean"],
                    dtype=np.float32,
                ).reshape(1, 1, -1),
                device=device,
            )

            std = torch.as_tensor(
                np.asarray(
                    ckpt["normalization_std"],
                    dtype=np.float32,
                ).reshape(1, 1, -1),
                device=device,
            )

            seed = int(ckpt.get("seed", len(self.members)))

            self.members.append(
                {
                    "path": str(p),
                    "seed": seed,
                    "model": model,
                    "mean": mean,
                    "std": std,
                }
            )

        if not self.members:
            raise RuntimeError("No TCN ensemble members were loaded.")

    @torch.inference_mode()
    def predict(self, window: np.ndarray):
        """
        window: [100, 22]
        returns:
            ensemble_probability, per_seed_probabilities
        """
        if window.shape != (100, len(FEATURE_NAMES)):
            raise ValueError(
                f"Expected window (100,{len(FEATURE_NAMES)}), "
                f"got {window.shape}"
            )

        x = torch.from_numpy(
            window.astype(np.float32, copy=False)
        ).unsqueeze(0).to(self.device)

        probs = {}

        for member in self.members:
            xn = (
                x - member["mean"]
            ) / member["std"]

            logit = member["model"](xn)

            p = float(
                torch.sigmoid(logit)
                .detach()
                .cpu()
                .item()
            )

            probs[
                f"seed{member['seed']}"
            ] = p

        ensemble = float(
            np.mean(
                list(probs.values())
            )
        )

        return ensemble, probs


# ============================================================
# Incremental 22-D feature extractor
# ============================================================

class IncrementalFeatureExtractor:
    """
    Mirrors the feature calculations in src/03_extract_features.py,
    but returns one 22-D vector immediately instead of waiting for
    the full video.
    """

    def __init__(self, target_fps: float):
        self.target_fps = float(target_fps)
        self.dt = 1.0 / self.target_fps

        self.prev_face = None
        self.prev_left = None
        self.prev_right = None
        self.prev_angles = None
        self.prev_neck_rel = None

        self.landmarker = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def close(self):
        self.landmarker.close()

    def extract(self, crop: np.ndarray):
        tic = time.perf_counter()

        rgb = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2RGB,
        )

        result = self.landmarker.process(rgb)

        mp_latency = time.perf_counter() - tic

        face = (
            result.face_landmarks.landmark
            if result.face_landmarks
            else None
        )

        pose = (
            result.pose_landmarks.landmark
            if result.pose_landmarks
            else None
        )

        left = (
            result.left_hand_landmarks.landmark
            if result.left_hand_landmarks
            else None
        )

        right = (
            result.right_hand_landmarks.landmark
            if result.right_hand_landmarks
            else None
        )

        face_valid = int(
            face is not None
            and len(face) > 291
        )

        eye_valid = int(
            face_valid
            and max(
                feature_mod.LEFT_EYE
                + feature_mod.RIGHT_EYE
            ) < len(face)
        )

        face_rel = (
            feature_mod.in_frame_ratio(face)
            if face_valid
            else 0.0
        )

        pose_rel = (
            feature_mod.pose_reliability(pose)
        )

        lrel = feature_mod.hand_rel(left)
        rrel = feature_mod.hand_rel(right)

        hand_reliability = (
            lrel + rrel
        ) / 2.0

        if eye_valid:
            ear_l = feature_mod.ear(
                face,
                feature_mod.LEFT_EYE,
            )

            ear_r = feature_mod.ear(
                face,
                feature_mod.RIGHT_EYE,
            )

            ear_m = np.nanmean(
                [ear_l, ear_r]
            )
        else:
            ear_l = 0.0
            ear_r = 0.0
            ear_m = 0.0

        angles = (
            feature_mod.head_pose(
                face,
                crop.shape[1],
                crop.shape[0],
            )
            if face_valid
            else None
        )

        if angles is None:
            pitch = yaw = roll = 0.0
            pvel = yvel = rvel = 0.0
            self.prev_angles = None
        else:
            pitch, yaw, roll = angles

            if self.prev_angles is None:
                pvel = yvel = rvel = 0.0
            else:
                pvel = (
                    feature_mod.angle_delta(
                        pitch,
                        self.prev_angles[0],
                    )
                    / self.dt
                )

                yvel = (
                    feature_mod.angle_delta(
                        yaw,
                        self.prev_angles[1],
                    )
                    / self.dt
                )

                rvel = (
                    feature_mod.angle_delta(
                        roll,
                        self.prev_angles[2],
                    )
                    / self.dt
                )

            self.prev_angles = angles

        pvel = clip_finite(
            pvel,
            -720,
            720,
        )

        yvel = clip_finite(
            yvel,
            -720,
            720,
        )

        rvel = clip_finite(
            rvel,
            -720,
            720,
        )

        head_motion = (
            feature_mod.mean_motion(
                face,
                self.prev_face,
                ids=[1, 33, 263, 61, 291],
                dt=self.dt,
            )
            if face_valid
            and self.prev_face
            else 0.0
        )

        nv = (
            feature_mod.neck_values(pose)
            if pose_rel > 0
            else None
        )

        if nv is None:
            neck_flexion_proxy = 0.0
            neck_lateral_proxy = 0.0
            neck_velocity = 0.0
            self.prev_neck_rel = None
        else:
            (
                neck_flexion_proxy,
                neck_lateral_proxy,
                neck_rel,
            ) = nv

            if self.prev_neck_rel is None:
                neck_velocity = 0.0
            else:
                neck_velocity = float(
                    np.linalg.norm(
                        neck_rel
                        - self.prev_neck_rel
                    )
                    / self.dt
                )

            self.prev_neck_rel = neck_rel

        left_valid = int(
            left is not None
            and len(left) >= 21
        )

        right_valid = int(
            right is not None
            and len(right) >= 21
        )

        left_motion = (
            feature_mod.mean_motion(
                left,
                self.prev_left,
                dt=self.dt,
            )
            if left_valid
            and self.prev_left
            else 0.0
        )

        right_motion = (
            feature_mod.mean_motion(
                right,
                self.prev_right,
                dt=self.dt,
            )
            if right_valid
            and self.prev_right
            else 0.0
        )

        values = [
            ear_l,
            ear_r,
            ear_m,
            eye_valid,
            pitch,
            yaw,
            roll,
            pvel,
            yvel,
            rvel,
            head_motion,
            neck_flexion_proxy,
            neck_lateral_proxy,
            neck_velocity,
            left_motion,
            right_motion,
            left_valid,
            right_valid,
            face_valid,
            face_rel,
            pose_rel,
            hand_reliability,
        ]

        vector = np.asarray(
            [
                clip_finite(v)
                for v in values
            ],
            dtype=np.float32,
        )

        if vector.shape != (
            len(FEATURE_NAMES),
        ):
            raise RuntimeError(
                f"Expected {len(FEATURE_NAMES)} features, "
                f"got {vector.shape}"
            )

        self.prev_face = (
            face
            if face_valid
            else None
        )

        self.prev_left = (
            left
            if left_valid
            else None
        )

        self.prev_right = (
            right
            if right_valid
            else None
        )

        return vector, {
            "mediapipe_sec":
                mp_latency,
            "face_valid":
                face_valid,
            "pose_reliability":
                float(pose_rel),
            "hand_reliability":
                float(hand_reliability),
        }


# ============================================================
# Temporal frame snapshot for Qwen
# ============================================================

def select_frames(
    frame_buffer,
    start_time: float,
    end_time: float,
    fps: float,
):
    available = [
        (float(t), img)
        for t, img in frame_buffer
        if start_time - 1e-6
        <= float(t)
        <= end_time + 1e-6
    ]

    if not available:
        return []

    target_step = 1.0 / float(fps)

    targets = []
    t = float(start_time)

    while t <= end_time + 1e-6:
        targets.append(t)
        t += target_step

    if not targets:
        targets = [end_time]

    selected = []

    for target in targets:
        _, image = min(
            available,
            key=lambda item:
                abs(item[0] - target),
        )

        selected.append(
            image.copy()
        )

    return selected


def save_sequence(
    images,
    directory: Path,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not images:
        raise RuntimeError(
            f"No images for Qwen snapshot: {directory}"
        )

    images = [
        x.copy()
        for x in images
    ]

    # Keep the same practical constraint used by the earlier
    # Qwen frame extraction script.
    if len(images) == 1:
        images.append(
            images[-1].copy()
        )
    elif len(images) % 2 == 1:
        images.append(
            images[-1].copy()
        )

    paths = []

    for i, image in enumerate(
        images,
        start=1,
    ):
        path = (
            directory
            / f"frame_{i:04d}.jpg"
        )

        ok = cv2.imwrite(
            str(path),
            image,
            [
                int(
                    cv2.IMWRITE_JPEG_QUALITY
                ),
                92,
            ],
        )

        if not ok:
            raise RuntimeError(
                f"Failed to write {path}"
            )

        paths.append(path)

    return paths


# ============================================================
# Terminal output
# ============================================================

def display_yes_no(value):
    value = str(value).strip().lower()

    if value == "yes":
        return "YES"

    if value == "no":
        return "NO"

    if value == "not_applicable":
        return "N/A"

    return "UNKNOWN"


def response_display(value):
    return {
        "active":
            "ACTIVE",
        "reduced":
            "REDUCED",
        "no_visible_response":
            "NO VISIBLE RESPONSE",
    }.get(
        str(value).strip().lower(),
        str(value).upper(),
    )


def print_result(
    driver_response,
    tcn_probability,
    threshold,
    candidate,
    purposeful_action,
    self_response,
    external_contact,
    response_after_contact,
    visibility,
    inference_sec,
):
    print()
    print(
        "Driver Response:",
        response_display(
            driver_response
        ),
        flush=True,
    )

    print()

    print(
        "TCN Probability:",
        f"{float(tcn_probability):.3f}",
        flush=True,
    )

    print(
        "TCN Threshold:",
        f"{float(threshold):.3f}",
        flush=True,
    )

    print(
        "Candidate:",
        (
            "DETECTED"
            if candidate
            else "CLEAR"
        ),
        flush=True,
    )

    print()

    if candidate:
        print(
            "Purposeful Action:",
            display_yes_no(
                purposeful_action
            ),
            flush=True,
        )

        print(
            "Self Response:",
            display_yes_no(
                self_response
            ),
            flush=True,
        )

        print(
            "External Contact:",
            display_yes_no(
                external_contact
            ),
            flush=True,
        )

        print(
            "Response to Contact:",
            display_yes_no(
                response_after_contact
            ),
            flush=True,
        )

        print(
            "Visibility:",
            (
                "GOOD"
                if str(
                    visibility
                ).lower() == "good"
                else "POOR"
            ),
            flush=True,
        )
    else:
        print(
            "Purposeful Action: N/A",
            flush=True,
        )
        print(
            "Self Response: N/A",
            flush=True,
        )
        print(
            "External Contact: N/A",
            flush=True,
        )
        print(
            "Response to Contact: N/A",
            flush=True,
        )
        print(
            "Visibility: N/A",
            flush=True,
        )

    print()

    print(
        "Inference Time:",
        f"{float(inference_sec):.2f} s",
        flush=True,
    )



# ============================================================
# Live dashboard
# ============================================================

def _panel_text_value(value):
    if value is None:
        return "N/A"
    value = str(value).strip()
    if not value:
        return "N/A"
    return value


def render_dashboard(
    frame,
    driver_box,
    video_time,
    threshold,
    probability,
    candidate,
    feature_buffer_size,
    qwen_status,
    latest_status,
):
    """
    One OpenCV window:
      LEFT  : current source frame + driver bbox
      RIGHT : current analysis state
    """

    canvas_h = 720
    panel_w = 520
    video_w = 1080

    # --------------------------------------------------------
    # Left side: current source frame
    # --------------------------------------------------------
    h, w = frame.shape[:2]
    scale = min(
        video_w / max(w, 1),
        canvas_h / max(h, 1),
    )

    resized = cv2.resize(
        frame,
        (
            max(1, int(round(w * scale))),
            max(1, int(round(h * scale))),
        ),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros(
        (
            canvas_h,
            video_w + panel_w,
            3,
        ),
        dtype=np.uint8,
    )

    yoff = (
        canvas_h - resized.shape[0]
    ) // 2

    xoff = (
        video_w - resized.shape[1]
    ) // 2

    canvas[
        yoff:yoff + resized.shape[0],
        xoff:xoff + resized.shape[1],
    ] = resized

    # Driver bbox in the same resized coordinate system.
    if driver_box is not None:
        x1, y1, x2, y2 = driver_box

        rx1 = int(
            round(x1 * scale)
        ) + xoff
        ry1 = int(
            round(y1 * scale)
        ) + yoff
        rx2 = int(
            round(x2 * scale)
        ) + xoff
        ry2 = int(
            round(y2 * scale)
        ) + yoff

        cv2.rectangle(
            canvas,
            (rx1, ry1),
            (rx2, ry2),
            (0, 255, 255),
            3,
        )

        cv2.putText(
            canvas,
            "DRIVER",
            (
                rx1,
                max(
                    25,
                    ry1 - 8,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # --------------------------------------------------------
    # Right side: analysis panel
    # --------------------------------------------------------
    px = video_w + 25

    cv2.putText(
        canvas,
        "REAL-TIME DRIVER ANALYSIS",
        (px, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    status = latest_status or {}

    driver_response = (
        status.get(
            "driver_response",
            "warming_up"
            if feature_buffer_size < 100
            else (
                "verifying"
                if qwen_status == "RUNNING"
                else "active"
            ),
        )
    )

    if driver_response == "warming_up":
        display_response = (
            f"WARMING UP "
            f"({feature_buffer_size}/100)"
        )
    elif driver_response == "verifying":
        display_response = "VERIFYING..."
    else:
        display_response = (
            response_display(
                driver_response
            )
        )

    purposeful = status.get(
        "purposeful_action",
        "not_applicable",
    )
    self_response = status.get(
        "self_response",
        "not_applicable",
    )
    external_contact = status.get(
        "external_contact",
        "not_applicable",
    )
    response_contact = status.get(
        "response_after_contact",
        "not_applicable",
    )
    visibility = status.get(
        "visibility",
        "not_applicable",
    )

    if probability is None:
        prob_text = "N/A"
    else:
        prob_text = (
            f"{float(probability):.3f}"
        )

    candidate_text = (
        "DETECTED"
        if candidate
        else "CLEAR"
    )

    if qwen_status == "RUNNING":
        candidate_text = (
            "DETECTED / VERIFYING"
        )

    lines = [
        (
            "Video Time",
            f"{float(video_time):.1f} s",
        ),
        (
            "Driver Response",
            display_response,
        ),
        (
            "TCN Probability",
            prob_text,
        ),
        (
            "TCN Threshold",
            f"{float(threshold):.3f}",
        ),
        (
            "Candidate",
            candidate_text,
        ),
        (
            "Qwen",
            qwen_status,
        ),
        (
            "Purposeful Action",
            display_yes_no(
                purposeful
            ),
        ),
        (
            "Self Response",
            display_yes_no(
                self_response
            ),
        ),
        (
            "External Contact",
            display_yes_no(
                external_contact
            ),
        ),
        (
            "Response to Contact",
            display_yes_no(
                response_contact
            ),
        ),
        (
            "Visibility",
            (
                "GOOD"
                if str(
                    visibility
                ).lower() == "good"
                else (
                    "POOR"
                    if str(
                        visibility
                    ).lower() == "poor"
                    else "N/A"
                )
            ),
        ),
        (
            "Inference Time",
            (
                f"{float(status.get('inference_time_sec', 0.0)):.2f} s"
                if "inference_time_sec" in status
                else "N/A"
            ),
        ),
    ]

    y = 95

    for label, value in lines:
        cv2.putText(
            canvas,
            f"{label}:",
            (px, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        y += 27

        cv2.putText(
            canvas,
            str(value),
            (px + 15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        y += 36

    cv2.putText(
        canvas,
        "Press q to quit",
        (
            px,
            canvas_h - 20,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )

    return canvas


# ============================================================
# Qwen worker
# ============================================================

class QwenVerifier:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int,
    ):
        from transformers import (
            AutoProcessor,
            Qwen3VLForConditionalGeneration,
        )

        self.atomic = load_module(
            "qwen_atomic_realtime",
            SRC / "32_qwen2b_atomic_verifier.py",
        )

        self.support = load_module(
            "qwen_support_realtime",
            SRC / "33_qwen2b_support_verifier.py",
        )

        self.supervisor = load_module(
            "safety_supervisor_realtime",
            SRC / "34_final_safety_supervisor.py",
        )

        print()
        print(
            "Loading Qwen3-VL-2B...",
            flush=True,
        )

        self.model = (
            Qwen3VLForConditionalGeneration
            .from_pretrained(
                model_name,
                dtype="auto",
                device_map="auto",
            )
        )

        self.processor = (
            AutoProcessor
            .from_pretrained(
                model_name
            )
        )

        self.max_new_tokens = int(
            max_new_tokens
        )

        print(
            "Qwen device:",
            next(
                self.model.parameters()
            ).device,
            flush=True,
        )

        print(
            "Qwen dtype:",
            next(
                self.model.parameters()
            ).dtype,
            flush=True,
        )

    def ask(
        self,
        context_uris,
        candidate_uris,
        prompt,
        allowed,
    ):
        return (
            self.atomic
            .run_atomic_question(
                self.model,
                self.processor,
                context_uris,
                candidate_uris,
                prompt,
                allowed,
                self.max_new_tokens,
            )
        )

    def verify(
        self,
        job,
        threshold: float,
    ):
        wall_start = time.perf_counter()

        context_uris = [
            p.resolve().as_uri()
            for p in job[
                "context_paths"
            ]
        ]

        candidate_uris = [
            p.resolve().as_uri()
            for p in job[
                "candidate_paths"
            ]
        ]

        visibility = self.ask(
            context_uris,
            candidate_uris,
            self.atomic.PROMPTS[
                "visibility"
            ],
            {
                "good",
                "poor",
            },
        )

        external = self.ask(
            context_uris,
            candidate_uris,
            self.atomic.PROMPTS[
                "external_contact"
            ],
            {
                "yes",
                "no",
                "uncertain",
            },
        )

        purposeful = self.ask(
            context_uris,
            candidate_uris,
            self.atomic.PROMPTS[
                "purposeful_action"
            ],
            {
                "yes",
                "no",
                "uncertain",
            },
        )

        self_response = self.ask(
            context_uris,
            candidate_uris,
            self.atomic.PROMPTS[
                "self_response"
            ],
            {
                "yes",
                "no",
                "uncertain",
            },
        )

        if external["answer"] == "yes":
            response_contact = self.ask(
                context_uris,
                candidate_uris,
                self.atomic.PROMPTS[
                    "response_after_contact"
                ],
                {
                    "yes",
                    "no",
                    "uncertain",
                },
            )
        elif external["answer"] == "no":
            response_contact = {
                "answer":
                    "not_applicable",
                "latency_sec":
                    0.0,
            }
        else:
            response_contact = {
                "answer":
                    "uncertain",
                "latency_sec":
                    0.0,
            }

        support_loss = self.ask(
            context_uris,
            candidate_uris,
            self.support.PROMPT_SUPPORT_LOSS,
            {
                "yes",
                "no",
                "uncertain",
            },
        )

        recovery = self.ask(
            context_uris,
            candidate_uris,
            self.support.PROMPT_RECOVERY,
            {
                "yes",
                "no",
                "not_applicable",
                "uncertain",
            },
        )

        final_support = self.ask(
            context_uris,
            candidate_uris,
            self.support.PROMPT_FINAL_SUPPORT,
            {
                "yes",
                "no",
                "uncertain",
            },
        )

        row = pd.Series(
            {
                "tcn_peak_probability":
                    float(
                        job[
                            "tcn_probability"
                        ]
                    ),
                "purposeful_action":
                    purposeful[
                        "answer"
                    ],
                "self_response":
                    self_response[
                        "answer"
                    ],
                "external_contact":
                    external[
                        "answer"
                    ],
                "response_after_contact":
                    response_contact[
                        "answer"
                    ],
                "visibility":
                    visibility[
                        "answer"
                    ],
                "loss_of_support":
                    support_loss[
                        "answer"
                    ],
                "deliberate_recovery":
                    recovery[
                        "answer"
                    ],
                "supported_at_end":
                    final_support[
                        "answer"
                    ],
            }
        )

        (
            response,
            reason,
            normalized_recovery,
        ) = (
            self.supervisor
            .safety_supervisor(
                row,
                threshold,
            )
        )

        qwen_generated_latency = sum(
            float(x.get("latency_sec", 0.0))
            for x in [
                visibility,
                external,
                purposeful,
                self_response,
                response_contact,
                support_loss,
                recovery,
                final_support,
            ]
        )

        wall_sec = (
            time.perf_counter()
            - wall_start
        )

        return {
            "event_id":
                int(job["event_id"]),
            "video_time":
                float(job["video_time"]),
            "driver_response":
                response,
            "decision_reason":
                reason,
            "tcn_probability":
                float(
                    job[
                        "tcn_probability"
                    ]
                ),
            "purposeful_action":
                purposeful[
                    "answer"
                ],
            "self_response":
                self_response[
                    "answer"
                ],
            "external_contact":
                external[
                    "answer"
                ],
            "response_after_contact":
                response_contact[
                    "answer"
                ],
            "visibility":
                visibility[
                    "answer"
                ],
            "loss_of_support":
                support_loss[
                    "answer"
                ],
            "deliberate_recovery":
                normalized_recovery,
            "supported_at_end":
                final_support[
                    "answer"
                ],
            "qwen_generated_latency_sec":
                qwen_generated_latency,
            "verification_wall_sec":
                wall_sec,
            "tcn_stage_sec":
                float(
                    job[
                        "tcn_stage_sec"
                    ]
                ),
        }


# ============================================================
# Status logging
# ============================================================

def save_latest(
    path: Path,
    result: dict,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Real-time-style MP4 -> 22-D -> "
            "5-seed TCN -> Qwen3-VL-2B -> "
            "Safety Supervisor"
        )
    )

    parser.add_argument(
        "--video",
        required=True,
    )

    parser.add_argument(
        "--config",
        default="config.json",
    )

    parser.add_argument(
        "--ensemble-config",
        default=(
            "models/"
            "binary_tcn_ensemble_v1.json"
        ),
    )

    parser.add_argument(
        "--qwen-model",
        default=(
            "Qwen/"
            "Qwen3-VL-2B-Instruct"
        ),
    )

    parser.add_argument(
        "--yolo-model",
        default="yolo26n.pt",
    )

    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
    )

    parser.add_argument(
        "--anchor-x",
        type=int,
        default=1260,
    )

    parser.add_argument(
        "--anchor-y",
        type=int,
        default=550,
    )

    parser.add_argument(
        "--start-sec",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--max-sec",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--tcn-interval-sec",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--trigger-persist-sec",
        type=float,
        default=0.6,
    )

    parser.add_argument(
        "--clear-persist-sec",
        type=float,
        default=0.6,
    )

    parser.add_argument(
        "--qwen-recheck-sec",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--context-seconds",
        type=float,
        default=6.0,
    )

    parser.add_argument(
        "--candidate-seconds",
        type=float,
        default=2.5,
    )

    parser.add_argument(
        "--context-fps",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--candidate-fps",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=24,
    )

    parser.add_argument(
        "--active-print-sec",
        type=float,
        default=2.0,
        help=(
            "ACTIVE result heartbeat interval. "
            "Set 0.2 to print every TCN tick."
        ),
    )

    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Pace MP4 so 1 video second "
            "approximately equals 1 wall-clock second."
        ),
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help=(
            "Show one live dashboard window: "
            "current frame on the left and "
            "analysis status on the right."
        ),
    )

    parser.add_argument(
        "--keep-qwen-frames",
        action="store_true",
    )

    parser.add_argument(
        "--work-dir",
        default=None,
    )

    args = parser.parse_args()

    video = Path(args.video)

    if not video.is_absolute():
        video = ROOT / video

    video = video.resolve()

    if not video.exists():
        raise FileNotFoundError(video)

    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = ROOT / config_path

    ensemble_config = Path(
        args.ensemble_config
    )

    if not ensemble_config.is_absolute():
        ensemble_config = (
            ROOT
            / ensemble_config
        )

    cfg = load_config(
        config_path
    )

    target_fps = float(
        cfg["target_fps"]
    )

    if not math.isclose(
        target_fps,
        10.0,
        abs_tol=1e-6,
    ):
        raise RuntimeError(
            "Current TCN was trained on 10 Hz input. "
            f"config target_fps={target_fps}"
        )

    feature_window_len = 100

    tcn_interval_samples = max(
        1,
        int(
            round(
                args.tcn_interval_sec
                * target_fps
            )
        ),
    )

    trigger_persist_ticks = max(
        1,
        int(
            math.ceil(
                args.trigger_persist_sec
                / args.tcn_interval_sec
            )
        ),
    )

    clear_persist_ticks = max(
        1,
        int(
            math.ceil(
                args.clear_persist_sec
                / args.tcn_interval_sec
            )
        ),
    )

    work_dir = (
        Path(args.work_dir)
        if args.work_dir
        else (
            ROOT
            / "data"
            / "realtime_runtime"
            / video.stem
        )
    )

    if not work_dir.is_absolute():
        work_dir = ROOT / work_dir

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    latest_json = (
        work_dir
        / "latest_status.json"
    )

    results_csv = (
        work_dir
        / "realtime_results.csv"
    )

    evaluation_csv = (
        work_dir
        / "evaluation_timeline.csv"
    )

    jobs_dir = (
        work_dir
        / "qwen_jobs"
    )

    jobs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 72)
    print("REAL-TIME DRIVER STATE STREAM")
    print("=" * 72)
    print("Video:", video)
    print("Device:", device)
    print("Feature rate:", f"{target_fps:.1f} Hz")
    print(
        "TCN interval:",
        f"{args.tcn_interval_sec:.1f} s"
    )
    print(
        "Trigger persistence:",
        f"{args.trigger_persist_sec:.1f} s"
    )
    print(
        "Real-time pacing:",
        args.realtime
    )

    print()
    print("Loading 5-seed TCN ensemble...")

    ensemble = TCNEnsemble(
        ensemble_config,
        device,
    )

    print(
        "TCN models:",
        len(ensemble.members)
    )

    print(
        "TCN threshold:",
        ensemble.threshold
    )

    cap = cv2.VideoCapture(
        str(video)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video}"
        )

    src_fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    n_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if src_fps <= 0:
        raise RuntimeError(
            f"Invalid source FPS: {src_fps}"
        )

    duration = (
        n_frames
        / src_fps
    )

    start_sec = max(
        0.0,
        float(args.start_sec),
    )

    end_sec = duration

    if args.max_sec is not None:
        end_sec = min(
            duration,
            start_sec
            + float(args.max_sec),
        )

    if start_sec >= end_sec:
        raise RuntimeError(
            "Invalid start/end time."
        )

    cap.set(
        cv2.CAP_PROP_POS_MSEC,
        start_sec * 1000.0,
    )

    frame_idx = int(
        round(
            start_sec
            * src_fps
        )
    )

    driver_selector = DriverSelector(
        model_path=args.yolo_model,
        tracker=args.tracker,
        anchor_x=args.anchor_x,
        anchor_y=args.anchor_y,
        fps=src_fps,
    )

    extractor = (
        IncrementalFeatureExtractor(
            target_fps=target_fps
        )
    )

    # Keep slightly more than needed for:
    # 6 s context + 2.5 s candidate.
    qwen_buffer_sec = max(
        10.0,
        args.context_seconds
        + args.candidate_seconds
        + 1.0,
    )

    frame_buffer = deque(
        maxlen=int(
            math.ceil(
                qwen_buffer_sec
                * target_fps
            )
        )
    )

    feature_buffer = deque(
        maxlen=feature_window_len
    )

    qwen = QwenVerifier(
        model_name=args.qwen_model,
        max_new_tokens=args.max_new_tokens,
    )

    executor = ThreadPoolExecutor(
        max_workers=1
    )

    qwen_future = None

    results_rows = []
    evaluation_rows = []

    next_sample_t = start_sec

    sample_count = 0

    trigger_streak = 0
    clear_streak = 0

    event_active = False
    event_id = 0

    last_qwen_submit_t = -1e18
    last_active_print_t = -1e18

    latest_tcn_probability = 0.0
    latest_tcn_inference_sec = float("nan")

    # Last finalized 3-class state. "verifying" is UI-only and is
    # never written as a class prediction to evaluation_timeline.csv.
    last_finalized_response = None
    last_finalized_inference_sec = float("nan")
    last_finalized_source = None

    # 1 Hz evaluation timeline. Warm-up seconds are skipped because
    # the TCN requires a full 100 x 22 (10 s) rolling window.
    next_eval_t = float(math.ceil(start_sec - 1e-9))

    # Shared state rendered in the dashboard.
    latest_ui_status = {
        "driver_response": "warming_up",
        "purposeful_action": "not_applicable",
        "self_response": "not_applicable",
        "external_contact": "not_applicable",
        "response_after_contact": "not_applicable",
        "visibility": "not_applicable",
    }

    qwen_status = "IDLE"

    wall_start = time.perf_counter()

    def write_active_status(
        video_time,
        probability,
        inference_sec,
    ):
        nonlocal last_active_print_t
        nonlocal latest_ui_status
        nonlocal last_finalized_response
        nonlocal last_finalized_inference_sec
        nonlocal last_finalized_source

        result = {
            "video_time":
                float(video_time),
            "driver_response":
                "active",
            "decision_reason":
                "tcn_clear",
            "tcn_probability":
                float(probability),
            "tcn_threshold":
                float(
                    ensemble.threshold
                ),
            "candidate_detected":
                False,
            "purposeful_action":
                "not_applicable",
            "self_response":
                "not_applicable",
            "external_contact":
                "not_applicable",
            "response_after_contact":
                "not_applicable",
            "visibility":
                "not_applicable",
            "inference_time_sec":
                float(inference_sec),
        }

        save_latest(
            latest_json,
            result,
        )

        latest_ui_status = result.copy()

        last_finalized_response = "active"
        last_finalized_inference_sec = float(inference_sec)
        last_finalized_source = "tcn_clear"

        print_result(
            driver_response="active",
            tcn_probability=probability,
            threshold=ensemble.threshold,
            candidate=False,
            purposeful_action="not_applicable",
            self_response="not_applicable",
            external_contact="not_applicable",
            response_after_contact="not_applicable",
            visibility="not_applicable",
            inference_sec=inference_sec,
        )

        last_active_print_t = (
            float(video_time)
        )

    def consume_qwen_if_done():
        nonlocal qwen_future
        nonlocal latest_ui_status
        nonlocal qwen_status
        nonlocal last_finalized_response
        nonlocal last_finalized_inference_sec
        nonlocal last_finalized_source

        if qwen_future is None:
            return

        if not qwen_future.done():
            return

        try:
            result = qwen_future.result()

            result[
                "tcn_threshold"
            ] = float(
                ensemble.threshold
            )

            result[
                "candidate_detected"
            ] = True

            result[
                "inference_time_sec"
            ] = (
                float(
                    result[
                        "verification_wall_sec"
                    ]
                )
                +
                float(
                    result[
                        "tcn_stage_sec"
                    ]
                )
            )

            save_latest(
                latest_json,
                result,
            )

            latest_ui_status = result.copy()
            qwen_status = "DONE"

            last_finalized_response = str(
                result["driver_response"]
            ).strip().lower()
            last_finalized_inference_sec = float(
                result["inference_time_sec"]
            )
            last_finalized_source = "qwen_supervisor"

            results_rows.append(
                result.copy()
            )

            pd.DataFrame(
                results_rows
            ).to_csv(
                results_csv,
                index=False,
            )

            print_result(
                driver_response=(
                    result[
                        "driver_response"
                    ]
                ),
                tcn_probability=(
                    result[
                        "tcn_probability"
                    ]
                ),
                threshold=(
                    ensemble.threshold
                ),
                candidate=True,
                purposeful_action=(
                    result[
                        "purposeful_action"
                    ]
                ),
                self_response=(
                    result[
                        "self_response"
                    ]
                ),
                external_contact=(
                    result[
                        "external_contact"
                    ]
                ),
                response_after_contact=(
                    result[
                        "response_after_contact"
                    ]
                ),
                visibility=(
                    result[
                        "visibility"
                    ]
                ),
                inference_sec=(
                    result[
                        "inference_time_sec"
                    ]
                ),
            )

            if not args.keep_qwen_frames:
                job_dir = (
                    jobs_dir
                    / f"event_{int(result['event_id']):04d}"
                    / f"t_{float(result['video_time']):010.3f}"
                )

                if job_dir.exists():
                    import shutil
                    shutil.rmtree(
                        job_dir,
                        ignore_errors=True,
                    )

        except Exception as exc:
            qwen_status = "ERROR"
            print()
            print(
                "Qwen verification error:",
                type(exc).__name__,
                str(exc),
                flush=True,
            )

        qwen_future = None

    def append_evaluation_if_due(video_time):
        """
        Save one operational system state per video second.

        Important:
        - warm-up (<100 features) is not scored
        - Qwen RUNNING is not treated as a fourth class
        - while Qwen is running, the most recent finalized 3-class
          state remains the operational output
        """
        nonlocal next_eval_t

        while (
            float(video_time) + 1e-9
            >= next_eval_t
        ):
            if len(feature_buffer) >= feature_window_len:
                predicted = (
                    last_finalized_response
                    if last_finalized_response
                    in {
                        "active",
                        "reduced",
                        "no_visible_response",
                    }
                    else "pending"
                )

                row = {
                    "timestamp":
                        float(next_eval_t),
                    "predicted_response":
                        predicted,
                    "tcn_probability":
                        float(latest_tcn_probability),
                    "tcn_threshold":
                        float(ensemble.threshold),
                    "candidate_detected":
                        int(event_active),
                    "qwen_status":
                        str(qwen_status),
                    "decision_source":
                        (
                            last_finalized_source
                            if last_finalized_source
                            is not None
                            else "pending"
                        ),
                    "tcn_inference_time_sec":
                        float(
                            latest_tcn_inference_sec
                        ),
                    "decision_inference_time_sec":
                        float(
                            last_finalized_inference_sec
                        ),
                }

                evaluation_rows.append(
                    row
                )

                pd.DataFrame(
                    evaluation_rows
                ).to_csv(
                    evaluation_csv,
                    index=False,
                )

            next_eval_t += 1.0

    try:
        while True:
            # Surface completed Qwen result without waiting
            # for another 10 Hz sample.
            consume_qwen_if_done()

            ok, frame = cap.read()

            if not ok:
                break

            frame_t = (
                frame_idx
                / src_fps
            )

            frame_idx += 1

            if frame_t > end_sec + 1e-9:
                break

            # Match the offline extractor:
            # track driver on EVERY source frame.
            driver_info = (
                driver_selector.update(
                    frame
                )
            )

            if (
                frame_t + 1e-9
                < next_sample_t
            ):
                if args.realtime:
                    target_elapsed = (
                        frame_t
                        - start_sec
                    )

                    wall_elapsed = (
                        time.perf_counter()
                        - wall_start
                    )

                    sleep_sec = (
                        target_elapsed
                        - wall_elapsed
                    )

                    if sleep_sec > 0:
                        time.sleep(
                            sleep_sec
                        )

                continue

            sample_start = (
                time.perf_counter()
            )

            t = next_sample_t

            next_sample_t += (
                1.0 / target_fps
            )

            if driver_info is not None:
                driver_box = (
                    expand_driver_box(
                        driver_info["box"],
                        frame.shape,
                    )
                )
            else:
                driver_box = None

            if driver_box is not None:
                x1, y1, x2, y2 = (
                    driver_box
                )

                crop = frame[
                    y1:y2,
                    x1:x2,
                ]
            else:
                # Never replace the driver
                # with the passenger.
                crop = np.zeros(
                    (480, 480, 3),
                    dtype=np.uint8,
                )

            feature_vector, _ = (
                extractor.extract(
                    crop
                )
            )

            feature_buffer.append(
                feature_vector
            )

            # Store a compact driver-only visual stream
            # for asynchronous Qwen verification.
            qwen_crop = cv2.resize(
                crop,
                (512, 448),
                interpolation=cv2.INTER_AREA,
            )

            frame_buffer.append(
                (
                    float(t),
                    qwen_crop.copy(),
                )
            )

            sample_count += 1

            should_run_tcn = (
                len(feature_buffer)
                == feature_window_len
                and (
                    sample_count
                    - feature_window_len
                )
                % tcn_interval_samples
                == 0
            )

            if should_run_tcn:
                window = np.stack(
                    list(feature_buffer)
                ).astype(
                    np.float32
                )

                tcn_start = (
                    time.perf_counter()
                )

                (
                    probability,
                    seed_probs,
                ) = ensemble.predict(
                    window
                )

                tcn_sec = (
                    time.perf_counter()
                    - tcn_start
                )

                latest_tcn_inference_sec = float(
                    tcn_sec
                )

                latest_tcn_probability = (
                    probability
                )

                candidate = (
                    probability
                    >= ensemble.threshold
                )

                if candidate:
                    trigger_streak += 1
                    clear_streak = 0

                    if (
                        not event_active
                        and trigger_streak
                        >= trigger_persist_ticks
                    ):
                        event_active = True
                        event_id += 1

                        print()
                        print(
                            f"Candidate detected "
                            f"(P={probability:.3f}). "
                            f"Event {event_id}.",
                            flush=True,
                        )

                    if event_active:
                        can_recheck = (
                            float(t)
                            - last_qwen_submit_t
                            >= args.qwen_recheck_sec
                        )

                        if (
                            qwen_future is None
                            and can_recheck
                        ):
                            candidate_start = (
                                float(t)
                                - args.candidate_seconds
                            )

                            context_end = (
                                candidate_start
                            )

                            context_start = (
                                context_end
                                - args.context_seconds
                            )

                            context_images = (
                                select_frames(
                                    frame_buffer,
                                    context_start,
                                    context_end,
                                    args.context_fps,
                                )
                            )

                            candidate_images = (
                                select_frames(
                                    frame_buffer,
                                    candidate_start,
                                    float(t),
                                    args.candidate_fps,
                                )
                            )

                            if (
                                len(context_images)
                                >= 1
                                and len(candidate_images)
                                >= 1
                            ):
                                job_dir = (
                                    jobs_dir
                                    / f"event_{event_id:04d}"
                                    / f"t_{float(t):010.3f}"
                                )

                                context_paths = (
                                    save_sequence(
                                        context_images,
                                        job_dir
                                        / "context_frames",
                                    )
                                )

                                candidate_paths = (
                                    save_sequence(
                                        candidate_images,
                                        job_dir
                                        / "candidate_frames",
                                    )
                                )

                                job = {
                                    "event_id":
                                        event_id,
                                    "video_time":
                                        float(t),
                                    "tcn_probability":
                                        float(
                                            probability
                                        ),
                                    "tcn_stage_sec":
                                        float(
                                            time.perf_counter()
                                            - sample_start
                                        ),
                                    "context_paths":
                                        context_paths,
                                    "candidate_paths":
                                        candidate_paths,
                                }

                                qwen_status = "RUNNING"

                                latest_ui_status = {
                                    "driver_response":
                                        "verifying",
                                    "purposeful_action":
                                        "not_applicable",
                                    "self_response":
                                        "not_applicable",
                                    "external_contact":
                                        "not_applicable",
                                    "response_after_contact":
                                        "not_applicable",
                                    "visibility":
                                        "not_applicable",
                                }

                                print(
                                    "Qwen3-VL-2B "
                                    "verification started...",
                                    flush=True,
                                )

                                qwen_future = (
                                    executor.submit(
                                        qwen.verify,
                                        job,
                                        ensemble.threshold,
                                    )
                                )

                                last_qwen_submit_t = (
                                    float(t)
                                )
                else:
                    trigger_streak = 0
                    clear_streak += 1

                    if (
                        event_active
                        and clear_streak
                        >= clear_persist_ticks
                    ):
                        event_active = False

                    active_due = (
                        float(t)
                        - last_active_print_t
                        >= args.active_print_sec
                    )

                    if active_due:
                        write_active_status(
                            video_time=t,
                            probability=probability,
                            inference_sec=(
                                time.perf_counter()
                                - sample_start
                            ),
                        )

            append_evaluation_if_due(t)

            if args.preview:
                dashboard = render_dashboard(
                    frame=frame,
                    driver_box=driver_box,
                    video_time=t,
                    threshold=ensemble.threshold,
                    probability=(
                        latest_tcn_probability
                        if len(feature_buffer) >= 100
                        else None
                    ),
                    candidate=event_active,
                    feature_buffer_size=len(
                        feature_buffer
                    ),
                    qwen_status=qwen_status,
                    latest_status=latest_ui_status,
                )

                cv2.imshow(
                    "Real-Time Driver Monitor",
                    dashboard,
                )

                if (
                    cv2.waitKey(1)
                    & 0xFF
                ) == ord("q"):
                    break

            if args.realtime:
                target_elapsed = (
                    frame_t
                    - start_sec
                )

                wall_elapsed = (
                    time.perf_counter()
                    - wall_start
                )

                sleep_sec = (
                    target_elapsed
                    - wall_elapsed
                )

                if sleep_sec > 0:
                    time.sleep(
                        sleep_sec
                    )

        # Wait for the last already-started semantic
        # verification so its final 3-class result is not lost.
        if qwen_future is not None:
            print()
            print(
                "Waiting for final Qwen "
                "verification...",
                flush=True,
            )

            while not qwen_future.done():
                time.sleep(0.05)

            consume_qwen_if_done()

    finally:
        cap.release()
        extractor.close()

        if args.preview:
            cv2.destroyAllWindows()

        executor.shutdown(
            wait=True
        )

    print()
    print("=" * 72)
    print("STREAM COMPLETE")
    print("=" * 72)

    print(
        "Latest status:",
        latest_json
    )

    print(
        "1 Hz evaluation timeline:",
        evaluation_csv
    )

    if results_rows:
        print(
            "Qwen/Supervisor results:",
            results_csv
        )


if __name__ == "__main__":
    main()
