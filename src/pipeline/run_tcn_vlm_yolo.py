#!/usr/bin/env python3
"""DGX-side driver monitor that publishes observations to the bridge.

The legacy path sends files/clips to ``POST /monitor``.  ``--video-path`` adds
a full-source-video streaming path: it displays the source immediately, runs
the 24-D TCN on a 5-second rolling window at 10 Hz, and sends only persistent non-active event evidence to
the bridge.  The bridge owns the Qwen inference and final three-class decision.
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
# sibling src/features and src/models directories.  Modules are loaded by path
# so the runners work straight from a clone without a package install.
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
REPO_ROOT = SRC_DIR.parent
FEATURES_DIR = SRC_DIR / "features"
MODELS_DIR = SRC_DIR / "models"
WEIGHTS_DIR = REPO_ROOT / "assets" / "weights"
if str(FEATURES_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURES_DIR))

from fast_onset import motion_evidence, fuse_onset_risk


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
RUNTIME_ROOT = REPO_ROOT
DEFAULT_TCN_CONFIG = REPO_ROOT / "configs" / "tcn" / "tcn_1p5s_32d.json"
DEFAULT_TCN_ENSEMBLE = REPO_ROOT / "configs" / "ensembles" / "observable_state_tcn_v38.json"

# The ordering is part of the trained-model contract. Any ensemble used by
# this runner must have been trained with precisely these 32 features.
TCN_FEATURE_NAMES = [
    "ear_left", "ear_right", "ear_mean", "eye_valid", "ear_asymmetry",
    "ear_velocity", "eye_closed", "eye_closed_duration", "perclos_1s",
    "head_pitch", "head_yaw", "head_roll", "pitch_velocity",
    "yaw_velocity", "roll_velocity", "head_angular_speed",
    "head_to_shoulder_dx", "head_to_shoulder_dy", "head_relative_vx",
    "head_relative_vy", "upper_body_motion", "face_valid",
    "face_reliability", "pose_reliability",
    "eye_blink_left", "eye_blink_right", "eye_blink_mean",
    "eye_blendshape_valid", "seat_head_dx", "seat_head_dy",
    "seat_head_vx", "seat_head_vy",
]
# A separate wrist/hand branch consumes these.  They are deliberately not
# appended to the 32-D NVR TCN: that would silently invalidate its weights.
CONTROL_FEATURE_NAMES = [
    "left_wrist_dx", "left_wrist_dy", "right_wrist_dx", "right_wrist_dy",
    "left_wrist_vx", "left_wrist_vy", "right_wrist_vx", "right_wrist_vy",
    "wrist_separation", "left_wrist_visibility", "right_wrist_visibility",
    "left_hand_landmarks_valid", "right_hand_landmarks_valid",
]
EYE_CLOSED_EAR_THRESHOLD = 0.20
# EAR is only a geometry proxy.  A large single-eye value or a large left/right
# disagreement usually means profile view, an occluding hand, or a bad face
# mesh fit; it must never be interpreted as confident "eyes open" evidence.
EYE_EAR_MAX_RELIABLE = 0.50
EYE_EAR_MAX_ASYMMETRY = 0.18
# The original TCN feature set was extracted from this driver-side ROI in a
# 1728x960 cabin video. Keep it normalized for other resolutions.
DEFAULT_FALLBACK_DRIVER_ROI = (765 / 1728, 108 / 960, 1.0, 1.0)
# In this right-driver cabin view, prevent the event VLM crop from expanding
# into the passenger seat when YOLO's body box is broad or merged.
DEFAULT_VLM_DRIVER_SEAT_LEFT_RATIO = 0.57
EYE_BLENDSHAPE_MODEL_PATH = WEIGHTS_DIR / "face_landmarker.task"


class RequestError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def load_runtime_module(name: str, path: Path) -> Any:
    """Import one of the validated TCN runtime modules by absolute path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"모듈을 불러올 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_repo_path(raw: str | Path) -> Path:
    """Resolve config-relative asset paths against the repository root."""
    path = Path(str(raw))
    return path if path.is_absolute() else REPO_ROOT / path


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


class StreamingTCN:
    """Load either the legacy binary gate or the direct three-state TCN."""

    def __init__(self, config_path: Path) -> None:
        try:
            import numpy as np
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "전체 영상 TCN 모드에는 numpy와 torch가 필요합니다."
            ) from exc
        self.np = np
        self.torch = torch
        self.common = load_runtime_module("driver_common_runtime", FEATURES_DIR / "common.py")
        self.features = load_runtime_module(
            "driver_features_runtime", FEATURES_DIR / "extract_features.py"
        )
        self.feature_names = list(TCN_FEATURE_NAMES)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.model_type = str(config.get("ensemble_type", "mean_probability"))
        is_multitask = self.model_type in (
            "mean_multitask_state_probability",
            "mean_observable_state_probability",
        )
        configured_module = config.get("model_module")
        model_path = resolve_repo_path(configured_module) if configured_module else None
        if model_path is not None and not model_path.is_file():
            raise FileNotFoundError(f"TCN model module이 없습니다: {model_path}")
        model_module = load_runtime_module(
            "multitask_state_tcn_runtime" if is_multitask else "binary_tcn_runtime",
            model_path if model_path is not None else MODELS_DIR / (
                "multitask_state_tcn.py" if is_multitask else "binary_tcn.py"
            ),
        )
        self.threshold = float(config["selected_threshold"])
        self.transition_threshold = float(config.get("transition_threshold", 0.5))
        self.early_warning_horizon_sec = float(
            config.get("early_warning_horizon_sec", 0.0)
        )
        self.transition_operational = bool(config.get("transition_operational", True))
        self.vlm_state_distribution_operational = bool(
            config.get("vlm_state_distribution_operational", True)
        )
        # The three-way state head still exposes a reduced probability for
        # diagnostics and VLM context.  Whether that head may directly emit an
        # operational reduced state is a separate, explicitly configured
        # policy.
        self.reduced_head_operational = bool(
            config.get("reduced_head_operational", False)
        )
        self.class_names = list(config.get("class_names", []))
        self.last_state_probabilities: dict[str, float] | None = None
        self.last_driver_response: str | None = None
        self.last_raw_driver_response: str | None = None
        self.last_transition_probability: float | None = None
        self.last_nvr_candidate: bool | None = None
        self.last_control_reduced_confirmed: bool | None = None
        self.control_disengagement_streak = 0
        self.last_observation_quality: float | None = None
        self.last_observable: bool | None = None
        self.minimum_observation_quality = config.get("minimum_observation_quality")
        self.minimum_state_confidence = float(config.get("minimum_state_confidence", 0.0))
        self.state_decision = str(config.get("state_decision", "argmax_mean_softmax"))
        self.observation_quality_fn = getattr(model_module, "observation_quality", None)
        self.window_steps = int(config.get("window_steps", 100))
        if self.window_steps < 2:
            raise RuntimeError(f"TCN ensemble window_steps가 올바르지 않습니다: {self.window_steps}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.members: list[dict[str, Any]] = []
        for raw_path in config["model_paths"]:
            model_path = resolve_repo_path(raw_path)
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            checkpoint_features = list(checkpoint["feature_names"])
            if checkpoint_features != self.feature_names:
                if int(checkpoint.get("feature_dim", len(checkpoint_features))) == 22:
                    raise RuntimeError(
                        "지정한 TCN은 구형 22-D 가중치입니다. 이 모델에는 "
                        "eye_closed/eye_closed_duration/perclos_1s 특징이 없어 "
                        "눈 감김 보완을 적용할 수 없습니다. "
                        "config.json + models/binary_tcn_ensemble_v24_5s.json "
                        "(24-D, 5초)을 사용하거나 24-D/1.5초 가중치를 새로 학습하세요: "
                        f"{model_path}"
                    )
                raise RuntimeError(
                    "TCN feature 순서가 맞지 않습니다. "
                    f"새 24-D feature로 재학습한 가중치를 사용하세요: {model_path}"
                )
            if int(checkpoint["feature_dim"]) != len(self.feature_names):
                raise RuntimeError(
                    f"TCN feature_dim이 {len(self.feature_names)}가 아닙니다: {model_path}"
                )
            checkpoint_type = checkpoint.get("model_type", "binary_tcn")
            if is_multitask:
                if checkpoint_type not in ("multitask_state_transition_tcn", "observable_state_tcn"):
                    raise RuntimeError(f"멀티태스크 ensemble에 이진 모델이 포함됐습니다: {model_path}")
                architecture_kwargs = {}
                if checkpoint_type == "observable_state_tcn":
                    architecture_kwargs["dilations"] = tuple(checkpoint["dilations"])
                if list(checkpoint.get("class_names", [])) != self.class_names:
                    raise RuntimeError(f"TCN class 순서가 ensemble과 다릅니다: {model_path}")
                model = model_module.MultiTaskDriverTCN(
                    feature_dim=int(checkpoint["feature_dim"]),
                    hidden_dim=int(checkpoint["hidden_dim"]),
                    num_classes=len(self.class_names),
                    dropout=float(checkpoint["dropout"]),
                    **architecture_kwargs,
                ).to(self.device)
            else:
                model = model_module.BinaryDriverTCN(
                    feature_dim=int(checkpoint["feature_dim"]),
                    hidden_dim=int(checkpoint["hidden_dim"]),
                    dropout=float(checkpoint["dropout"]),
                ).to(self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            self.members.append(
                {
                    "model": model,
                    "mean": torch.as_tensor(
                        np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
                        .reshape(1, 1, -1), device=self.device,
                    ),
                    "std": torch.as_tensor(
                        np.asarray(checkpoint["normalization_std"], dtype=np.float32)
                        .reshape(1, 1, -1), device=self.device,
                    ),
                    "seed": int(checkpoint.get("seed", len(self.members))),
                    "absolute_value_features": list(
                        checkpoint.get("absolute_value_features", [])
                    ),
                }
            )
        if not self.members:
            raise RuntimeError("TCN ensemble에 모델이 없습니다.")

    def predict(self, window: Any) -> tuple[float, dict[str, float]]:
        expected = (self.window_steps, len(self.feature_names))
        if window.shape != expected:
            raise ValueError(f"TCN 입력은 {expected}이어야 합니다.")
        torch = self.torch
        with torch.inference_mode():
            raw_values = window.astype(self.np.float32, copy=True)
            per_seed: dict[str, float] = {}
            state_members: list[Any] = []
            transition_members: list[float] = []
            for member in self.members:
                member_values = raw_values.copy()
                for feature_name in member["absolute_value_features"]:
                    if feature_name in self.feature_names:
                        feature_index = self.feature_names.index(feature_name)
                        member_values[:, feature_index] = self.np.abs(
                            member_values[:, feature_index]
                        )
                values = torch.from_numpy(member_values).unsqueeze(0).to(self.device)
                normalized = (values - member["mean"]) / member["std"]
                output = member["model"](normalized)
                if is_multitask := self.model_type in (
                    "mean_multitask_state_probability",
                    "mean_observable_state_probability",
                ):
                    state_logits, transition_logit = output
                    state_probability = torch.softmax(state_logits, dim=1)[0]
                    probability = float(state_probability[2].item())
                    state_members.append(state_probability.cpu().numpy())
                    transition_members.append(float(torch.sigmoid(transition_logit).item()))
                else:
                    probability = float(torch.sigmoid(output).item())
                per_seed[f"seed{member['seed']}"] = probability
        if self.model_type in ("mean_multitask_state_probability", "mean_observable_state_probability"):
            mean_state = self.np.mean(self.np.stack(state_members), axis=0)
            self.last_state_probabilities = {
                name: float(mean_state[index])
                for index, name in enumerate(self.class_names)
            }
            self.last_raw_driver_response = self.class_names[int(mean_state.argmax())]
            self.last_nvr_candidate = bool(float(mean_state[2]) >= self.threshold)
            if self.observation_quality_fn is not None:
                quality = self.observation_quality_fn(
                    raw_values[None, :, :], self.feature_names
                )
                self.last_observation_quality = float(self.np.asarray(quality).reshape(-1)[0])
            else:
                self.last_observation_quality = None
            observable = (
                self.last_observation_quality is None
                or self.minimum_observation_quality is None
                or self.last_observation_quality >= float(self.minimum_observation_quality)
            ) and float(mean_state.max()) >= self.minimum_state_confidence
            self.last_observable = bool(observable)
            if not observable:
                self.last_driver_response = "unobservable"
            elif self.state_decision == "nvr_argmax_when_observable_else_unobservable":
                # The old reduced head has F1=0.  Do not masquerade a model
                # artifact as a meaningful impairment state.  Reduced may only
                # be added later by a separately calibrated control branch.
                self.last_driver_response = (
                    "no_visible_response"
                    if self.last_raw_driver_response == "no_visible_response" else "active"
                )
            else:
                self.last_driver_response = self.last_raw_driver_response
            self.last_transition_probability = float(self.np.mean(transition_members))
            return float(mean_state[2]), per_seed
        self.last_state_probabilities = None
        self.last_driver_response = None
        self.last_raw_driver_response = None
        self.last_transition_probability = None
        self.last_nvr_candidate = None
        self.last_control_reduced_confirmed = None
        self.last_observation_quality = None
        self.last_observable = None
        return float(self.np.mean(list(per_seed.values()))), per_seed

    def apply_control_reduced(
        self, control_runtime: "StreamingControlTCN", confirmation_windows: int,
    ) -> None:
        """Promote sustained control disengagement only from an active state."""
        confirmation_windows = max(1, int(confirmation_windows))
        candidate = bool(control_runtime.last_control_disengaged_candidate)
        if self.last_driver_response != "active" or not candidate:
            self.control_disengagement_streak = 0
            self.last_control_reduced_confirmed = False
            return
        self.control_disengagement_streak += 1
        confirmed = self.control_disengagement_streak >= confirmation_windows
        self.last_control_reduced_confirmed = confirmed
        if control_runtime.operational and confirmed:
            self.last_driver_response = "reduced"


class StreamingControlTCN:
    """Independent wrist/hand TCN used only for a control-engagement signal.

    Its semantics are intentionally narrower than `reduced`: it answers
    whether the observed wrist geometry resembles hands away from the wheel.
    The model is opt-in through the primary ensemble JSON and remains logging
    only while its `control_operational` flag is false.
    """

    def __init__(self, config_path: Path) -> None:
        try:
            import numpy as np
            import torch
        except ImportError as exc:
            raise RuntimeError("제어 관여 TCN에는 numpy와 torch가 필요합니다.") from exc
        self.np, self.torch = np, torch
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if str(config.get("ensemble_type")) != "mean_control_engagement_probability":
            raise RuntimeError(f"제어 TCN ensemble 형식이 올바르지 않습니다: {config_path}")
        if list(config.get("feature_names", [])) != CONTROL_FEATURE_NAMES:
            raise RuntimeError("제어 TCN feature 순서가 runtime과 다릅니다.")
        module_path = resolve_repo_path(config["model_module"])
        if not module_path.is_file():
            raise FileNotFoundError(f"제어 TCN model module이 없습니다: {module_path}")
        self.module = load_runtime_module("control_engagement_tcn_runtime", module_path)
        self.class_names = list(config["class_names"])
        self.window_steps = int(config["window_steps"])
        self.threshold = float(config.get("selected_threshold", 0.50))
        self.operational = bool(config.get("control_operational", False))
        self.confirmation_windows = int(config.get("reduced_confirmation_windows", 5))
        self.minimum_observation_quality = float(config.get("minimum_control_observation_quality", 0.70))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.members: list[dict[str, Any]] = []
        for raw_path in config["model_paths"]:
            checkpoint = torch.load(resolve_repo_path(raw_path), map_location="cpu", weights_only=False)
            if checkpoint.get("model_type") != "control_engagement_tcn":
                raise RuntimeError(f"제어 앙상블에 올바르지 않은 checkpoint가 있습니다: {raw_path}")
            model = self.module.ControlEngagementTCN(
                feature_dim=int(checkpoint["feature_dim"]), hidden_dim=int(checkpoint["hidden_dim"]),
                dropout=float(checkpoint["dropout"]), dilations=tuple(checkpoint["dilations"]),
            ).to(self.device)
            model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
            self.members.append({
                "model": model, "seed": int(checkpoint.get("seed", len(self.members))),
                "mean": torch.as_tensor(self.np.asarray(checkpoint["normalization_mean"], dtype=self.np.float32).reshape(1, 1, -1), device=self.device),
                "std": torch.as_tensor(self.np.asarray(checkpoint["normalization_std"], dtype=self.np.float32).reshape(1, 1, -1), device=self.device),
            })
        self.last_probabilities: dict[str, float] | None = None
        self.last_observation_quality: float | None = None
        self.last_observable: bool | None = None
        self.last_control_disengaged_candidate: bool | None = None
        self.last_member_probabilities: list[float] = []

    def predict(self, window: Any) -> dict[str, float]:
        expected = (self.window_steps, len(CONTROL_FEATURE_NAMES))
        if window.shape != expected:
            raise ValueError(f"제어 TCN 입력은 {expected}이어야 합니다.")
        raw = window.astype(self.np.float32, copy=False)
        # Wrist visibility is an observation gate, not evidence of engagement.
        quality = float(self.np.maximum(raw[:, 9], raw[:, 10]).mean())
        members: list[Any] = []
        with self.torch.inference_mode():
            for member in self.members:
                values = self.torch.from_numpy(raw).unsqueeze(0).to(self.device)
                logits = member["model"]((values - member["mean"]) / member["std"])
                members.append(self.torch.softmax(logits, dim=1)[0].cpu().numpy())
        mean = self.np.mean(self.np.stack(members), axis=0)
        self.last_member_probabilities = [float(member[1]) for member in members]
        self.last_probabilities = {name: float(mean[i]) for i, name in enumerate(self.class_names)}
        self.last_observation_quality = quality
        self.last_observable = quality >= self.minimum_observation_quality
        self.last_control_disengaged_candidate = bool(
            self.last_observable and mean[1] >= self.threshold
        )
        return self.last_probabilities


def resolve_control_ensemble_path(primary_config_path: Path, primary_config: dict[str, Any]) -> Path | None:
    raw = primary_config.get("control_ensemble_path")
    if not raw:
        return None
    return resolve_repo_path(raw)


class StreamingFeatureExtractor:
    """Incrementally extract the 32-D eye, head, and fixed-seat TCN input."""

    def __init__(
        self,
        runtime: StreamingTCN,
        target_fps: float = 10.0,
        eye_state_model_path: Path | None = None,
        eye_state_smooth_steps: int = 10,
        eye_closed_persist_steps: int = 6,
        eye_closed_probability_threshold: float = 0.70,
        eye_min_visible_fraction: float = 0.60,
    ) -> None:
        try:
            import cv2
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "전체 영상 TCN 모드에는 opencv-python과 mediapipe가 필요합니다."
            ) from exc
        self.cv2, self.mp, self.runtime = cv2, mp, runtime
        self.np = runtime.np
        self.dt = 1.0 / float(target_fps)
        self.prev_angles = None
        self.prev_ear_mean: float | None = None
        self.prev_head_relative = None
        self.prev_upper_body_points = None
        self.prev_wrist_relative = None
        self.latest_control_vector: Any | None = None
        self.seat_head_reference = None
        self.seat_reference_scale: float | None = None
        self.prev_seat_head = None
        self.eye_closed_samples: deque[int] = deque(
            maxlen=max(1, int(round(float(target_fps))))
        )
        self.eye_closed_duration = 0.0
        # Pose is still supplied by legacy Holistic when available.  Facial
        # geometry, eye crops, and blink blendshapes below all come from the
        # same FaceLandmarker result, avoiding contradictory eye detections.
        self.landmarker = None
        if getattr(mp, "solutions", None) is not None:
            self.landmarker = mp.solutions.holistic.Holistic(
                static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                enable_segmentation=False, refine_face_landmarks=False,
                min_detection_confidence=0.5, min_tracking_confidence=0.5,
            )
        if not EYE_BLENDSHAPE_MODEL_PATH.is_file():
            raise RuntimeError(
                f"눈 보조 모델이 없습니다: {EYE_BLENDSHAPE_MODEL_PATH}"
            )
        eye_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(EYE_BLENDSHAPE_MODEL_PATH)
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
        )
        self.eye_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            eye_options
        )
        self.eye_state = None
        self.eye_state_module = None
        if eye_state_model_path is not None:
            eye_module_path = MODELS_DIR / "eye_state_model.py"
            if not eye_module_path.is_file():
                raise FileNotFoundError(f"눈 상태 모델 모듈이 없습니다: {eye_module_path}")
            self.eye_state_module = load_runtime_module(
                "eye_state_specialist_runtime", eye_module_path
            )
            self.eye_state = self.eye_state_module.EyeStateSpecialist(
                eye_state_model_path,
                device=self.runtime.device,
                smooth_steps=eye_state_smooth_steps,
                closed_persist_steps=eye_closed_persist_steps,
                closed_probability_threshold=eye_closed_probability_threshold,
                min_visible_fraction=eye_min_visible_fraction,
            )

    def close(self) -> None:
        if self.landmarker is not None:
            self.landmarker.close()
        self.eye_landmarker.close()

    def extract(
        self,
        crop: Any,
        driver_box: tuple[int, int, int, int] | None = None,
        source_shape: tuple[int, ...] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        f, np, clip = self.runtime.features, self.np, self.runtime.common.clip_finite
        rgb = self.cv2.cvtColor(crop, self.cv2.COLOR_BGR2RGB)
        legacy_result = self.landmarker.process(rgb) if self.landmarker is not None else None
        eye_image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB, data=rgb
        )
        eye_result = self.eye_landmarker.detect(eye_image)
        if eye_result.face_blendshapes:
            eye_scores = {
                category.category_name: float(category.score)
                for category in eye_result.face_blendshapes[0]
            }
            eye_blink_l = eye_scores.get("eyeBlinkLeft", 0.0)
            eye_blink_r = eye_scores.get("eyeBlinkRight", 0.0)
            eye_blink_valid = 1
        else:
            eye_blink_l = eye_blink_r = 0.0
            eye_blink_valid = 0
        eye_blink_mean = (eye_blink_l + eye_blink_r) / 2.0
        eye_state_diagnostics: dict[str, float | int] = {
            "eye_open_probability": 0.0,
            "eye_closed_probability": 0.0,
            "eye_unknown_probability": 1.0,
            "eye_state_valid": 0,
            "eye_visible_fraction": 0.0,
            "eye_closed_sustained": 0,
            "eye_crop_contrast": 0.0,
            "eye_crop_sharpness": 0.0,
            "eye_crop_quality_valid": 0,
        }
        task_face = eye_result.face_landmarks[0] if eye_result.face_landmarks else None
        if self.eye_state is not None:
            eye_state_diagnostics = self.eye_state.predict(rgb, task_face)
        # FaceLandmarker uses the same mesh indices as the eye specialist and
        # is therefore the sole source for EAR, head pose, and face validity.
        # Legacy Holistic remains pose-only until the TCN is retrained with a
        # Tasks pose extractor.
        face = task_face
        pose = (
            legacy_result.pose_landmarks.landmark
            if legacy_result is not None and legacy_result.pose_landmarks
            else None
        )
        left_hand = (
            legacy_result.left_hand_landmarks.landmark
            if legacy_result is not None and legacy_result.left_hand_landmarks
            else None
        )
        right_hand = (
            legacy_result.right_hand_landmarks.landmark
            if legacy_result is not None and legacy_result.right_hand_landmarks
            else None
        )
        face_valid = int(face is not None and len(face) > 291)
        eye_valid = int(face_valid and max(f.LEFT_EYE + f.RIGHT_EYE) < len(face))
        face_rel = f.in_frame_ratio(face) if face_valid else 0.0
        pose_rel = f.pose_reliability(pose)
        if eye_valid:
            ear_l, ear_r = f.ear(face, f.LEFT_EYE), f.ear(face, f.RIGHT_EYE)
            ear_m = np.nanmean([ear_l, ear_r])
            if not np.isfinite(ear_m):
                eye_valid = 0
                ear_l = ear_r = ear_m = 0.0
        else:
            ear_l = ear_r = ear_m = 0.0
        ear_asymmetry = abs(ear_l - ear_r) if eye_valid else 0.0
        eye_geometry_reliable = bool(
            eye_valid
            and np.isfinite(ear_l)
            and np.isfinite(ear_r)
            and np.isfinite(ear_m)
            and 0.0 <= ear_l <= EYE_EAR_MAX_RELIABLE
            and 0.0 <= ear_r <= EYE_EAR_MAX_RELIABLE
            and ear_asymmetry <= EYE_EAR_MAX_ASYMMETRY
        )
        # The 32-D TCN was trained with zero-valued missing eye measurements.
        # Preserve that contract instead of leaking a geometrically impossible
        # EAR (for example 1.09) into the pretrained TCN.
        if not eye_geometry_reliable:
            eye_valid = 0
            ear_l = ear_r = ear_m = ear_asymmetry = 0.0
        ear_velocity = (
            (ear_m - self.prev_ear_mean) / self.dt
            if eye_valid and self.prev_ear_mean is not None else 0.0
        )
        self.prev_ear_mean = ear_m if eye_valid else None
        eye_closed = int(eye_valid and ear_m < EYE_CLOSED_EAR_THRESHOLD)
        self.eye_closed_duration = self.eye_closed_duration + self.dt if eye_closed else 0.0
        self.eye_closed_samples.append(eye_closed)
        perclos_1s = float(np.mean(self.eye_closed_samples))
        angles = f.head_pose(face, crop.shape[1], crop.shape[0]) if face_valid else None
        if angles is None:
            pitch = yaw = roll = pvel = yvel = rvel = 0.0
            self.prev_angles = None
        else:
            pitch, yaw, roll = angles
            if self.prev_angles is None:
                pvel = yvel = rvel = 0.0
            else:
                pvel = f.angle_delta(pitch, self.prev_angles[0]) / self.dt
                yvel = f.angle_delta(yaw, self.prev_angles[1]) / self.dt
                rvel = f.angle_delta(roll, self.prev_angles[2]) / self.dt
            self.prev_angles = angles
        pvel, yvel, rvel = (clip(value, -720, 720) for value in (pvel, yvel, rvel))
        head_angular_speed = float(math.sqrt(pvel ** 2 + yvel ** 2 + rvel ** 2))
        body_valid = bool(pose and len(pose) > 12)
        if body_valid:
            body_points = np.asarray(
                [[pose[index].x, pose[index].y] for index in (0, 11, 12)],
                dtype=np.float64,
            )
            shoulder_width = float(np.linalg.norm(body_points[1] - body_points[2]))
            valid_points = bool(
                np.isfinite(body_points).all() and shoulder_width >= 1e-6
                and all(-0.10 <= point[0] <= 1.10 and -0.10 <= point[1] <= 1.10
                        for point in body_points)
            )
        else:
            body_points, shoulder_width, valid_points = None, 0.0, False
        if not valid_points:
            head_dx = head_dy = head_vx = head_vy = upper_body_motion = 0.0
            self.prev_head_relative = self.prev_upper_body_points = None
        else:
            shoulder_center = (body_points[1] + body_points[2]) / 2.0
            head_relative = (body_points[0] - shoulder_center) / shoulder_width
            head_dx, head_dy = float(head_relative[0]), float(head_relative[1])
            if self.prev_head_relative is None:
                head_vx = head_vy = 0.0
            else:
                head_velocity = (head_relative - self.prev_head_relative) / self.dt
                head_vx, head_vy = float(head_velocity[0]), float(head_velocity[1])
            upper_body_motion = (
                float(np.linalg.norm(body_points - self.prev_upper_body_points, axis=1).mean()
                      / shoulder_width)
                if self.prev_upper_body_points is not None else 0.0
            )
            self.prev_head_relative = head_relative
            self.prev_upper_body_points = body_points
        if valid_points and driver_box is not None and source_shape is not None:
            source_h, source_w = source_shape[:2]
            x1, y1, _, _ = driver_box
            crop_h, crop_w = crop.shape[:2]
            global_points = np.empty_like(body_points)
            global_points[:, 0] = (x1 + body_points[:, 0] * crop_w) / source_w
            global_points[:, 1] = (y1 + body_points[:, 1] * crop_h) / source_h
            global_head = global_points[0]
            global_shoulder_width = float(
                np.linalg.norm(global_points[1] - global_points[2])
            )
            if global_shoulder_width >= 1e-6:
                if self.seat_head_reference is None:
                    self.seat_head_reference = global_head.copy()
                    self.seat_reference_scale = global_shoulder_width
                scale = max(float(self.seat_reference_scale), 1e-6)
                seat_delta = (global_head - self.seat_head_reference) / scale
                seat_head_dx, seat_head_dy = float(seat_delta[0]), float(seat_delta[1])
                if self.prev_seat_head is None:
                    seat_head_vx = seat_head_vy = 0.0
                else:
                    seat_velocity = (global_head - self.prev_seat_head) / (scale * self.dt)
                    seat_head_vx, seat_head_vy = float(seat_velocity[0]), float(seat_velocity[1])
                self.prev_seat_head = global_head
                baseline_alpha = min(0.02, self.dt / 15.0)
                self.seat_head_reference = (
                    (1.0 - baseline_alpha) * self.seat_head_reference
                    + baseline_alpha * global_head
                )
            else:
                seat_head_dx = seat_head_dy = seat_head_vx = seat_head_vy = 0.0
                self.prev_seat_head = None
        else:
            seat_head_dx = seat_head_dy = seat_head_vx = seat_head_vy = 0.0
            self.prev_seat_head = None
        # This side-channel exactly follows the offline 13-D feature contract.
        # It stays separate from the pretrained 32-D NVR vector.
        wrists_valid = bool(
            pose and len(pose) > 16 and valid_points
            and np.isfinite([
                pose[15].x, pose[15].y, pose[16].x, pose[16].y,
            ]).all()
        )
        if wrists_valid:
            shoulder_center = (body_points[1] + body_points[2]) / 2.0
            wrists = np.asarray([
                [pose[15].x, pose[15].y], [pose[16].x, pose[16].y],
            ], dtype=np.float64)
            wrist_relative = (wrists - shoulder_center) / shoulder_width
            wrist_velocity = (
                np.zeros((2, 2), dtype=np.float64)
                if self.prev_wrist_relative is None
                else (wrist_relative - self.prev_wrist_relative) / self.dt
            )
            self.prev_wrist_relative = wrist_relative
            wrist_separation = float(np.linalg.norm(wrists[0] - wrists[1]) / shoulder_width)
            left_wrist_visibility = float(np.clip(getattr(pose[15], "visibility", 0.0), 0.0, 1.0))
            right_wrist_visibility = float(np.clip(getattr(pose[16], "visibility", 0.0), 0.0, 1.0))
            control_values = [
                wrist_relative[0, 0], wrist_relative[0, 1], wrist_relative[1, 0], wrist_relative[1, 1],
                wrist_velocity[0, 0], wrist_velocity[0, 1], wrist_velocity[1, 0], wrist_velocity[1, 1],
                wrist_separation, left_wrist_visibility, right_wrist_visibility,
                float(left_hand is not None), float(right_hand is not None),
            ]
        else:
            self.prev_wrist_relative = None
            control_values = [0.0] * len(CONTROL_FEATURE_NAMES)
        self.latest_control_vector = np.asarray([clip(value) for value in control_values], dtype=np.float32)
        values = [
            ear_l, ear_r, ear_m, eye_valid, ear_asymmetry, ear_velocity,
            eye_closed, self.eye_closed_duration, perclos_1s,
            pitch, yaw, roll, pvel, yvel, rvel, head_angular_speed,
            head_dx, head_dy, head_vx, head_vy, upper_body_motion,
            face_valid, face_rel, pose_rel,
            eye_blink_l, eye_blink_r, eye_blink_mean, eye_blink_valid,
            seat_head_dx, seat_head_dy, seat_head_vx, seat_head_vy,
        ]
        if len(values) != len(TCN_FEATURE_NAMES):
            raise RuntimeError("내부 오류: 32-D TCN feature 벡터 길이가 올바르지 않습니다.")
        vector = np.asarray([clip(value) for value in values], dtype=np.float32)
        # When the specialist is enabled, its crop-quality gate is the
        # authoritative answer to "can we currently observe the eyes?".  The
        # legacy geometric fit remains diagnostic-only and cannot force an
        # unnecessary eye-uncertain watchdog when the specialist has a good
        # crop.
        eye_measurement_reliable = bool(
            eye_state_diagnostics["eye_state_valid"]
            if self.eye_state is not None else eye_geometry_reliable
        )
        # Keep the TCN feature vector exactly as it was trained.  Reliability
        # is an independent safety signal used only to prevent a weak eye
        # measurement from suppressing a near-threshold visual verification.
        return vector, {
            "eye_measurement_reliable": eye_measurement_reliable,
            "eye_geometry_reliable": int(eye_geometry_reliable),
            "ear_left": float(ear_l),
            "ear_right": float(ear_r),
            "ear_mean": float(ear_m),
            "eye_valid": int(eye_valid),
            "face_valid": int(face_valid),
            "face_reliability": float(face_rel),
            "eye_blink_left": float(eye_blink_l),
            "eye_blink_right": float(eye_blink_r),
            "eye_blink_mean": float(eye_blink_mean),
            "eye_blendshape_valid": int(eye_blink_valid),
            "seat_head_dx": float(seat_head_dx),
            "seat_head_dy": float(seat_head_dy),
            "control_observation_quality": float(max(control_values[9], control_values[10])),
            **eye_state_diagnostics,
        }


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
        "--tcn-score-dir",
        type=Path,
        default=None,
        help=(
            "preprocess_driver_safety_pipeline.py가 생성한 "
            "<media stem>.summary.json 디렉터리. 지정하면 TCN temporal "
            "evidence를 /monitor 요청에 함께 보냅니다."
        ),
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
        "--vlm-match-tcn-crop", action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "VLM 이벤트 영상에 TCN이 실제로 입력받은 YOLO driver crop을 그대로 사용. "
            "크롭 좌표·크기·추적 결과를 TCN과 동일하게 맞춘다."
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
            "전체 MP4를 실시간으로 재생·분석합니다. 5초/10 Hz TCN이 reduced 이상 "
            "후보를 지속 검출할 때만 Qwen 브리지에 이벤트 영상을 전송합니다."
        ),
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=RUNTIME_ROOT,
        help="저장소 루트(상대 경로 해석 기준)",
    )
    parser.add_argument(
        "--tcn-config",
        type=Path,
        default=None,
        help=(
            "TCN 입력 window 설정 JSON "
            "(기본: configs/tcn/tcn_1p5s_32d.json, 1.5초·10 Hz·32차원)"
        ),
    )
    parser.add_argument(
        "--tcn-ensemble-config",
        type=Path,
        default=None,
        help="주 TCN ensemble JSON (기본: configs/ensembles/observable_state_tcn_v38.json)",
    )
    parser.add_argument(
        "--fall-onset-tcn-config",
        type=Path,
        default=None,
        help=(
            "선택 사항: 쓰러짐 전이 전용 보조 TCN의 입력 window 설정 JSON. "
            "지정하면 보조 ensemble도 함께 지정해야 합니다."
        ),
    )
    parser.add_argument(
        "--fast-onset-risk", action=argparse.BooleanOptionalAction, default=False,
        help="최근 0.5초 하강 동작과 보조 onset TCN이 함께 감지될 때 조기 경고 점수를 올립니다.",
    )
    parser.add_argument(
        "--fast-onset-risk-floor", type=float, default=0.72,
        help="조기 경고 점수 하한; 의식 상실의 보정된 확률이 아닙니다 (기본: 0.72).",
    )
    parser.add_argument(
        "--fall-onset-tcn-ensemble-config",
        type=Path,
        default=None,
        help=(
            "선택 사항: 1초 fall-onset TCN ensemble JSON. "
            "최종 상태 판정이 아닌 조기 VLM 호출 트리거로만 사용합니다."
        ),
    )
    parser.add_argument(
        "--driver-yolo-model",
        type=Path,
        default=WEIGHTS_DIR / "yolo26n.pt",
        help="전체 영상에서 운전자만 추적할 YOLO 가중치",
    )
    parser.add_argument("--driver-tracker", default="bytetrack.yaml")
    parser.add_argument(
        "--driver-yolo-confidence",
        type=float,
        default=0.15,
        help="운전자 YOLO person 검출 최소 confidence (기본: 0.15)",
    )
    parser.add_argument(
        "--driver-yolo-interval",
        type=int,
        default=1,
        help=(
            "운전자 YOLO를 실행할 원본 프레임 간격 (기본: 1). "
            "중간 프레임은 마지막 운전자 box를 재사용합니다."
        ),
    )
    parser.add_argument(
        "--driver-min-x-ratio",
        type=float,
        default=0.45,
        help="운전자 후보를 허용할 최소 화면 x 중심 비율 (기본: 0.45)",
    )
    parser.add_argument(
        "--fallback-driver-roi",
        type=float,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=DEFAULT_FALLBACK_DRIVER_ROI,
        help=(
            "YOLO 박스가 없을 때 TCN에 사용할 정규화 운전자 ROI. "
            "기본값은 기존 학습 ROI를 현재 해상도에 맞게 환산합니다."
        ),
    )
    parser.add_argument(
        "--driver-box-hold-sec",
        type=float,
        default=2.0,
        help=(
            "YOLO가 일시적으로 운전자를 놓쳤을 때 마지막 유효 운전자 박스를 "
            "TCN 입력에 유지할 최대 시간(초). 그 뒤에는 0 특징을 넣지 않고 "
            "TCN을 다시 워밍업합니다."
        ),
    )
    parser.add_argument(
        "--draw-driver-box",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "실시간 미리보기에서 YOLO가 선택한 운전자 박스를 표시합니다 "
            "(기본: 활성화; TCN/Qwen 입력 원본에는 그리지 않음)"
        ),
    )
    parser.add_argument("--driver-anchor-x", type=int, default=1260)
    parser.add_argument("--driver-anchor-y", type=int, default=550)
    parser.add_argument(
        "--video-start-sec", type=float, default=0.0,
        help="전체 영상 분석 시작 시각(초)",
    )
    parser.add_argument(
        "--video-max-sec", type=float, default=None,
        help="전체 영상에서 분석할 최대 길이(초)",
    )
    parser.add_argument(
        "--tcn-interval-sec", type=float, default=0.2,
        help="TCN rolling-window 평가 주기(초)",
    )
    parser.add_argument(
        "--event-threshold", type=float, default=None,
        help="Qwen 이벤트 호출용 TCN 확률 임계값 (기본: 선택한 ensemble의 검증 threshold)",
    )
    parser.add_argument(
        "--fall-onset-event-threshold", type=float, default=None,
        help="fall-onset 보조 TCN의 VLM 호출 임계값 (기본: onset ensemble 검증 threshold)",
    )
    parser.add_argument(
        "--fall-onset-trigger-persist-sec", type=float, default=0.2,
        help="fall-onset 보조 TCN이 VLM을 호출하기 위해 유지되어야 하는 시간(초)",
    )
    parser.add_argument(
        "--fall-onset-recheck-sec", type=float, default=1.0,
        help="fall-onset 후보가 지속될 때 VLM 재확인 최소 간격(초)",
    )
    parser.add_argument(
        "--trigger-persist-sec", type=float, default=0.4,
        help="Qwen 이벤트를 시작하기 위해 threshold 이상이어야 하는 지속 시간(초)",
    )
    parser.add_argument(
        "--clear-persist-sec", type=float, default=0.6,
        help="이벤트를 해제하기 위해 threshold 미만이어야 하는 지속 시간(초)",
    )
    parser.add_argument(
        "--qwen-recheck-sec", type=float, default=1.0,
        help="후보가 계속될 때 같은 이벤트에서 Qwen 재검증 간격(초)",
    )
    parser.add_argument(
        "--visual-watchdog-sec", type=float, default=0.0,
        help=(
            "TCN 확률이 낮아도 Qwen이 놓친 붕괴를 확인하는 시각 안전 감시 간격(초; "
            "0이면 비활성화, 기본 비활성화)"
        ),
    )
    parser.add_argument(
        "--eye-uncertain-watchdog-sec", type=float, default=1.0,
        help=(
            "눈 랜드마크가 지속적으로 불신 가능하고 TCN이 임계값 근처일 때 "
            "VLM 재확인을 허용하는 최소 간격. 0이면 비활성화."
        ),
    )
    parser.add_argument(
        "--eye-uncertain-window-sec", type=float, default=0.8,
        help="눈 랜드마크 불확실성이 지속되어야 하는 시간(초)",
    )
    parser.add_argument(
        "--eye-uncertain-tcn-ratio", type=float, default=0.65,
        help=(
            "눈 불확실 감시를 시작할 TCN 하한의 event threshold 대비 비율. "
            "기본 0.65는 임계값 0.31일 때 P>=0.202입니다."
        ),
    )
    parser.add_argument(
        "--eye-state-model", type=Path, default=None,
        help=(
            "선택 사항: open/closed/unknown 눈 상태 specialist checkpoint. 지정 시 "
            "주 TCN과 보수적으로 결합한 위험도를 계산합니다."
        ),
    )
    parser.add_argument(
        "--eye-state-smoothing-sec", type=float, default=1.0,
        help="눈 상태 specialist 확률을 causal 평균할 시간(초, 기본 1초)",
    )
    parser.add_argument(
        "--eye-closed-probability-threshold", type=float, default=0.70,
        help="눈 specialist가 폐안 후보로 간주할 최소 확률(기본 0.70)",
    )
    parser.add_argument(
        "--eye-closed-persist-sec", type=float, default=0.6,
        help="눈 확률이 유지되어야 눈 위험도를 더하는 최소 시간(초, 기본 0.6)",
    )
    parser.add_argument(
        "--eye-min-visible-fraction", type=float, default=0.60,
        help="최근 평활 구간에서 눈 crop이 유효해야 하는 최소 비율(0~1, 기본 0.60)",
    )
    parser.add_argument(
        "--eye-risk-weight", type=float, default=0.45,
        help="확실한 장기 폐안이 주 TCN 위험도에 기여하는 최대 가중치(0~1)",
    )
    parser.add_argument(
        "--combined-risk-threshold", type=float, default=None,
        help=(
            "눈 specialist 결합 위험도 임계값. 미지정 시 --event-threshold를 사용하며, "
            "운영 전 별도 검증 구간에서 보정해야 합니다."
        ),
    )
    parser.add_argument(
        "--context-seconds", type=float, default=0.0,
        help="Qwen 이벤트 영상에 추가할 후보 직전 컨텍스트 길이(초, 기본 0)",
    )
    parser.add_argument(
        "--candidate-seconds", type=float, default=1.5,
        help="Qwen 이벤트 영상에 포함할 최근 TCN 후보 구간 길이(초, 기본 1.5)",
    )
    parser.add_argument(
        "--event-fps", type=float, default=4.0,
        help=(
            "브리지로 전송하는 짧은 이벤트 증거 영상 FPS (기본 4). "
            "--event-full-source-fps 사용 시 MP4 경로에서는 무시됩니다."
        ),
    )
    parser.add_argument(
        "--event-full-source-fps",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "이벤트 MP4를 시점 샘플링하지 않고 원본 FPS의 연속 프레임 전체로 "
            "구성합니다 (기본: 비활성화)."
        ),
    )
    parser.add_argument(
        "--qwen-event-max-width", type=int, default=768,
        help=(
            "Qwen 이벤트 MP4의 최대 가로 해상도 (기본 768). "
            "0이면 원본 body-crop 해상도를 유지합니다."
        ),
    )
    parser.add_argument(
        "--event-media-format", choices=("storyboard", "video"),
        default="video",
        help=(
            "VLM 이벤트 증거 형식. video는 고정 운전자 박스의 시간 순서 MP4이며 "
            "기본값입니다. storyboard는 4패널 JPEG입니다."
        ),
    )
    parser.add_argument(
        "--event-storyboard-panel-size", type=int, default=256,
        help=(
            "VLM용 정사각 스토리보드 패널 한 변 픽셀 (기본 256). "
            "4패널 전체는 약 520x520이며 360보다 이미지 토큰을 약 절반으로 줄입니다."
        ),
    )
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="입력 MP4를 실제 시간 속도로 재생합니다 (기본: 활성화)",
    )
    parser.add_argument(
        "--video-preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="전체 입력 영상을 즉시 표시하고 TCN/Qwen 상태를 오버레이합니다",
    )
    parser.add_argument(
        "--event-output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "realtime_events",
        help="이벤트 결과 JSONL과 임시 증거 영상 경로",
    )
    parser.add_argument(
        "--keep-event-media",
        action="store_true",
        help="전송 후에도 생성한 이벤트 MP4/JPEG를 보관합니다",
    )
    parser.add_argument(
        "--tcn-timeline-file", type=Path, default=None,
        help=(
            "모든 TCN 추론의 확률·seed별 점수·눈 품질을 저장할 CSV. "
            "미지정 시 이벤트 디렉터리에 실행별 CSV를 만듭니다."
        ),
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


def load_tcn_evidence(
    score_dir: Path | None,
    media_path: Path,
) -> dict[str, Any] | None:
    """Load the stable TCN candidate summary for one evidence media file.

    The TCN is temporal evidence only. Its result is deliberately kept apart
    from the existing driver_response state machine in the bridge.
    """
    if score_dir is None:
        return None
    summary_path = score_dir / f"{media_path.stem}.summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"TCN summary가 없습니다: {summary_path}. "
            "전처리를 다시 실행하거나 --tcn-score-dir를 생략하세요."
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"TCN summary JSON이 올바르지 않습니다: {summary_path}") from exc
    if not isinstance(summary, dict):
        raise ValueError(f"TCN summary는 JSON object여야 합니다: {summary_path}")

    peak = summary.get("peak")
    events = summary.get("events", [])
    if peak is not None and not isinstance(peak, dict):
        raise ValueError(f"TCN peak 형식이 올바르지 않습니다: {summary_path}")
    if not isinstance(events, list):
        raise ValueError(f"TCN events 형식이 올바르지 않습니다: {summary_path}")

    # Send only the stable, bounded cross-process contract rather than raw
    # per-window/per-seed records. This keeps /monitor requests compact.
    evidence: dict[str, Any] = {
        "schema_version": int(summary.get("schema_version", 1)),
        "source_media": media_path.name,
        "threshold": summary.get("threshold"),
        "window_count": summary.get("window_count"),
        "candidate_count": summary.get("candidate_count"),
        "candidate_detected": summary.get("candidate_detected", False),
        "event_count": summary.get("event_count", len(events)),
        "events": events[:64],
        "peak": peak,
    }
    return evidence


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
    tcn_evidence = load_tcn_evidence(args.tcn_score_dir, image_path)
    if tcn_evidence is not None:
        payload["tcn_evidence"] = tcn_evidence
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
    continuous_source_frames: bool = False,
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
        if continuous_source_frames:
            source_fps = float(capture.get(cv2.CAP_PROP_FPS))
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if source_fps <= 0.0 or total_frames <= 0:
                raise RuntimeError("이벤트 원본 영상의 FPS/프레임 수를 읽지 못했습니다.")
            full_frame_count = max(2, int(round(seconds * source_fps)))
            end_frame = min(total_frames - 1, int(round(end_time * source_fps)))
            start_frame = max(0, end_frame - full_frame_count + 1)
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            selected: list[tuple[float, Any]] = []
            for frame_index in range(start_frame, end_frame + 1):
                ok, frame = capture.read()
                timestamp = frame_index / source_fps
                if not ok:
                    raise RuntimeError(
                        f"이벤트 연속 프레임을 읽지 못했습니다: {timestamp:.3f}s"
                    )
                crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                if crop.size == 0:
                    raise RuntimeError("고정 운전자 crop이 비어 있습니다.")
                selected.append((timestamp, crop))
            return selected
        else:
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


def build_tcn_event_payload(
    args: argparse.Namespace,
    source_video: Path,
    event_media: Path,
    observation_id: int,
    video_time: float,
    probability: float,
    threshold: float,
    window_count: int,
    trigger_window_count: int,
    candidate_start_sec: float,
    candidate_detected: bool,
    temporal_context: dict[str, Any],
    headers: dict[str, str],
) -> dict[str, Any]:
    """Create bounded TCN context for a candidate or visual safety-watch call."""
    telemetry: dict[str, Any] | None = None
    telemetry_age_s: float | None = None
    if not args.no_inline_telemetry and not args.vlm_only:
        telemetry, telemetry_age_s = latest_telemetry(args, headers)
    recent_probabilities = [
        round(float(value), 6)
        for value in temporal_context.get("recent_probabilities", [])
    ]
    peak_probability = float(
        temporal_context.get("peak_probability", probability)
    )
    peak_timestamp_sec = float(
        temporal_context.get("peak_timestamp_sec", video_time)
    )
    event = {
        "start_sec": round(max(0.0, candidate_start_sec), 3),
        "end_sec": round(video_time, 3),
        "duration_sec": round(max(0.001, video_time - candidate_start_sec), 3),
        "peak_probability": round(peak_probability, 6),
        "peak_timestamp_sec": round(peak_timestamp_sec, 3),
        "trigger_window_count": int(trigger_window_count),
    }
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "observation_id": observation_id,
        "captured_at_unix_s": time.time(),
        "event_mode": True,
        "cabin_media": {
            "mime_type": MIME_TYPES[event_media.suffix.lower()],
            "data_base64": base64.b64encode(event_media.read_bytes()).decode("ascii"),
        },
        "tcn_evidence": {
            "schema_version": 1,
            "source_media": source_video.name,
            "threshold": round(threshold, 6),
            "window_count": int(window_count),
            "candidate_count": int(trigger_window_count) if candidate_detected else 0,
            "candidate_detected": candidate_detected,
            "trigger_source": str(temporal_context["trigger_source"]),
            "current_probability": round(probability, 6),
            "risk_trend": str(temporal_context["risk_trend"]),
            "recent_probabilities": recent_probabilities,
            "above_threshold_duration_sec": round(
                float(temporal_context["above_threshold_duration_sec"]), 3
            ),
            "event_count": 1 if candidate_detected else 0,
            "events": [event] if candidate_detected else [],
            "peak": {
                "tcn_probability": round(peak_probability, 6),
                "timestamp_sec": round(peak_timestamp_sec, 3),
            },
        },
    }
    state_probabilities = temporal_context.get("state_probabilities")
    if isinstance(state_probabilities, dict):
        payload["tcn_evidence"]["state_probabilities"] = {
            str(name): round(float(value), 6)
            for name, value in state_probabilities.items()
        }
    transition_probability = temporal_context.get("transition_probability")
    if isinstance(transition_probability, (int, float)):
        payload["tcn_evidence"]["transition_probability"] = round(
            float(transition_probability), 6
        )
        payload["tcn_evidence"]["transition_horizon_sec"] = round(
            float(temporal_context.get("transition_horizon_sec", 0.0)), 3
        )
    # Keep the VLM contract compact but include the evidence needed to judge
    # whether a temporal score is trustworthy.  These are measurements and
    # policies, never visual labels or an instruction to override the video.
    for source_name, payload_name in (
        ("main_tcn", "main_tcn"),
        ("eye_evidence", "eye_evidence"),
        ("control_tcn", "control_tcn"),
    ):
        evidence = temporal_context.get(source_name)
        if isinstance(evidence, dict):
            payload["tcn_evidence"][payload_name] = evidence
    if telemetry is not None:
        payload["telemetry"] = telemetry
        payload["driving_summary"] = driving_summary(telemetry)
    if telemetry_age_s is not None:
        payload["telemetry_age_s"] = telemetry_age_s
    return payload


def post_tcn_event(
    args: argparse.Namespace,
    headers: dict[str, str],
    source_video: Path,
    event_media: Path,
    observation_id: int,
    video_time: float,
    probability: float,
    threshold: float,
    window_count: int,
    trigger_window_count: int,
    candidate_start_sec: float,
    candidate_detected: bool,
    temporal_context: dict[str, Any],
) -> dict[str, Any]:
    payload = build_tcn_event_payload(
        args, source_video, event_media, observation_id, video_time, probability,
        threshold, window_count, trigger_window_count, candidate_start_sec,
        candidate_detected, temporal_context, headers,
    )
    return request_json(
        "POST", args.bridge_url.rstrip("/") + "/monitor", headers, payload,
        args.bridge_timeout,
    )


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
    """Stream one complete MP4: TCN on every rolling window, Qwen on events only."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("--video-path 모드에는 opencv-python이 필요합니다.") from exc

    source_video = args.video_path.resolve()
    config_path = (
        args.tcn_config.resolve()
        if args.tcn_config is not None
        else DEFAULT_TCN_CONFIG
    )
    ensemble_path = (
        args.tcn_ensemble_config.resolve()
        if args.tcn_ensemble_config is not None
        else DEFAULT_TCN_ENSEMBLE
    )
    if not source_video.is_file():
        raise FileNotFoundError(f"입력 영상이 없습니다: {source_video}")
    if not config_path.is_file() or not ensemble_path.is_file():
        raise FileNotFoundError("TCN config 또는 ensemble JSON이 없습니다.")
    onset_paths_given = (
        args.fall_onset_tcn_config is not None,
        args.fall_onset_tcn_ensemble_config is not None,
    )
    if onset_paths_given[0] != onset_paths_given[1]:
        raise ValueError(
            "fall-onset 보조 TCN은 --fall-onset-tcn-config와 "
            "--fall-onset-tcn-ensemble-config를 함께 지정해야 합니다."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    target_fps = float(config["target_fps"])
    if not math.isclose(target_fps, 10.0, abs_tol=1e-6):
        raise RuntimeError(f"현재 TCN은 10 Hz 입력만 지원합니다: {target_fps}")

    window_steps = int(round(float(config.get("window_sec", 10.0)) * target_fps))
    if window_steps < 2:
        raise RuntimeError(f"TCN 입력 window가 너무 짧습니다: {window_steps} frames")
    runtime = StreamingTCN(ensemble_path)
    if runtime.window_steps != window_steps:
        raise RuntimeError(
            "TCN config와 ensemble의 입력 길이가 다릅니다: "
            f"config={window_steps}, ensemble={runtime.window_steps}"
        )
    control_ensemble_path = resolve_control_ensemble_path(ensemble_path, json.loads(ensemble_path.read_text(encoding="utf-8")))
    control_runtime: StreamingControlTCN | None = None
    if control_ensemble_path is not None:
        if not control_ensemble_path.is_file():
            raise FileNotFoundError(f"제어 관여 ensemble JSON이 없습니다: {control_ensemble_path}")
        control_runtime = StreamingControlTCN(control_ensemble_path)
        if control_runtime.window_steps != window_steps:
            raise RuntimeError(
                "주 TCN과 제어 TCN의 입력 길이가 다릅니다: "
                f"main={window_steps}, control={control_runtime.window_steps}"
            )
    event_threshold = (
        float(args.event_threshold)
        if args.event_threshold is not None
        else runtime.threshold
    )
    combined_risk_threshold = (
        float(args.combined_risk_threshold)
        if args.combined_risk_threshold is not None
        else event_threshold
    )
    onset_runtime: StreamingTCN | None = None
    if args.fast_onset_risk:
        if not all(onset_paths_given):
            raise ValueError("--fast-onset-risk에는 보조 onset TCN 설정과 ensemble이 필요합니다.")
        if not combined_risk_threshold <= args.fast_onset_risk_floor <= 1.0:
            raise ValueError("fast onset risk floor는 combined threshold 이상, 1 이하이어야 합니다.")
    onset_window_steps = 0
    onset_threshold: float | None = None
    if onset_paths_given[0]:
        onset_config_path = args.fall_onset_tcn_config.resolve()
        onset_ensemble_path = args.fall_onset_tcn_ensemble_config.resolve()
        if not onset_config_path.is_file() or not onset_ensemble_path.is_file():
            raise FileNotFoundError("fall-onset TCN config 또는 ensemble JSON이 없습니다.")
        onset_config = json.loads(onset_config_path.read_text(encoding="utf-8"))
        onset_fps = float(onset_config["target_fps"])
        if not math.isclose(onset_fps, target_fps, abs_tol=1e-6):
            raise RuntimeError(
                "주 TCN과 fall-onset TCN의 target_fps가 다릅니다: "
                f"main={target_fps}, onset={onset_fps}"
            )
        onset_window_steps = int(
            round(float(onset_config.get("window_sec", 1.0)) * target_fps)
        )
        onset_runtime = StreamingTCN(onset_ensemble_path)
        if onset_runtime.window_steps != onset_window_steps:
            raise RuntimeError(
                "fall-onset TCN config와 ensemble의 입력 길이가 다릅니다: "
                f"config={onset_window_steps}, ensemble={onset_runtime.window_steps}"
            )
        onset_threshold = (
            float(args.fall_onset_event_threshold)
            if args.fall_onset_event_threshold is not None
            else onset_runtime.threshold
        )
    eye_state_smooth_steps = max(1, int(round(args.eye_state_smoothing_sec * target_fps)))
    eye_closed_persist_steps = max(1, int(round(args.eye_closed_persist_sec * target_fps)))
    extractor = StreamingFeatureExtractor(
        runtime,
        target_fps,
        eye_state_model_path=(args.eye_state_model.resolve() if args.eye_state_model else None),
        eye_state_smooth_steps=eye_state_smooth_steps,
        eye_closed_persist_steps=eye_closed_persist_steps,
        eye_closed_probability_threshold=args.eye_closed_probability_threshold,
        eye_min_visible_fraction=args.eye_min_visible_fraction,
    )
    selector_module = load_runtime_module("driver_selector_runtime", FEATURES_DIR / "driver_selector.py")
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        extractor.close()
        raise RuntimeError(f"영상을 열 수 없습니다: {source_video}")
    src_fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if src_fps <= 0.0:
        cap.release()
        extractor.close()
        raise RuntimeError("입력 영상 FPS를 읽을 수 없습니다.")
    duration = total_frames / src_fps
    start_sec = max(0.0, float(args.video_start_sec))
    end_sec = duration if args.video_max_sec is None else min(duration, start_sec + args.video_max_sec)
    if start_sec >= end_sec:
        cap.release()
        extractor.close()
        raise ValueError("--video-start-sec/--video-max-sec 범위가 올바르지 않습니다.")
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
    frame_index = int(round(start_sec * src_fps))
    selector = selector_module.DriverSelector(
        model_path=str(args.driver_yolo_model), tracker=args.driver_tracker,
        anchor_x=args.driver_anchor_x, anchor_y=args.driver_anchor_y, fps=src_fps,
        confidence=args.driver_yolo_confidence,
        min_driver_x_ratio=args.driver_min_x_ratio,
    )
    # VLM and TCN share the same selected-driver body box. This makes the
    # evidence exactly match the box drawn in the preview and retains head,
    # torso, hand, and nearby wheel context without reacquiring the passenger
    # through a separate face detector.
    last_tcn_driver_box: tuple[int, int, int, int] | None = None
    last_tcn_driver_box_time = -float("inf")
    last_driver_info: dict[str, Any] | None = None
    driver_yolo_frame_count = 0

    event_dir = (args.event_output_dir / source_video.stem).resolve()
    media_dir = event_dir / "media"
    results_path = event_dir / "events.jsonl"
    event_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = f"{int(time.time())}_{int(round(start_sec * 1000)):010d}"
    timeline_path = (
        args.tcn_timeline_file.resolve()
        if args.tcn_timeline_file is not None
        else event_dir / f"tcn_timeline_{run_stamp}.csv"
    )
    timeline_path.parent.mkdir(parents=True, exist_ok=True)
    timeline_stream = timeline_path.open("w", newline="", encoding="utf-8")
    timeline_writer = csv.DictWriter(
        timeline_stream,
        fieldnames=[
            "timestamp", "tcn_probability", "tcn_threshold",
            "tcn_direct_state", "tcn_raw_argmax_state", "tcn_observable",
            "tcn_observation_quality", "tcn_active_probability",
            "tcn_reduced_probability", "tcn_nvr_probability",
            "tcn_nvr_candidate", "tcn_inference_ms",
            "control_engaged_probability", "control_disengaged_probability",
            "control_observable", "control_observation_quality",
            "control_disengaged_candidate", "control_reduced_confirmed",
            "control_confirmation_windows", "control_operational",
            "control_tcn_inference_ms",
            "tcn_transition_probability", "tcn_transition_threshold",
            "tcn_transition_horizon_sec", "tcn_transition_operational",
            "combined_risk", "combined_risk_threshold",
            "base_combined_risk", "fast_onset_active", "fast_onset_motion",
            "fast_onset_down_speed", "fast_onset_down_acceleration",
            "eye_measurement_reliable", "eye_geometry_reliable", "eye_unreliable_fraction",
            "ear_left", "ear_right", "ear_mean", "eye_valid",
            "face_valid", "face_reliability",
            "eye_blink_left", "eye_blink_right", "eye_blink_mean",
            "eye_blendshape_valid", "seat_head_dx", "seat_head_dy",
            "eye_open_probability", "eye_closed_probability",
            "eye_unknown_probability", "eye_state_valid",
            "eye_visible_fraction", "eye_closed_sustained",
            "eye_crop_contrast", "eye_crop_sharpness", "eye_crop_quality_valid",
        ]
        + [f"seed{member['seed']}" for member in runtime.members]
        + (
            ["fall_onset_probability", "fall_onset_threshold",
             "fall_onset_tcn_inference_ms"]
            + [f"fall_onset_seed{member['seed']}" for member in onset_runtime.members]
            if onset_runtime is not None else []
        ),
    )
    timeline_writer.writeheader()
    feature_buffer: deque[Any] = deque(maxlen=window_steps)
    control_feature_buffer: deque[Any] = deque(maxlen=window_steps)
    onset_feature_buffer: deque[Any] = deque(maxlen=onset_window_steps) if onset_runtime else deque()
    eye_quality_samples = max(
        1, int(math.ceil(args.eye_uncertain_window_sec * target_fps))
    )
    eye_unreliable_buffer: deque[int] = deque(maxlen=eye_quality_samples)
    buffer_seconds = max(10.0, args.context_seconds + args.candidate_seconds + 1.0)
    visual_buffer: deque[tuple[float, Any]] = deque(
        maxlen=int(math.ceil(buffer_seconds * target_fps))
    )
    driver_box_buffer: deque[tuple[float, tuple[int, int, int, int]]] = deque(
        maxlen=int(math.ceil(buffer_seconds * target_fps))
    )
    interval_samples = max(1, int(round(args.tcn_interval_sec * target_fps)))
    trigger_ticks = max(1, int(math.ceil(args.trigger_persist_sec / args.tcn_interval_sec)))
    clear_ticks = max(1, int(math.ceil(args.clear_persist_sec / args.tcn_interval_sec)))
    onset_trigger_ticks = max(
        1, int(math.ceil(args.fall_onset_trigger_persist_sec / args.tcn_interval_sec))
    )
    next_sample_time, sample_count = start_sec, 0
    trigger_streak = clear_streak = 0
    onset_trigger_streak = 0
    event_trigger_window_count = 0
    event_active, event_id, observation_id = False, 0, args.start_observation_id
    last_submit_time = -float("inf")
    last_onset_submit_time = -float("inf")
    last_watchdog_submit_time = -float("inf")
    last_eye_watchdog_submit_time = -float("inf")
    probability_history_steps = max(
        2, int(math.ceil(1.0 / args.tcn_interval_sec)) + 1
    )
    tcn_probability_history: deque[tuple[float, float]] = deque(
        maxlen=probability_history_steps
    )
    combined_risk_history: deque[tuple[float, float]] = deque(
        maxlen=probability_history_steps
    )
    control_probability_history: deque[tuple[float, float]] = deque(
        maxlen=probability_history_steps
    )
    onset_probability_history: deque[tuple[float, float]] = deque(
        maxlen=probability_history_steps
    )
    event_records_written = 0
    probability: float | None = None
    combined_risk: float | None = None
    onset_probability: float | None = None
    latest_eye_diagnostics: dict[str, Any] = {
        "eye_measurement_reliable": False,
        "eye_geometry_reliable": 0,
        "ear_left": 0.0,
        "ear_right": 0.0,
        "ear_mean": 0.0,
        "eye_valid": 0,
        "face_valid": 0,
        "face_reliability": 0.0,
        "eye_blink_left": 0.0,
        "eye_blink_right": 0.0,
        "eye_blink_mean": 0.0,
        "eye_blendshape_valid": 0,
        "seat_head_dx": 0.0,
        "seat_head_dy": 0.0,
        "eye_open_probability": 0.0,
        "eye_closed_probability": 0.0,
        "eye_unknown_probability": 1.0,
        "eye_state_valid": 0,
        "eye_visible_fraction": 0.0,
        "eye_closed_sustained": 0,
        "eye_crop_contrast": 0.0,
        "eye_crop_sharpness": 0.0,
        "eye_crop_quality_valid": 0,
    }
    last_response, qwen_status = "waiting_for_tcn_event", "IDLE"
    tcn_ready_logged = False
    last_no_driver_warning_time = -float("inf")
    pending: Future[dict[str, Any]] | None = None
    pending_media: Path | None = None
    pending_metadata: dict[str, Any] | None = None
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tcn-qwen-event")
    wall_start = time.perf_counter()

    print(
        f"전체 영상 실시간 분석 시작: {source_video.name} ({duration:.1f}s), "
        f"TCN window={window_steps / target_fps:.1f}s/{window_steps}f, "
        f"event threshold={event_threshold:.3f} "
        f"(학습 threshold={runtime.threshold:.3f})", flush=True,
    )
    if extractor.eye_state is not None:
        print(
            "눈 상태 specialist 활성화: "
            f"weight={args.eye_risk_weight:.2f}, risk threshold={combined_risk_threshold:.3f}, "
            f"smoothing={args.eye_state_smoothing_sec:.1f}s, "
            f"closed>={args.eye_closed_probability_threshold:.2f} for "
            f"{args.eye_closed_persist_sec:.1f}s, visible>="
            f"{args.eye_min_visible_fraction:.0%}",
            flush=True,
        )
    if onset_runtime is not None and onset_threshold is not None:
        print(
            "fall-onset 보조 TCN 활성화: "
            f"window={onset_window_steps / target_fps:.1f}s/{onset_window_steps}f, "
            f"VLM trigger threshold={onset_threshold:.3f}",
            flush=True,
        )
    try:
        while True:
            if pending is not None and pending.done():
                try:
                    bridge_result = pending.result()
                    if pending_metadata is None:
                        raise RuntimeError("Qwen 이벤트 메타데이터가 없습니다.")
                    last_response = print_qwen_event_result(
                        bridge_result, pending_metadata, args.print_json
                    )
                    preview.record_vlm_response(
                        last_response,
                        time.monotonic() - pending_metadata["submitted_monotonic"],
                    )
                    qwen_status = "DONE"
                    record = {
                        "source_video": str(source_video),
                        "event_id": pending_metadata["event_id"],
                        "video_time": pending_metadata["video_time"],
                        "event_video_start_sec": pending_metadata["event_video_start_sec"],
                        "tcn_probability": pending_metadata["tcn_probability"],
                        "fall_onset_probability": pending_metadata.get("fall_onset_probability"),
                        "eye_unreliable_fraction": pending_metadata.get(
                            "eye_unreliable_fraction"
                        ),
                        "trigger": pending_metadata.get("trigger"),
                        "tcn_threshold": pending_metadata["tcn_threshold"],
                        "qwen_model_inference_ms": bridge_result.get("inference_ms"),
                        "event_end_to_end_sec": round(
                            time.monotonic() - pending_metadata["submitted_monotonic"], 4
                        ),
                        "bridge_result": bridge_result,
                    }
                    with results_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    event_records_written += 1
                except Exception as exc:
                    qwen_status = "ERROR"
                    preview.finish_vlm_inference()
                    print(f"Qwen 이벤트 실패: {exc}", file=sys.stderr, flush=True)
                finally:
                    if pending_media is not None and not args.keep_event_media:
                        pending_media.unlink(missing_ok=True)
                    pending, pending_media, pending_metadata = None, None, None

            ok, frame = cap.read()
            if not ok:
                break
            frame_time = frame_index / src_fps
            frame_index += 1
            if frame_time > end_sec + 1e-9:
                break
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
                # The cabin camera is fixed and the TCN samples at 10 Hz, so
                # reuse the most recent box between detector updates.
                driver_info = {**last_driver_info, "held": True}
            driver_box = None
            driver_box_source = "missing"
            if driver_info is not None:
                driver_box = selector_module.expand_driver_box(driver_info["box"], frame.shape)
            if driver_box is not None:
                last_tcn_driver_box = driver_box
                last_tcn_driver_box_time = frame_time
                driver_box_source = "held" if driver_info.get("held", False) else "tracked"
            elif (
                last_tcn_driver_box is not None
                and frame_time - last_tcn_driver_box_time <= args.driver_box_hold_sec
            ):
                # A short detector dropout must not turn the temporal input
                # into an all-zero sequence.  The image inside the held box
                # still changes as the driver moves, so features remain live.
                driver_box = last_tcn_driver_box
                driver_box_source = "held"

            if driver_box is None:
                # Never feed a blank frame to the TCN. The static driver-side
                # ROI matches the region used to build the training features,
                # keeping temporal inference live while YOLO reacquires.
                driver_box = normalized_roi_to_box(
                    args.fallback_driver_roi,
                    frame.shape,
                )
                if driver_box is not None:
                    driver_box_source = "fallback_roi"

            if driver_box is not None and qwen_status == "NO_DRIVER":
                qwen_status = "IDLE"

            # Display the input video independently of TCN/Qwen latency.
            display_frame = frame.copy()
            # Give Qwen the selected-driver location in the wide context panel,
            # but keep its crop and all TCN/MediaPipe inputs free of overlays.
            qwen_context_frame = frame
            track_label = (
                "driver: YOLO tracked"
                if driver_box_source == "tracked"
                else "driver: holding last box"
                if driver_box_source == "held"
                else "driver: fallback ROI"
                if driver_box_source == "fallback_roi"
                else "driver: not tracked"
            )
            if driver_box is not None:
                x1, y1, x2, y2 = driver_box
                if args.draw_driver_box:
                    box_is_fallback = driver_box_source == "fallback_roi"
                    box_color = (0, 180, 255) if box_is_fallback else (40, 220, 40)
                    box_label = "DRIVER (FALLBACK ROI)" if box_is_fallback else "DRIVER (YOLO)"
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 3)
                    cv2.putText(
                        display_frame, box_label, (x1, max(24, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2, cv2.LINE_AA,
                    )
                    qwen_context_frame = frame.copy()
                    cv2.rectangle(qwen_context_frame, (x1, y1), (x2, y2), box_color, 2)

            # Use the source-video timestamp (not wall-clock time), so this
            # remains meaningful if inference temporarily runs slower than the
            # video's native frame rate.
            video_time_label = f"Video time: {frame_time:.1f}s"
            text_size, text_baseline = cv2.getTextSize(
                video_time_label, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2
            )
            label_left, label_top = 8, 8
            cv2.rectangle(
                display_frame,
                (label_left - 4, label_top - 4),
                (label_left + text_size[0] + 4, label_top + text_size[1] + text_baseline + 4),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                display_frame, video_time_label,
                (label_left, label_top + text_size[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA,
            )
            if probability is None:
                status_probability = (
                    "waiting for driver"
                    if driver_box is None
                    else f"warming up ({len(feature_buffer)}/{window_steps})"
                )
            else:
                status_probability = (
                    f"risk {combined_risk:.3f} / {combined_risk_threshold:.3f} "
                    f"(TCN {probability:.3f})"
                    if extractor.eye_state is not None and combined_risk is not None
                    else f"{probability:.3f} / {event_threshold:.3f}"
                )
            if preview.show_frame(display_frame, "", []):
                break

            if frame_time + 1e-9 >= next_sample_time:
                next_sample_time += 1.0 / target_fps
                if driver_box is None:
                    # Feeding a blank crop yields a deterministic model bias
                    # (about 0.527 for the previous ensemble), which looks
                    # like a frozen TCN score.  A missing driver is invalid
                    # temporal input, not a driver state.
                    if feature_buffer:
                        print(
                            f"운전자 추적이 {args.driver_box_hold_sec:.1f}s 이상 끊겨 "
                            "TCN 버퍼를 초기화합니다.",
                            flush=True,
                        )
                    elif frame_time - last_no_driver_warning_time >= 2.0:
                        print(
                            "운전자 박스를 찾지 못해 TCN 입력을 수집하지 않습니다. "
                            "운전자 추적이 재개되면 5초 뒤 자동으로 추론합니다.",
                            flush=True,
                        )
                        last_no_driver_warning_time = frame_time
                    feature_buffer.clear()
                    control_feature_buffer.clear()
                    onset_feature_buffer.clear()
                    tcn_probability_history.clear()
                    combined_risk_history.clear()
                    control_probability_history.clear()
                    onset_probability_history.clear()
                    eye_unreliable_buffer.clear()
                    probability = None
                    combined_risk = None
                    onset_probability = None
                    tcn_ready_logged = False
                    trigger_streak = clear_streak = 0
                    onset_trigger_streak = 0
                    event_active = False
                    event_trigger_window_count = 0
                    qwen_status = "NO_DRIVER" if pending is None else qwen_status
                    if args.realtime:
                        target_elapsed = frame_time - start_sec
                        sleep_seconds = target_elapsed - (time.perf_counter() - wall_start)
                        if sleep_seconds > 0:
                            preview.wait(sleep_seconds)
                    continue
                else:
                    x1, y1, x2, y2 = driver_box
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        feature_buffer.clear()
                        control_feature_buffer.clear()
                        onset_feature_buffer.clear()
                        tcn_probability_history.clear()
                        combined_risk_history.clear()
                        control_probability_history.clear()
                        onset_probability_history.clear()
                        eye_unreliable_buffer.clear()
                        probability = None
                        combined_risk = None
                        onset_probability = None
                        tcn_ready_logged = False
                        trigger_streak = clear_streak = 0
                        onset_trigger_streak = 0
                        event_active = False
                        event_trigger_window_count = 0
                        qwen_status = "NO_DRIVER" if pending is None else qwen_status
                        continue
                feature_vector, latest_eye_diagnostics = extractor.extract(
                    crop, driver_box, frame.shape
                )
                feature_buffer.append(feature_vector)
                if control_runtime is not None:
                    if extractor.latest_control_vector is None:
                        raise RuntimeError("제어 TCN feature가 추출되지 않았습니다.")
                    control_feature_buffer.append(extractor.latest_control_vector)
                if onset_runtime is not None:
                    onset_feature_buffer.append(feature_vector)
                eye_unreliable_buffer.append(
                    int(not latest_eye_diagnostics["eye_measurement_reliable"])
                )
                # Keep a cheap live fallback. The normal event path below
                # rereads the source using one fixed pre-event driver box.
                visual_buffer.append((frame_time, crop.copy()))
                driver_box_buffer.append((frame_time, driver_box))
                sample_count += 1
                should_run_tcn = (
                    len(feature_buffer) == window_steps
                    and (sample_count - window_steps) % interval_samples == 0
                )
                if should_run_tcn:
                    tcn_started = time.perf_counter()
                    probability, seed_probabilities = runtime.predict(
                        runtime.np.stack(feature_buffer)
                    )
                    tcn_inference_ms = (time.perf_counter() - tcn_started) * 1000.0
                    control_tcn_inference_ms: float | None = None
                    if (
                        control_runtime is not None
                        and len(control_feature_buffer) == control_runtime.window_steps
                    ):
                        control_tcn_started = time.perf_counter()
                        control_runtime.predict(runtime.np.stack(control_feature_buffer))
                        control_tcn_inference_ms = (
                            time.perf_counter() - control_tcn_started
                        ) * 1000.0
                        runtime.apply_control_reduced(
                            control_runtime, control_runtime.confirmation_windows
                        )
                        control_disengaged = (control_runtime.last_probabilities or {}).get(
                            "control_disengaged"
                        )
                        if isinstance(control_disengaged, (int, float)):
                            control_probability_history.append(
                                (frame_time, float(control_disengaged))
                            )
                    tcn_probability_history.append((frame_time, probability))
                    combined_risk = probability
                    if extractor.eye_state is not None:
                        combined_risk = extractor.eye_state_module.fuse_eye_and_tcn_risk(
                            probability,
                            latest_eye_diagnostics["eye_closed_probability"],
                            latest_eye_diagnostics["eye_unknown_probability"],
                            args.eye_risk_weight,
                            latest_eye_diagnostics["eye_closed_sustained"],
                            latest_eye_diagnostics["eye_visible_fraction"],
                            args.eye_min_visible_fraction,
                        )
                    combined_risk_history.append((frame_time, combined_risk))
                    preview.record_tcn(
                        frame_time,
                        probability,
                        event_threshold,
                        (control_runtime.last_probabilities or {}).get("control_disengaged")
                        if control_runtime is not None else None,
                    )
                    onset_seed_probabilities: dict[str, float] = {}
                    fall_onset_tcn_inference_ms: float | None = None
                    if (
                        onset_runtime is not None
                        and len(onset_feature_buffer) == onset_window_steps
                    ):
                        onset_tcn_started = time.perf_counter()
                        onset_probability, onset_seed_probabilities = onset_runtime.predict(
                            onset_runtime.np.stack(onset_feature_buffer)
                        )
                        fall_onset_tcn_inference_ms = (
                            time.perf_counter() - onset_tcn_started
                        ) * 1000.0
                        onset_probability_history.append(
                            (frame_time, onset_probability)
                        )
                    base_combined_risk = combined_risk
                    fast_motion, fast_speed, fast_acceleration = motion_evidence(
                        feature_buffer, TCN_FEATURE_NAMES, target_fps
                    )
                    fast_onset_active = False
                    if args.fast_onset_risk:
                        combined_risk, fast_onset_active = fuse_onset_risk(
                            base_combined_risk, onset_probability, onset_threshold,
                            fast_motion, args.fast_onset_risk_floor,
                        )
                    eye_unreliable_fraction = float(
                        sum(eye_unreliable_buffer) / len(eye_unreliable_buffer)
                    ) if eye_unreliable_buffer else 1.0
                    timeline_writer.writerow({
                        "timestamp": round(frame_time, 3),
                        "tcn_probability": round(probability, 6),
                        "tcn_threshold": round(event_threshold, 6),
                        "tcn_direct_state": runtime.last_driver_response or "",
                        "tcn_raw_argmax_state": runtime.last_raw_driver_response or "",
                        "tcn_observable": int(runtime.last_observable)
                        if runtime.last_observable is not None else "",
                        "tcn_observation_quality": round(runtime.last_observation_quality, 6)
                        if runtime.last_observation_quality is not None else "",
                        "tcn_active_probability": round(
                            (runtime.last_state_probabilities or {}).get("active", 0.0), 6
                        ) if runtime.last_state_probabilities else "",
                        "tcn_reduced_probability": round(
                            (runtime.last_state_probabilities or {}).get("reduced", 0.0), 6
                        ) if runtime.last_state_probabilities else "",
                        "tcn_nvr_candidate": int(runtime.last_nvr_candidate)
                        if runtime.last_nvr_candidate is not None else "",
                        "tcn_inference_ms": round(tcn_inference_ms, 3),
                        "control_engaged_probability": round(
                            (control_runtime.last_probabilities or {}).get("control_engaged", 0.0), 6
                        ) if control_runtime and control_runtime.last_probabilities else "",
                        "control_disengaged_probability": round(
                            (control_runtime.last_probabilities or {}).get("control_disengaged", 0.0), 6
                        ) if control_runtime and control_runtime.last_probabilities else "",
                        "control_observable": int(control_runtime.last_observable)
                        if control_runtime and control_runtime.last_observable is not None else "",
                        "control_observation_quality": round(control_runtime.last_observation_quality, 6)
                        if control_runtime and control_runtime.last_observation_quality is not None else "",
                        "control_disengaged_candidate": int(control_runtime.last_control_disengaged_candidate)
                        if control_runtime and control_runtime.last_control_disengaged_candidate is not None else "",
                        "control_reduced_confirmed": int(runtime.last_control_reduced_confirmed)
                        if runtime.last_control_reduced_confirmed is not None else "",
                        "control_confirmation_windows": control_runtime.confirmation_windows
                        if control_runtime else "",
                        "control_operational": int(control_runtime.operational)
                        if control_runtime else "",
                        "control_tcn_inference_ms": round(
                            control_tcn_inference_ms, 3
                        ) if control_tcn_inference_ms is not None else "",
                        "tcn_nvr_probability": round(
                            (runtime.last_state_probabilities or {}).get(
                                "no_visible_response", 0.0
                            ), 6
                        ) if runtime.last_state_probabilities else "",
                        "tcn_transition_probability": round(
                            runtime.last_transition_probability, 6
                        ) if runtime.last_transition_probability is not None else "",
                        "tcn_transition_threshold": round(
                            runtime.transition_threshold, 6
                        ) if runtime.last_transition_probability is not None else "",
                        "tcn_transition_horizon_sec": round(
                            runtime.early_warning_horizon_sec, 3
                        ) if runtime.last_transition_probability is not None else "",
                        "tcn_transition_operational": int(runtime.transition_operational)
                        if runtime.last_transition_probability is not None else "",
                        "combined_risk": round(combined_risk, 6),
                        "combined_risk_threshold": round(combined_risk_threshold, 6),
                        "base_combined_risk": round(base_combined_risk, 6),
                        "fast_onset_active": int(fast_onset_active),
                        "fast_onset_motion": int(fast_motion),
                        "fast_onset_down_speed": round(fast_speed, 6),
                        "fast_onset_down_acceleration": round(fast_acceleration, 6),
                        "eye_measurement_reliable": int(
                            latest_eye_diagnostics["eye_measurement_reliable"]
                        ),
                        "eye_geometry_reliable": latest_eye_diagnostics[
                            "eye_geometry_reliable"
                        ],
                        "eye_unreliable_fraction": round(eye_unreliable_fraction, 6),
                        "ear_left": round(latest_eye_diagnostics["ear_left"], 6),
                        "ear_right": round(latest_eye_diagnostics["ear_right"], 6),
                        "ear_mean": round(latest_eye_diagnostics["ear_mean"], 6),
                        "eye_valid": latest_eye_diagnostics["eye_valid"],
                        "face_valid": latest_eye_diagnostics["face_valid"],
                        "face_reliability": round(
                            latest_eye_diagnostics["face_reliability"], 6
                        ),
                        "eye_blink_left": round(
                            latest_eye_diagnostics["eye_blink_left"], 6
                        ),
                        "eye_blink_right": round(
                            latest_eye_diagnostics["eye_blink_right"], 6
                        ),
                        "eye_blink_mean": round(
                            latest_eye_diagnostics["eye_blink_mean"], 6
                        ),
                        "eye_blendshape_valid": latest_eye_diagnostics[
                            "eye_blendshape_valid"
                        ],
                        "seat_head_dx": round(
                            latest_eye_diagnostics["seat_head_dx"], 6
                        ),
                        "seat_head_dy": round(
                            latest_eye_diagnostics["seat_head_dy"], 6
                        ),
                        "eye_open_probability": round(
                            latest_eye_diagnostics["eye_open_probability"], 6
                        ),
                        "eye_closed_probability": round(
                            latest_eye_diagnostics["eye_closed_probability"], 6
                        ),
                        "eye_unknown_probability": round(
                            latest_eye_diagnostics["eye_unknown_probability"], 6
                        ),
                        "eye_state_valid": latest_eye_diagnostics["eye_state_valid"],
                        "eye_visible_fraction": round(
                            latest_eye_diagnostics["eye_visible_fraction"], 6
                        ),
                        "eye_closed_sustained": latest_eye_diagnostics[
                            "eye_closed_sustained"
                        ],
                        "eye_crop_contrast": round(
                            latest_eye_diagnostics["eye_crop_contrast"], 6
                        ),
                        "eye_crop_sharpness": round(
                            latest_eye_diagnostics["eye_crop_sharpness"], 6
                        ),
                        "eye_crop_quality_valid": latest_eye_diagnostics[
                            "eye_crop_quality_valid"
                        ],
                        **{name: round(value, 6) for name, value in seed_probabilities.items()},
                        **(
                            {
                                "fall_onset_probability": round(onset_probability, 6),
                                "fall_onset_threshold": round(onset_threshold, 6),
                                "fall_onset_tcn_inference_ms": round(
                                    fall_onset_tcn_inference_ms, 3
                                ) if fall_onset_tcn_inference_ms is not None else "",
                                **{
                                    f"fall_onset_{name}": round(value, 6)
                                    for name, value in onset_seed_probabilities.items()
                                },
                            }
                            if onset_runtime is not None
                            and onset_probability is not None
                            and onset_threshold is not None
                            else {}
                        ),
                    })
                    timeline_stream.flush()
                    if not tcn_ready_logged:
                        print(
                            f"TCN 준비 완료: {window_steps / target_fps:.1f}초/"
                            f"{window_steps}프레임, 첫 확률={probability:.3f}",
                            flush=True,
                        )
                        tcn_ready_logged = True
                    candidate = combined_risk >= combined_risk_threshold
                    if candidate:
                        trigger_streak += 1
                        clear_streak = 0
                        if not event_active and trigger_streak >= trigger_ticks:
                            event_active, event_id = True, event_id + 1
                            event_trigger_window_count = trigger_streak
                            print(
                                f"TCN{' + 눈' if extractor.eye_state is not None else ''} 후보 감지: "
                                f"t={frame_time:.1f}s risk={combined_risk:.3f} "
                                f">= {combined_risk_threshold:.3f}; "
                                f"event {event_id}", flush=True,
                            )
                        elif event_active:
                            event_trigger_window_count = max(
                                event_trigger_window_count, trigger_streak
                            )
                    else:
                        trigger_streak = 0
                        clear_streak += 1
                        if event_active and clear_streak >= clear_ticks:
                            event_active = False
                            event_trigger_window_count = 0
                            qwen_status = "IDLE" if pending is None else qwen_status

                    onset_candidate = bool(
                        onset_probability is not None
                        and onset_threshold is not None
                        and onset_probability >= onset_threshold
                        and (not args.fast_onset_risk or fast_motion)
                    )
                    if onset_candidate:
                        onset_trigger_streak += 1
                    else:
                        onset_trigger_streak = 0

                    # TCN ranks where to look first, but it must not become a
                    # single point of failure for a visually obvious collapse.
                    # A sparse Qwen safety watch therefore runs even below the
                    # temporal threshold.  Candidate events retain priority and
                    # the shorter event recheck interval.
                    tcn_submit_due = (
                        event_active
                        and frame_time - last_submit_time >= args.qwen_recheck_sec
                    )
                    # The Drive&Act branch makes `reduced` a VLM verification
                    # trigger as well: TCN selects a sustained control-loss
                    # candidate, then the VLM receives the same YOLO driver
                    # video evidence before any fused final decision.
                    reduced_submit_due = (
                        not event_active
                        and runtime.last_driver_response == "reduced"
                        and runtime.last_control_reduced_confirmed is True
                        and control_runtime is not None
                        and frame_time - last_submit_time >= args.qwen_recheck_sec
                    )
                    # The short onset model is deliberately only a VLM gate.
                    # It does not mark the long-horizon event as active and it
                    # never supplies a driver state on its own.
                    onset_submit_due = (
                        not event_active
                        and onset_candidate
                        and onset_trigger_streak >= onset_trigger_ticks
                        and frame_time - last_onset_submit_time
                        >= args.fall_onset_recheck_sec
                    )
                    # A bad EAR fit must not be converted into "eyes open".
                    # Send a VLM check only when that uncertainty persists and
                    # the temporal model is already near its calibrated gate;
                    # this avoids turning an ordinary one-frame face-detector
                    # miss into a safety alert.
                    eye_uncertain = (
                        len(eye_unreliable_buffer) == eye_unreliable_buffer.maxlen
                        and sum(eye_unreliable_buffer) / len(eye_unreliable_buffer) >= 0.80
                    )
                    eye_watchdog_submit_due = (
                        not event_active
                        and args.eye_uncertain_watchdog_sec > 0.0
                        and eye_uncertain
                        and probability >= event_threshold * args.eye_uncertain_tcn_ratio
                        and frame_time - last_eye_watchdog_submit_time
                        >= args.eye_uncertain_watchdog_sec
                    )
                    watchdog_submit_due = (
                        not event_active
                        and not onset_submit_due
                        and not eye_watchdog_submit_due
                        and args.visual_watchdog_sec > 0.0
                        and frame_time - last_watchdog_submit_time >= args.visual_watchdog_sec
                    )
                    if pending is None and (
                        tcn_submit_due or reduced_submit_due or onset_submit_due
                        or eye_watchdog_submit_due or watchdog_submit_due
                    ):
                        is_tcn_candidate = bool(tcn_submit_due)
                        is_reduced_candidate = bool(
                            not is_tcn_candidate and reduced_submit_due
                        )
                        is_onset_candidate = bool(
                            not is_tcn_candidate and not is_reduced_candidate
                            and onset_submit_due
                        )
                        trigger_source = (
                            "main_tcn" if is_tcn_candidate
                            else "control_reduced_tcn" if is_reduced_candidate
                            else "fall_onset_tcn" if is_onset_candidate
                            else "eye_uncertain_watchdog" if eye_watchdog_submit_due
                            else "visual_watchdog"
                        )
                        if not is_tcn_candidate:
                            event_id += 1
                        event_probability = (
                            combined_risk if is_tcn_candidate
                            else float((control_runtime.last_probabilities or {}).get("control_disengaged", 0.0))
                            if is_reduced_candidate
                            else onset_probability
                        )
                        event_gate_threshold = (
                            combined_risk_threshold if is_tcn_candidate
                            else control_runtime.threshold if is_reduced_candidate and control_runtime is not None
                            else onset_threshold
                        )
                        event_window_steps = (
                            window_steps if is_tcn_candidate
                            else control_runtime.window_steps if is_reduced_candidate and control_runtime is not None
                            else onset_window_steps
                        )
                        event_trigger_count = (
                            event_trigger_window_count if is_tcn_candidate
                            else control_runtime.confirmation_windows if is_reduced_candidate and control_runtime is not None
                            else onset_trigger_streak if is_onset_candidate else 0
                        )
                        event_candidate_detected = bool(
                            is_tcn_candidate or is_reduced_candidate or is_onset_candidate
                        )
                        if event_probability is None or event_gate_threshold is None:
                            raise RuntimeError("이벤트 TCN 확률 또는 threshold가 없습니다.")
                        probability_history = (
                            combined_risk_history if is_tcn_candidate
                            else control_probability_history if is_reduced_candidate
                            else onset_probability_history if is_onset_candidate
                            else tcn_probability_history
                        )
                        recent_probabilities = [
                            float(value) for _, value in probability_history
                        ]
                        if len(recent_probabilities) < 2:
                            risk_trend = "unavailable"
                        else:
                            probability_change = (
                                recent_probabilities[-1] - recent_probabilities[0]
                            )
                            risk_trend = (
                                "increasing" if probability_change >= 0.03
                                else "decreasing" if probability_change <= -0.03
                                else "stable"
                            )
                        if probability_history:
                            peak_timestamp_sec, peak_probability = max(
                                probability_history, key=lambda item: item[1]
                            )
                        else:
                            peak_timestamp_sec, peak_probability = (
                                frame_time, event_probability
                            )
                        main_member_values = [float(value) for value in seed_probabilities.values()]
                        main_ensemble = {
                            "member_count": len(main_member_values),
                            "std": float(runtime.np.std(main_member_values)) if main_member_values else 0.0,
                            "minimum": min(main_member_values) if main_member_values else 0.0,
                            "maximum": max(main_member_values) if main_member_values else 0.0,
                        }
                        control_member_values = (
                            list(control_runtime.last_member_probabilities)
                            if control_runtime is not None else []
                        )
                        control_ensemble = {
                            "member_count": len(control_member_values),
                            "std": float(runtime.np.std(control_member_values)) if control_member_values else 0.0,
                            "minimum": min(control_member_values) if control_member_values else 0.0,
                            "maximum": max(control_member_values) if control_member_values else 0.0,
                        }
                        temporal_context = {
                            "trigger_source": trigger_source,
                            "risk_trend": risk_trend,
                            "recent_probabilities": recent_probabilities,
                            "above_threshold_duration_sec": (
                                event_trigger_count * args.tcn_interval_sec
                                if event_candidate_detected else 0.0
                            ),
                            "peak_probability": peak_probability,
                            "peak_timestamp_sec": peak_timestamp_sec,
                            "state_probabilities": (
                                runtime.last_state_probabilities
                                if runtime.vlm_state_distribution_operational else None
                            ),
                            "transition_probability": (
                                runtime.last_transition_probability
                                if runtime.transition_operational else None
                            ),
                            "transition_horizon_sec": runtime.early_warning_horizon_sec,
                            "main_tcn": {
                                "nvr_probability": probability,
                                "combined_risk": combined_risk,
                                "combined_risk_threshold": combined_risk_threshold,
                                "observable": runtime.last_observable,
                                "observation_quality": runtime.last_observation_quality,
                                "raw_argmax_state": runtime.last_raw_driver_response,
                                "state_decision": runtime.state_decision,
                                "reduced_head_operational": runtime.reduced_head_operational,
                                "ensemble": main_ensemble,
                                "history": [
                                    {
                                        "time_offset_sec": round(timestamp - frame_time, 3),
                                        "probability": value,
                                    }
                                    for timestamp, value in combined_risk_history
                                ],
                            },
                            "eye_evidence": {
                                "measurement_reliable": bool(
                                    latest_eye_diagnostics["eye_measurement_reliable"]
                                ),
                                "visible_fraction": latest_eye_diagnostics["eye_visible_fraction"],
                                "closed_probability": latest_eye_diagnostics["eye_closed_probability"],
                                "unknown_probability": latest_eye_diagnostics["eye_unknown_probability"],
                                "closed_sustained": bool(
                                    latest_eye_diagnostics["eye_closed_sustained"]
                                ),
                            },
                            "control_tcn": (
                                {
                                    "engaged_probability": (
                                        control_runtime.last_probabilities or {}
                                    ).get("control_engaged", 0.0),
                                    "disengaged_probability": (
                                        control_runtime.last_probabilities or {}
                                    ).get("control_disengaged", 0.0),
                                    "observable": control_runtime.last_observable,
                                    "observation_quality": control_runtime.last_observation_quality,
                                    "candidate": control_runtime.last_control_disengaged_candidate,
                                    "reduced_confirmed": runtime.last_control_reduced_confirmed,
                                    "confirmation_windows": control_runtime.confirmation_windows,
                                    "ensemble": control_ensemble,
                                    "history": [
                                        {
                                            "time_offset_sec": round(timestamp - frame_time, 3),
                                            "probability": value,
                                        }
                                        for timestamp, value in control_probability_history
                                    ],
                                }
                                if control_runtime is not None else None
                            ),
                        }
                        evidence_seconds = args.context_seconds + args.candidate_seconds
                        evidence_start = frame_time - evidence_seconds
                        full_source_video = bool(
                            not args.vlm_match_tcn_crop
                            and args.event_media_format == "video"
                            and args.event_full_source_fps
                        )
                        event_output_fps = src_fps if full_source_video else args.event_fps
                        if args.vlm_match_tcn_crop:
                            # `visual_buffer` contains `crop.copy()` directly
                            # after the selected YOLO driver box was used by
                            # StreamingFeatureExtractor.  Reusing it makes the
                            # VLM and TCN spatial inputs exactly identical.
                            evidence_frames = select_event_video_frames(
                                visual_buffer, frame_time,
                                evidence_seconds, args.event_fps,
                            )
                            print("VLM에 TCN 동일 YOLO crop 증거 생성", flush=True)
                        else:
                            reference = (
                                min(
                                    driver_box_buffer,
                                    key=lambda item: abs(item[0] - evidence_start),
                                )
                                if driver_box_buffer else None
                            )
                            if reference is not None:
                                reference_time, reference_box = reference
                                fixed_frame_count = (
                                    4 if args.event_media_format == "storyboard"
                                    else max(2, int(round(evidence_seconds * event_output_fps)))
                                )
                                try:
                                    evidence_frames = select_fixed_driver_box_event_frames(
                                        source_video,
                                        frame_time,
                                        evidence_seconds,
                                        fixed_frame_count,
                                        reference_box,
                                        args.vlm_fixed_driver_box_padding,
                                        args.vlm_driver_seat_left_ratio,
                                        continuous_source_frames=full_source_video,
                                    )
                                    print(
                                        "VLM 고정 운전자 box 증거 생성 "
                                        f"(reference t={reference_time:.2f}s)",
                                        flush=True,
                                    )
                                except RuntimeError as exc:
                                    print(
                                        f"VLM 고정 box 생성 실패, 라이브 crop으로 대체: {exc}",
                                        flush=True,
                                    )
                                    evidence_frames = select_event_video_frames(
                                        visual_buffer, frame_time,
                                        evidence_seconds, args.event_fps,
                                    )
                            else:
                                evidence_frames = select_event_video_frames(
                                    visual_buffer, frame_time,
                                    evidence_seconds, args.event_fps,
                                )
                        media_stem = media_dir / f"event_{event_id:04d}_{frame_time:010.3f}"
                        event_media = (
                            encode_event_storyboard(
                                evidence_frames, media_stem.with_suffix(".jpg"),
                                args.event_storyboard_panel_size,
                            )
                            if args.event_media_format == "storyboard"
                            else encode_event_video(
                                evidence_frames, media_stem.with_suffix(".mp4"),
                                event_output_fps, args.qwen_event_max_width,
                            )
                        )
                        # Keep the TCN candidate interval separate from the
                        # actual video interval.  Qwen needs the configured
                        # pre-event context to see a fall and distinguish it
                        # from a later normal-looking frame.
                        candidate_start = frame_time - args.candidate_seconds
                        pending = executor.submit(
                            post_tcn_event, args, headers, source_video, event_media,
                            observation_id, frame_time, event_probability, event_gate_threshold,
                            event_window_steps, event_trigger_count,
                            candidate_start, event_candidate_detected,
                            temporal_context,
                        )
                        pending_media = event_media
                        pending_metadata = {
                            "event_id": event_id,
                            "video_time": frame_time,
                            "event_video_start_sec": evidence_start,
                            "tcn_probability": event_probability,
                            "fall_onset_probability": onset_probability,
                            "tcn_threshold": event_gate_threshold,
                            "tcn_window_steps": event_window_steps,
                            "eye_unreliable_fraction": eye_unreliable_fraction,
                            "submitted_monotonic": time.monotonic(),
                            "trigger": trigger_source,
                        }
                        # Source-video time anchors the request marker.  The
                        # later response uses wall-clock elapsed time because
                        # replay speed is not VLM inference latency.
                        preview.record_vlm_request(
                            frame_time, args.candidate_seconds, event_probability,
                            event_gate_threshold, trigger_source,
                        )
                        observation_id += 1
                        if is_tcn_candidate:
                            last_submit_time = frame_time
                            seed_text = ", ".join(
                                f"{name}={value:.3f}"
                                for name, value in seed_probabilities.items()
                            )
                            print(
                                "Qwen TCN 이벤트 추론 시작: "
                                f"t={frame_time:.1f}s, "
                                f"TCN={probability:.3f} >= threshold={event_threshold:.3f}; "
                                f"seeds[{seed_text}]",
                                flush=True,
                            )
                        elif is_reduced_candidate:
                            last_submit_time = frame_time
                            print(
                                "Qwen 제어 이탈 reduced 검증 시작: "
                                f"t={frame_time:.1f}s, "
                                f"control_disengaged={event_probability:.3f} "
                                f"({control_runtime.confirmation_windows}회 지속)",
                                flush=True,
                            )
                        elif is_onset_candidate:
                            last_onset_submit_time = frame_time
                            print(
                                "Qwen fall-onset 보조 TCN 추론 시작 "
                                f"(P={onset_probability:.3f} >= {onset_threshold:.3f})",
                                flush=True,
                            )
                        elif eye_watchdog_submit_due:
                            last_eye_watchdog_submit_time = frame_time
                            print(
                                "Qwen 눈-불확실 안전 감시 추론 시작 "
                                f"(눈 품질 불량 {eye_unreliable_fraction:.0%})",
                                flush=True,
                            )
                        else:
                            last_watchdog_submit_time = frame_time
                            print("Qwen 시각 안전 감시 추론 시작 (TCN 미검출)", flush=True)
                        qwen_status = "RUNNING"
            if args.realtime:
                target_elapsed = frame_time - start_sec
                sleep_seconds = target_elapsed - (time.perf_counter() - wall_start)
                if sleep_seconds > 0:
                    preview.wait(sleep_seconds)
        if pending is not None:
            print("마지막 Qwen 이벤트 결과를 기다립니다.", flush=True)
            bridge_result = pending.result()
            if pending_metadata is None:
                raise RuntimeError("마지막 Qwen 이벤트 메타데이터가 없습니다.")
            last_response = print_qwen_event_result(
                bridge_result, pending_metadata, args.print_json
            )
            preview.record_vlm_response(
                last_response,
                time.monotonic() - pending_metadata["submitted_monotonic"],
            )
            record = {
                "source_video": str(source_video),
                "event_id": pending_metadata["event_id"],
                "video_time": pending_metadata["video_time"],
                "event_video_start_sec": pending_metadata["event_video_start_sec"],
                "tcn_probability": pending_metadata["tcn_probability"],
                "fall_onset_probability": pending_metadata.get("fall_onset_probability"),
                "eye_unreliable_fraction": pending_metadata.get(
                    "eye_unreliable_fraction"
                ),
                "trigger": pending_metadata.get("trigger"),
                "tcn_threshold": pending_metadata["tcn_threshold"],
                "qwen_model_inference_ms": bridge_result.get("inference_ms"),
                "event_end_to_end_sec": round(
                    time.monotonic() - pending_metadata["submitted_monotonic"], 4
                ),
                "bridge_result": bridge_result,
            }
            with results_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                event_records_written += 1
            if pending_media is not None and not args.keep_event_media:
                pending_media.unlink(missing_ok=True)
    finally:
        cap.release()
        extractor.close()
        timeline_stream.close()
        executor.shutdown(wait=True, cancel_futures=False)
    print(f"이벤트 결과 저장: {results_path}", flush=True)
    print(f"TCN/눈 품질 타임라인 저장: {timeline_path}", flush=True)
    if args.ground_truth_file is not None:
        if event_records_written:
            evaluate_qwen_event_results(
                args,
                results_path,
                event_records_written,
                match_video_time=True,
                plot_path_override=args.accuracy_plot_file,
            )
        else:
            print("Qwen 이벤트가 없어 이번 실행의 모델 답변 정확도는 계산하지 않습니다.", flush=True)
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
            self.tcn_history: deque[tuple[float, float, float, float | None]] = deque(maxlen=240)
            self.vlm_events: deque[dict[str, Any]] = deque(maxlen=36)
            # This is deliberately live state only.  A completed VLM verdict
            # is evidence for its event, not a verdict that remains in force
            # for the following source-video frames.
            self.vlm_inference_active = False
            self.vlm_live_result: str | None = None
            self.vlm_result_visible_until = 0.0
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
            # Keep one stable image-only window and update its pixels in place.
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
        image = self._prepare_frame(image, f"{label}  {image_path.name}")
        self.cv2.imshow(self.WINDOW_NAME, image)
        self.cv2.waitKey(1)

    def record_tcn(
        self, video_time: float, nvr_probability: float, threshold: float,
        control_disengaged_probability: float | None = None,
    ) -> None:
        if self.enabled:
            self.tcn_history.append((
                float(video_time), float(nvr_probability), float(threshold),
                None if control_disengaged_probability is None else float(control_disengaged_probability),
            ))

    def record_vlm_request(
        self, video_time: float, candidate_seconds: float = 1.5,
        tcn_probability: float | None = None, tcn_threshold: float | None = None,
        trigger_source: str = "",
    ) -> None:
        if self.enabled:
            self.vlm_inference_active = True
            self.vlm_live_result = None
            self.vlm_result_visible_until = 0.0
            self.vlm_events.append({
                "video_time": float(video_time),
                "candidate_seconds": max(0.1, float(candidate_seconds)),
                "tcn_probability": (
                    None if tcn_probability is None else float(tcn_probability)
                ),
                "tcn_threshold": (
                    None if tcn_threshold is None else float(tcn_threshold)
                ),
                "trigger_source": str(trigger_source),
                "response": None,
                "latency_sec": None,
            })

    def record_vlm_response(self, response: str | None, latency_sec: float) -> None:
        if not self.enabled:
            return
        self.vlm_inference_active = False
        if response not in DRIVER_RESPONSE_LABELS:
            return
        # Briefly expose the conclusion to the operator, then clear it.  This
        # makes a conclusion readable without presenting it as ongoing state.
        self.vlm_live_result = str(response)
        self.vlm_result_visible_until = time.monotonic() + 2.0
        for event in reversed(self.vlm_events):
            if event["response"] is None:
                event["response"] = str(response)
                event["latency_sec"] = max(0.0, float(latency_sec))
                return

    def finish_vlm_inference(self) -> None:
        """Clear the live indicator after a failed or completed VLM request."""
        if self.enabled:
            self.vlm_inference_active = False
            self.vlm_live_result = None
            self.vlm_result_visible_until = 0.0

    def show_frame(self, frame: Any, caption: str, status_lines: list[str] | None = None) -> bool:
        """Render the full source-video frame immediately; return True on q."""
        if not self.enabled or self.cv2 is None:
            return False
        image = self._prepare_frame(frame.copy(), caption)
        if status_lines:
            y = 70
            for line in status_lines:
                self.cv2.putText(
                    image, line, (12, y), self.cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (0, 255, 255), 2, self.cv2.LINE_AA,
                )
                y += 28
        image = self._append_live_dashboard(image)
        self.cv2.imshow(self.WINDOW_NAME, image)
        return (self.cv2.waitKey(1) & 0xFF) == ord("q")

    def _append_live_dashboard(self, image: Any) -> Any:
        """Attach charts instead of overlaying inference text on the video."""
        import numpy as np

        cv2 = self.cv2
        height, width = image.shape[:2]
        # Render the dashboard below the source video at the same width.
        panel_height = 420
        panel = np.full((panel_height, width, 3), (20, 20, 20), dtype=np.uint8)
        left, right, top, chart_bottom = 50, max(52, width - 14), 30, 145
        chart_width = max(1, right - left)
        # The preview begins before the first TCN window is complete.  Keep a
        # harmless placeholder time range during those first few frames so the
        # VLM panel can still render its continuous timestamp axis.
        now_time = self.tcn_history[-1][0] if self.tcn_history else 0.0
        start_time = now_time - 12.0
        for fraction in (0.0, 0.5, 1.0):
            y = int(round(chart_bottom - fraction * (chart_bottom - top)))
            cv2.line(panel, (left, y), (right, y), (65, 65, 65), 1, cv2.LINE_AA)
        if self.tcn_history:
            values = [item for item in self.tcn_history if item[0] >= start_time]
            points = []
            control_points = []
            for timestamp, nvr, _threshold, control in values:
                x = left + int(round((timestamp - start_time) / 12.0 * chart_width))
                points.append((x, int(round(chart_bottom - max(0.0, min(1.0, nvr)) * (chart_bottom - top)))))
                if control is not None:
                    control_points.append((x, int(round(chart_bottom - max(0.0, min(1.0, control)) * (chart_bottom - top)))))
            threshold = values[-1][2]
            threshold_y = int(round(chart_bottom - max(0.0, min(1.0, threshold)) * (chart_bottom - top)))
            cv2.line(panel, (left, threshold_y), (right, threshold_y), (45, 45, 220), 1, cv2.LINE_AA)
            if len(points) > 1:
                cv2.polylines(panel, [np.asarray(points, dtype=np.int32)], False, (0, 180, 255), 2, cv2.LINE_AA)
            if len(control_points) > 1:
                cv2.polylines(panel, [np.asarray(control_points, dtype=np.int32)], False, (235, 120, 150), 2, cv2.LINE_AA)
            # Use source-video timestamps here too, so the TCN curve aligns
            # directly with the VLM verification timeline below it.
            for tick_index in range(5):
                fraction = tick_index / 4.0
                x = left + int(round(fraction * chart_width))
                label = f"{start_time + fraction * 12.0:.1f}s"
                label_width = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1
                )[0][0]
                label_x = max(left, min(right - label_width, x - label_width // 2))
                cv2.putText(panel, label, (label_x, 159), cv2.FONT_HERSHEY_SIMPLEX,
                            0.34, (185, 185, 185), 1, cv2.LINE_AA)
        # Numeric y-axis keeps the probability range explicit.
        for fraction, label in ((1.0, "1.0"), (0.5, "0.5"), (0.0, "0.0")):
            y = int(round(chart_bottom - fraction * (chart_bottom - top)))
            cv2.putText(panel, label, (10, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (190, 190, 190), 1, cv2.LINE_AA)
        cv2.putText(panel, "TCN RISK TREND", (left, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (220, 220, 220), 1, cv2.LINE_AA)
        # Legend: orange=No Visible candidate score, purple=Reduced candidate score,
        # red=VLM candidate threshold.
        legend_y = 176
        legend_items = [
            ((0, 180, 255), "No visible response score"),
            ((235, 120, 150), "Reduced score"),
            ((45, 45, 220), "threshold"),
        ]
        legend_x = left
        for color, label in legend_items:
            cv2.line(panel, (legend_x, legend_y), (legend_x + 16, legend_y), color, 2, cv2.LINE_AA)
            cv2.putText(panel, label, (legend_x + 21, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.40, (220, 220, 220), 1, cv2.LINE_AA)
            # The labels have different widths; advance by the rendered text
            # width instead of a fixed offset so adjacent legend entries never
            # overlap when wording changes.
            label_width = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1
            )[0][0]
            legend_x += 21 + label_width + 18
        # Separate the VLM section from the TCN legend so their colors do not
        # look like one combined legend.
        section_divider_y = 188
        cv2.line(panel, (left, section_divider_y), (right, section_divider_y),
                 (85, 85, 85), 1, cv2.LINE_AA)
        cv2.putText(panel, "VLM EVENT VERIFICATION", (left, 207), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (220, 220, 220), 1, cv2.LINE_AA)
        # A translucent sky-blue vertical band marks a TCN threshold crossing
        # that actually submitted a VLM call.  It spans every VLM-result lane
        # so candidate and final-result times align at a glance.
        vlm_legend_y = 224
        legend_x = left
        cv2.rectangle(panel, (legend_x + 2, vlm_legend_y - 11), (legend_x + 10, vlm_legend_y + 2),
                      (165, 115, 55), -1)
        trigger_label = "TCN-triggered VLM request"
        cv2.putText(panel, trigger_label, (legend_x + 16, vlm_legend_y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, (220, 220, 220), 1, cv2.LINE_AA)
        trigger_width = cv2.getTextSize(
            trigger_label, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1
        )[0][0]
        legend_x += 16 + trigger_width + 18
        colors = {
            "active": (70, 190, 70), "reduced": (0, 210, 240),
            "no_visible_response": (45, 45, 230),
        }
        for state, short_label in (
            ("active", "active"),
            ("reduced", "reduced"),
            ("no_visible_response", "no visible response"),
        ):
            cv2.rectangle(panel, (legend_x, vlm_legend_y - 8), (legend_x + 7, vlm_legend_y - 1), colors[state], -1)
            cv2.putText(panel, short_label, (legend_x + 11, vlm_legend_y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                        0.36, (220, 220, 220), 1, cv2.LINE_AA)
            label_width = cv2.getTextSize(
                short_label, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1
            )[0][0]
            legend_x += 11 + label_width + 18
        # VLM outputs are event-specific categorical conclusions, not a
        # continuously held driver state.  Each completed conclusion is an
        # individual marker at its source-video timestamp; it must never
        # extend forward until the next VLM request.
        lane_top, lane_bottom = 240, 342
        lane_centers = {
            "no_visible_response": 257,
            "reduced": 291,
            "active": 325,
        }
        lane_labels = {
            "no_visible_response": "NO VISIBLE RESPONSE",
            "reduced": "REDUCED",
            "active": "ACTIVE",
        }
        timeline_tick_count = 4
        for tick_index in range(timeline_tick_count + 1):
            fraction = tick_index / timeline_tick_count
            grid_x = left + int(round(fraction * chart_width))
            cv2.line(panel, (grid_x, lane_top), (grid_x, lane_bottom),
                     (55, 55, 55), 1, cv2.LINE_AA)
        for lane, lane_y in lane_centers.items():
            cv2.rectangle(panel, (left, lane_y - 9), (right, lane_y + 9),
                          (34, 34, 34), -1)
            cv2.line(panel, (left, lane_y + 10), (right, lane_y + 10),
                     (70, 70, 70), 1, cv2.LINE_AA)
            if lane == "no_visible_response":
                # Keep the same 0.32 scale as ACTIVE/REDUCED without taking
                # horizontal space from the timeline itself.
                cv2.putText(panel, "NO VISIBLE", (2, lane_y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, colors[lane], 1,
                            cv2.LINE_AA)
                cv2.putText(panel, "RESPONSE", (2, lane_y + 9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, colors[lane], 1,
                            cv2.LINE_AA)
            else:
                cv2.putText(panel, lane_labels[lane], (2, lane_y + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, colors[lane], 1,
                            cv2.LINE_AA)
        visible_events: list[dict[str, Any]] = []
        resolved_events: list[dict[str, Any]] = []
        if self.tcn_history:
            visible_events = [
                event for event in self.vlm_events
                if float(event["video_time"]) >= start_time
            ]
            resolved_events = sorted(
                (event for event in self.vlm_events if event["response"] in colors),
                key=lambda event: float(event["video_time"]),
            )

        # Draw candidate bands first.  The completed VLM-result markers are
        # drawn afterwards, so a request band can never cover their verdict.
        band_half_width = max(3, int(round(chart_width * 0.004)))
        for event_index, event in enumerate(visible_events):
            event_time = float(event["video_time"])
            x = left + int(round((event_time - start_time) / 12.0 * chart_width))
            x = max(left, min(right, x))
            x0, x1 = max(left, x - band_half_width), min(right, x + band_half_width)
            if x1 >= x0:
                roi = panel[lane_top:lane_bottom + 1, x0:x1 + 1]
                # BGR (165, 115, 55): muted sky blue that is distinct from
                # the green/yellow/red VLM result-state palette.
                tint = np.full_like(roi, (165, 115, 55))
                cv2.addWeighted(tint, 0.35, roi, 0.65, 0.0, roi)
                cv2.line(panel, (x, lane_top), (x, lane_bottom),
                         (225, 185, 105), 1, cv2.LINE_AA)
            timestamp_label = f"{event_time:.1f}s"
            label_width = cv2.getTextSize(timestamp_label, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)[0][0]
            label_x = max(left, min(right - label_width, x - label_width // 2))
            label_y = 356 if event_index % 2 == 0 else 369
            cv2.putText(panel, timestamp_label, (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, (210, 210, 210), 1, cv2.LINE_AA)

        # Completed VLM verdicts are individual square markers (not held
        # state bars), deliberately drawn in the foreground.
        for event in resolved_events:
            state = str(event["response"])
            event_time = float(event["video_time"])
            if event_time < start_time or event_time > now_time:
                continue
            x = left + int(round((event_time - start_time) / 12.0 * chart_width))
            x = max(left, min(right, x))
            lane_y = lane_centers[state]
            cv2.rectangle(panel, (x - 6, lane_y - 6), (x + 6, lane_y + 6),
                          colors[state], -1)

        # Never keep the last VLM conclusion in the live status field.  Show
        # it briefly on completion, then return to IDLE; the event point above
        # remains the only historical record on this screen.
        status_top, status_bottom = 378, 406
        cv2.rectangle(panel, (left, status_top), (right, status_bottom), (42, 42, 42), -1)
        cv2.rectangle(panel, (left, status_top), (right, status_bottom), (85, 85, 85), 1)
        state_display = {
            "active": "ACTIVE",
            "reduced": "REDUCED",
            "no_visible_response": "NO VISIBLE RESPONSE",
        }
        if self.vlm_inference_active:
            live_label, live_color = "VLM INFERENCE RUNNING", (255, 210, 80)
        elif (
            self.vlm_live_result in state_display
            and time.monotonic() < self.vlm_result_visible_until
        ):
            result = str(self.vlm_live_result)
            live_label, live_color = f"VLM RESULT: {state_display[result]}", colors[result]
        else:
            live_label, live_color = "VLM WAITING - NOT INFERENCING", (190, 190, 190)
        cv2.putText(panel, live_label, (left + 10, 398), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, live_color, 2, cv2.LINE_AA)
        return cv2.vconcat([image, panel])

    def _prepare_frame(self, image: Any, caption: str) -> Any:
        """Resize a frame to the preview area and add a status caption."""
        height, width = image.shape[:2]
        # Reserve vertical space for the 420 px live dashboard.
        scale = min(960.0 / width, 700.0 / height, 1.0)
        if scale < 1.0:
            image = self.cv2.resize(
                image,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=self.cv2.INTER_AREA,
            )
        if caption:
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
        return image

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
        tcn_evidence = result.get("tcn_evidence")
        tcn_text = ""
        if isinstance(tcn_evidence, dict):
            tcn_text = (
                f" tcn_candidate={tcn_evidence.get('candidate_detected')}"
                f" tcn_peak={tcn_evidence.get('peak_probability')}"
            )
        print(
            "DRIVER MONITOR result "
            f"image={image_path.name} "
            f"eye_state={state.get('eye_state')} "
            f"head_pose={state.get('head_pose')} "
            f"upper_body_posture={state.get('upper_body_posture')} "
            f"driver_response={state.get('driver_response')} "
            f"inference_ms={result.get('inference_ms')}"
            f"{tcn_text}",
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
    result = record.get("bridge_result")
    if isinstance(result, dict):
        evidence = result.get("tcn_evidence")
        if isinstance(evidence, dict):
            events = evidence.get("events")
            if isinstance(events, list) and events and isinstance(events[0], dict):
                start = events[0].get("start_sec")
                if isinstance(start, (int, float)):
                    return float(start)
    video_time = record.get("video_time")
    if isinstance(video_time, (int, float)):
        # Legacy files did not persist the evidence range. Their event video
        # was 2.5 s, so retain a deterministic, conservative fallback.
        return float(video_time) - 2.5
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
            "tcn_probability": record.get("tcn_probability", ""),
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
    if args.driver_box_hold_sec < 0:
        raise ValueError("--driver-box-hold-sec은 0 이상이어야 합니다.")
    if not 0.0 < args.driver_yolo_confidence <= 1.0:
        raise ValueError("--driver-yolo-confidence는 0 초과 1 이하여야 합니다.")
    if args.driver_yolo_interval < 1:
        raise ValueError("--driver-yolo-interval은 1 이상이어야 합니다.")
    if not 0.0 <= args.driver_min_x_ratio < 1.0:
        raise ValueError("--driver-min-x-ratio는 0 이상 1 미만이어야 합니다.")
    fallback_roi = args.fallback_driver_roi
    if not all(0.0 <= value <= 1.0 for value in fallback_roi) or not (
        fallback_roi[0] < fallback_roi[2]
        and fallback_roi[1] < fallback_roi[3]
    ):
        raise ValueError("--fallback-driver-roi는 0~1 범위의 X1 Y1 X2 Y2여야 합니다.")
    if args.yolo_face_crop and not args.yolo_face_model.is_file():
        raise FileNotFoundError(f"YOLO 얼굴 가중치가 없습니다: {args.yolo_face_model}")
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
    if args.tcn_score_dir is not None and not args.tcn_score_dir.is_dir():
        raise FileNotFoundError(
            f"--tcn-score-dir 디렉터리가 없습니다: {args.tcn_score_dir}"
        )
    if args.video_path is not None:
        if args.video_path.suffix.lower() != ".mp4":
            raise ValueError("--video-path에는 MP4 전체 영상을 지정하세요.")
        if args.image_path is not None or args.watch:
            raise ValueError("--video-path는 --image-path 또는 --watch와 함께 사용할 수 없습니다.")
        if args.video_start_sec < 0 or args.video_max_sec is not None and args.video_max_sec <= 0:
            raise ValueError("--video-start-sec/--video-max-sec 값이 올바르지 않습니다.")
        if min(args.tcn_interval_sec, args.trigger_persist_sec, args.clear_persist_sec,
               args.qwen_recheck_sec, args.candidate_seconds,
               args.event_fps, args.fall_onset_trigger_persist_sec,
               args.fall_onset_recheck_sec) <= 0:
            raise ValueError("실시간 TCN/Qwen 시간 옵션은 모두 0보다 커야 합니다.")
        if args.context_seconds < 0.0:
            raise ValueError("--context-seconds는 0 이상이어야 합니다.")
        if args.qwen_event_max_width < 0:
            raise ValueError("--qwen-event-max-width는 0 이상이어야 합니다.")
        if args.visual_watchdog_sec < 0.0:
            raise ValueError("--visual-watchdog-sec은 0 이상이어야 합니다.")
        if args.eye_uncertain_watchdog_sec < 0.0:
            raise ValueError("--eye-uncertain-watchdog-sec은 0 이상이어야 합니다.")
        if args.eye_uncertain_window_sec <= 0.0:
            raise ValueError("--eye-uncertain-window-sec은 0보다 커야 합니다.")
        if args.eye_state_smoothing_sec <= 0.0:
            raise ValueError("--eye-state-smoothing-sec은 0보다 커야 합니다.")
        if not 0.0 < args.eye_closed_probability_threshold < 1.0:
            raise ValueError("--eye-closed-probability-threshold은 0~1 사이여야 합니다.")
        if args.eye_closed_persist_sec <= 0.0:
            raise ValueError("--eye-closed-persist-sec은 0보다 커야 합니다.")
        if not 0.0 < args.eye_min_visible_fraction <= 1.0:
            raise ValueError("--eye-min-visible-fraction은 0 초과 1 이하여야 합니다.")
        if not 0.0 <= args.eye_risk_weight <= 1.0:
            raise ValueError("--eye-risk-weight는 0~1 사이여야 합니다.")
        if (
            args.combined_risk_threshold is not None
            and not 0.0 < args.combined_risk_threshold <= 1.0
        ):
            raise ValueError("--combined-risk-threshold는 0 초과 1 이하여야 합니다.")
        if args.eye_state_model is not None and not args.eye_state_model.is_file():
            raise FileNotFoundError(f"--eye-state-model이 없습니다: {args.eye_state_model}")
        if not 160 <= args.event_storyboard_panel_size <= 512:
            raise ValueError("--event-storyboard-panel-size는 160~512여야 합니다.")
        if not 0.0 < args.eye_uncertain_tcn_ratio <= 1.0:
            raise ValueError("--eye-uncertain-tcn-ratio는 0 초과 1 이하여야 합니다.")
        if args.event_threshold is not None and not 0.0 < args.event_threshold <= 1.0:
            raise ValueError("--event-threshold는 0 초과 1 이하여야 합니다.")
        if (
            args.fall_onset_event_threshold is not None
            and not 0.0 < args.fall_onset_event_threshold <= 1.0
        ):
            raise ValueError("--fall-onset-event-threshold는 0 초과 1 이하여야 합니다.")
        if not args.runtime_root.is_dir():
            raise FileNotFoundError(f"--runtime-root가 없습니다: {args.runtime_root}")
        if args.tcn_config is not None and not args.tcn_config.is_file():
            raise FileNotFoundError(f"--tcn-config 파일이 없습니다: {args.tcn_config}")
        if (
            args.fall_onset_tcn_config is not None
            and not args.fall_onset_tcn_config.is_file()
        ):
            raise FileNotFoundError(
                f"--fall-onset-tcn-config 파일이 없습니다: {args.fall_onset_tcn_config}"
            )
        if (
            args.fall_onset_tcn_ensemble_config is not None
            and not args.fall_onset_tcn_ensemble_config.is_file()
        ):
            raise FileNotFoundError(
                "--fall-onset-tcn-ensemble-config 파일이 없습니다: "
                f"{args.fall_onset_tcn_ensemble_config}"
            )
        if not args.driver_yolo_model.is_file():
            raise FileNotFoundError(f"--driver-yolo-model이 없습니다: {args.driver_yolo_model}")


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
