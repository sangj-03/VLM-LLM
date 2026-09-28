"""Eye-state specialist shared by training and the real-time monitor.

The model deliberately answers a narrower question than the driver TCN:
whether the eyes are open, closed, or not reliably observable.  Treating
``unknown`` as its own class prevents a profile view or an occluding hand from
becoming a false closed-eye signal.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


EYE_STATE_NAMES = ("open", "closed", "unknown")
EYE_STATE_TO_ID = {name: index for index, name in enumerate(EYE_STATE_NAMES)}
BINARY_EYE_STATE_NAMES = ("open", "closed")
DEFAULT_IMAGE_MEAN = (0.485, 0.456, 0.406)
DEFAULT_IMAGE_STD = (0.229, 0.224, 0.225)
DEFAULT_MIN_EYE_CONTRAST = 15.0
DEFAULT_MIN_EYE_SHARPNESS = 8.0

# Face Mesh contours, not iris landmarks.  They remain available when refined
# iris landmarks are disabled and are stable enough to define a generous crop.
LEFT_EYE_IDS = (33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7)
RIGHT_EYE_IDS = (263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249)


def build_eye_state_model(num_classes: int = len(EYE_STATE_NAMES), pretrained: bool = False) -> Any:
    """Create the small RGB eye-pair classifier used by this project."""
    import torch.nn as nn
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _eye_box(
    landmarks: Any,
    ids: tuple[int, ...],
    width: int,
    height: int,
    margin: float = 0.70,
) -> tuple[int, int, int, int] | None:
    if landmarks is None or max(ids) >= len(landmarks):
        return None
    points = np.asarray(
        [(float(landmarks[index].x), float(landmarks[index].y)) for index in ids],
        dtype=np.float32,
    )
    if not np.isfinite(points).all():
        return None
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = (lo + hi) * 0.5
    size = np.maximum(hi - lo, np.asarray((1.0 / width, 1.0 / height), dtype=np.float32))
    # A wide vertical margin retains eyelid texture and a little eyebrow
    # context, but does not make this a full-face classifier.
    half = size * (0.5 + margin)
    x1 = int(np.floor((center[0] - half[0]) * width))
    y1 = int(np.floor((center[1] - half[1]) * height))
    x2 = int(np.ceil((center[0] + half[0]) * width))
    y2 = int(np.ceil((center[1] + half[1]) * height))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    return (x1, y1, x2, y2) if x2 - x1 >= 4 and y2 - y1 >= 4 else None


def make_eye_pair_crop(
    rgb: np.ndarray,
    landmarks: Any,
    output_size: tuple[int, int] = (192, 64),
) -> np.ndarray | None:
    """Return one canonical RGB image containing left and right eye crops."""
    import cv2

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return None
    height, width = rgb.shape[:2]
    left_box = _eye_box(landmarks, LEFT_EYE_IDS, width, height)
    right_box = _eye_box(landmarks, RIGHT_EYE_IDS, width, height)
    if left_box is None or right_box is None:
        return None
    eye_images = []
    for x1, y1, x2, y2 in (left_box, right_box):
        eye_images.append(cv2.resize(rgb[y1:y2, x1:x2], (output_size[0] // 2, output_size[1]), interpolation=cv2.INTER_LINEAR))
    return np.concatenate(eye_images, axis=1)


def eye_crop_quality(
    crop: np.ndarray | None,
    min_contrast: float = DEFAULT_MIN_EYE_CONTRAST,
    min_sharpness: float = DEFAULT_MIN_EYE_SHARPNESS,
) -> dict[str, float | int]:
    """Measure whether an eye crop contains usable eyelid texture.

    The score is deliberately a gate, not an eye-closure estimate: low detail
    from motion blur, occlusion, or a tiny face must become ``unknown`` rather
    than a false closed-eye probability.
    """
    import cv2

    if crop is None or crop.size == 0:
        return {"eye_crop_contrast": 0.0, "eye_crop_sharpness": 0.0, "eye_crop_quality_valid": 0}
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    valid = int(
        np.isfinite(contrast)
        and np.isfinite(sharpness)
        and contrast >= float(min_contrast)
        and sharpness >= float(min_sharpness)
    )
    return {
        "eye_crop_contrast": contrast if np.isfinite(contrast) else 0.0,
        "eye_crop_sharpness": sharpness if np.isfinite(sharpness) else 0.0,
        "eye_crop_quality_valid": valid,
    }


class EyeStateSpecialist:
    """Causal eye-state inference with an explicit unknown fallback."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: Any | None = None,
        smooth_steps: int = 10,
        closed_probability_threshold: float = 0.70,
        closed_persist_steps: int = 6,
        min_visible_fraction: float = 0.60,
        min_contrast: float = DEFAULT_MIN_EYE_CONTRAST,
        min_sharpness: float = DEFAULT_MIN_EYE_SHARPNESS,
    ) -> None:
        import torch

        self.torch = torch
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        class_names = tuple(str(name) for name in checkpoint.get("class_names", EYE_STATE_NAMES))
        if class_names not in (EYE_STATE_NAMES, BINARY_EYE_STATE_NAMES):
            raise RuntimeError(
                "눈 상태 checkpoint class_names는 "
                f"{BINARY_EYE_STATE_NAMES} 또는 {EYE_STATE_NAMES} 순서여야 합니다: {class_names}"
            )
        self.class_names = class_names
        self.model = build_eye_state_model(len(class_names)).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.image_mean = np.asarray(checkpoint.get("image_mean", DEFAULT_IMAGE_MEAN), dtype=np.float32).reshape(3, 1, 1)
        self.image_std = np.asarray(checkpoint.get("image_std", DEFAULT_IMAGE_STD), dtype=np.float32).reshape(3, 1, 1)
        if np.any(self.image_std <= 0):
            raise RuntimeError("눈 상태 checkpoint image_std가 올바르지 않습니다.")
        self.output_size = tuple(int(value) for value in checkpoint.get("image_size", (192, 64)))
        self.history: deque[np.ndarray] = deque(maxlen=max(1, int(smooth_steps)))
        self.visibility_history: deque[int] = deque(maxlen=max(1, int(smooth_steps)))
        self.closed_probability_threshold = float(closed_probability_threshold)
        self.closed_persist_steps = max(1, int(closed_persist_steps))
        self.min_visible_fraction = float(min_visible_fraction)
        self.min_contrast = float(min_contrast)
        self.min_sharpness = float(min_sharpness)
        self.closed_steps = 0

    def _unknown(self, quality: dict[str, float | int] | None = None) -> dict[str, float | int]:
        self.history.clear()
        self.visibility_history.append(0)
        self.closed_steps = 0
        return {
            "eye_open_probability": 0.0,
            "eye_closed_probability": 0.0,
            "eye_unknown_probability": 1.0,
            "eye_state_valid": 0,
            "eye_visible_fraction": float(np.mean(self.visibility_history)),
            "eye_closed_sustained": 0,
            **(quality or {"eye_crop_contrast": 0.0, "eye_crop_sharpness": 0.0, "eye_crop_quality_valid": 0}),
        }

    def predict(self, rgb: np.ndarray, landmarks: Any) -> dict[str, float | int]:
        crop = make_eye_pair_crop(rgb, landmarks, self.output_size)
        quality = eye_crop_quality(crop, self.min_contrast, self.min_sharpness)
        if not int(quality["eye_crop_quality_valid"]):
            return self._unknown(quality)
        image = crop.astype(np.float32).transpose(2, 0, 1) / 255.0
        image = (image - self.image_mean) / self.image_std
        with self.torch.inference_mode():
            logits = self.model(self.torch.from_numpy(image).unsqueeze(0).to(self.device))
            probabilities = self.torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        self.history.append(probabilities)
        self.visibility_history.append(1)
        probabilities = np.mean(np.stack(self.history), axis=0)
        values = {name: float(probabilities[index]) for index, name in enumerate(self.class_names)}
        unknown_probability = values.get("unknown", 0.0)
        visible_fraction = float(np.mean(self.visibility_history))
        closed_confirmed = bool(
            values["closed"] >= self.closed_probability_threshold
            and unknown_probability < 0.50
            and visible_fraction >= self.min_visible_fraction
        )
        self.closed_steps = self.closed_steps + 1 if closed_confirmed else 0
        return {
            "eye_open_probability": values["open"],
            "eye_closed_probability": values["closed"],
            # A binary checkpoint is deliberately only used when a valid crop
            # exists.  Crop extraction failure is handled above as unknown.
            "eye_unknown_probability": unknown_probability,
            "eye_state_valid": 1,
            "eye_visible_fraction": visible_fraction,
            "eye_closed_sustained": int(self.closed_steps >= self.closed_persist_steps),
            **quality,
        }


def fuse_eye_and_tcn_risk(
    tcn_probability: float,
    eye_closed_probability: float,
    eye_unknown_probability: float,
    eye_weight: float,
    eye_closed_sustained: int = 0,
    eye_visible_fraction: float = 0.0,
    min_visible_fraction: float = 0.60,
) -> float:
    """Conservative noisy-OR fusion; only sustained, visible closure adds risk."""
    tcn = float(np.clip(tcn_probability, 0.0, 1.0))
    closed = float(np.clip(eye_closed_probability, 0.0, 1.0))
    known_eye = 1.0 - float(np.clip(eye_unknown_probability, 0.0, 1.0))
    eye_usable = int(eye_closed_sustained) and eye_visible_fraction >= min_visible_fraction
    eye_risk = float(np.clip(eye_weight, 0.0, 1.0)) * closed * known_eye * int(eye_usable)
    return float(1.0 - (1.0 - tcn) * (1.0 - eye_risk))
