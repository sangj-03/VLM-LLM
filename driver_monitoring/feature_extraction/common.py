from pathlib import Path
import json
import numpy as np

FEATURE_NAMES = [
    "ear_left",
    "ear_right",
    "ear_mean",
    "eye_valid",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "pitch_velocity",
    "yaw_velocity",
    "roll_velocity",
    "head_motion",
    "neck_flexion_proxy",
    "neck_lateral_proxy",
    "neck_velocity",
    "left_hand_motion",
    "right_hand_motion",
    "left_hand_valid",
    "right_hand_valid",
    "face_valid",
    "face_reliability",
    "pose_reliability",
    "hand_reliability",
]

BINARY_FEATURES = {
    "eye_valid", "left_hand_valid", "right_hand_valid", "face_valid"
}


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def clip_finite(x, lo=-1e6, hi=1e6, default=0.0):
    if x is None or not np.isfinite(x):
        return float(default)
    return float(np.clip(x, lo, hi))
