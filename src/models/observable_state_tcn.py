#!/usr/bin/env python3
"""Train the v38 observable-state TCN for a 1.5 s / 10 Hz input.

The model is deliberately sized to the available temporal context:

* three causal residual blocks, dilations 1/2/4, one k=3 convolution each
* receptive field = 1 + 2 * (1 + 2 + 4) = 15 frames = 1.5 seconds
* final representation = 0.5 * final timestep + 0.5 * temporal mean

``no_visible_response`` remains a video-observation label, not a medical
diagnosis.  The transition head is retained only as an experimental log; its
loss weight is zero in the v38 training command because six incidents cannot
support an early-warning claim.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


CLASS_NAMES = ("active", "reduced", "no_visible_response")
ABSOLUTE_VALUE_FEATURES = (
    "head_pitch", "head_yaw", "head_roll",
    "pitch_velocity", "yaw_velocity", "roll_velocity",
)
DEFAULT_DILATIONS = (1, 2, 4)


def load_v36_helpers():
    path = Path(__file__).with_name("multitask_state_tcn_event_balanced.py")
    spec = importlib.util.spec_from_file_location("v36_training_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V36 = load_v36_helpers()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CausalConv1d(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = 2 * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(values, (self.left_padding, 0)))


class TimewiseLayerNorm(nn.Module):
    """Normalize channels independently at each timestamp (no future leakage)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.norm(values.transpose(1, 2)).transpose(1, 2)


class ShallowTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv = CausalConv1d(channels, dilation)
        self.norm = TimewiseLayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.conv(values)
        values = self.norm(values)
        values = self.dropout(F.relu(values))
        return F.relu(values + residual)


class MultiTaskDriverTCN(nn.Module):
    """15-frame receptive-field causal TCN with robust temporal pooling."""

    def __init__(
        self, feature_dim: int, hidden_dim: int = 48, num_classes: int = 3,
        dropout: float = 0.15, dilations: tuple[int, ...] = DEFAULT_DILATIONS,
    ) -> None:
        super().__init__()
        self.dilations = tuple(int(value) for value in dilations)
        self.input_projection = nn.Conv1d(feature_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([
            ShallowTCNBlock(hidden_dim, dilation, dropout)
            for dilation in self.dilations
        ])
        self.state_classifier = nn.Linear(hidden_dim, num_classes)
        self.transition_classifier = nn.Linear(hidden_dim, 1)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.input_projection(values.transpose(1, 2))
        for block in self.blocks:
            encoded = block(encoded)
        representation = 0.5 * encoded[:, :, -1] + 0.5 * encoded.mean(dim=2)
        return (
            self.state_classifier(representation),
            self.transition_classifier(representation).squeeze(1),
        )


def receptive_field_frames(dilations: tuple[int, ...]) -> int:
    return 1 + 2 * sum(dilations)


def apply_feature_transform(values: np.ndarray, feature_names: list[str]) -> np.ndarray:
    transformed = values.copy()
    for name in ABSOLUTE_VALUE_FEATURES:
        if name in feature_names:
            index = feature_names.index(name)
            transformed[:, :, index] = np.abs(transformed[:, :, index])
    return transformed


def observation_quality(values: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Return deterministic observation quality for [N,T,D] raw features.

    It answers whether the face/eyes/pose are sufficiently visible, not whether
    the person is conscious.  It is deliberately independent of state labels.
    """
    def series(name: str, default: float = 0.0) -> np.ndarray:
        if name not in feature_names:
            return np.full(values.shape[:2], default, dtype=np.float32)
        return np.clip(values[:, :, feature_names.index(name)], 0.0, 1.0)

    eye = series("eye_valid").mean(axis=1)
    face = series("face_valid").mean(axis=1)
    blend = series("eye_blendshape_valid").mean(axis=1)
    face_reliability = series("face_reliability").mean(axis=1)
    pose = series("pose_reliability").mean(axis=1)
    return (0.25 * eye + 0.25 * face + 0.15 * blend + 0.20 * face_reliability + 0.15 * pose)


class AugmentedWindows(Dataset):
    def __init__(
        self, values: np.ndarray, labels: np.ndarray, transitions: np.ndarray,
        mask: np.ndarray, continuous_indices: list[int], jitter: float, mask_probability: float,
    ) -> None:
        self.values = torch.from_numpy(values.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.transitions = torch.from_numpy(transitions.astype(np.float32))
        self.mask = torch.from_numpy(mask.astype(np.float32))
        self.continuous_indices = continuous_indices
        self.jitter = jitter
        self.mask_probability = mask_probability

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        values = self.values[index].clone()
        if self.jitter and self.continuous_indices:
            values[:, self.continuous_indices] += torch.randn_like(
                values[:, self.continuous_indices]
            ) * self.jitter
        if self.mask_probability and torch.rand(()) < self.mask_probability:
            values[torch.randint(values.shape[0], ()).item()] = 0.0
        return values, self.labels[index], self.transitions[index], self.mask[index]


@torch.inference_mode()
def predict(model: nn.Module, values: np.ndarray, device: torch.device, batch_size: int):
    loader = DataLoader(torch.from_numpy(values.astype(np.float32)), batch_size=batch_size)
    states, transitions = [], []
    model.eval()
    for batch in loader:
        state_logits, transition_logits = model(batch.to(device))
        states.append(torch.softmax(state_logits, dim=1).cpu().numpy())
        transitions.append(torch.sigmoid(transition_logits).cpu().numpy())
    return np.concatenate(states), np.concatenate(transitions)


def train_one(args, seed: int, train_data, val_data, output: Path):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = [str(value) for value in train_data["feature_names"]]
    raw_train = train_data["X"].astype(np.float32)
    raw_val = val_data["X"].astype(np.float32)
    train_quality = observation_quality(raw_train, names)
    val_quality = observation_quality(raw_val, names)
    X_train = apply_feature_transform(raw_train, names)
    X_val = apply_feature_transform(raw_val, names)
    y_train = train_data["y_original"].astype(np.int64)
    y_val = val_data["y_original"].astype(np.int64)
    transition_train, eligible_train = V36.derive_transition_targets(train_data, args.horizon_sec)
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    X_train, X_val = (X_train - mean) / std, (X_val - mean) / std
    sampling_target = np.asarray(args.sampling_target, dtype=np.float64)
    weights, sampling_audit = V36.event_balanced_sample_weights(train_data, y_train, sampling_target)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), args.samples_per_epoch,
        replacement=True, generator=torch.Generator().manual_seed(seed),
    )
    binary = {"eye_valid", "eye_closed", "face_valid", "eye_blendshape_valid"}
    continuous = [index for index, name in enumerate(names) if name not in binary]
    loader = DataLoader(
        AugmentedWindows(X_train, y_train, transition_train, eligible_train, continuous,
                         args.jitter_std, args.time_mask_probability),
        batch_size=args.batch, sampler=sampler,
    )
    model = MultiTaskDriverTCN(
        feature_dim=X_train.shape[2], hidden_dim=args.hidden, dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    positives = float(np.sum(transition_train * eligible_train))
    negatives = float(np.sum((1.0 - transition_train) * eligible_train))
    transition_loss = nn.BCEWithLogitsLoss(
        reduction="none", pos_weight=torch.tensor(min(30.0, negatives / max(1.0, positives)), device=device),
    )
    best_score, best_epoch, best_state, stale = -1.0, 0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = 0.0; count = 0
        for values, labels, targets, mask in loader:
            values, labels = values.to(device), labels.to(device)
            targets, mask = targets.to(device), mask.to(device)
            optimizer.zero_grad()
            state_logits, transition_logits = model(values)
            state_loss = F.cross_entropy(state_logits, labels, label_smoothing=args.label_smoothing)
            auxiliary_loss = (transition_loss(transition_logits, targets) * mask).sum() / mask.sum().clamp_min(1.0)
            loss = state_loss + args.transition_loss_weight * auxiliary_loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            total_loss += float(loss.item()) * len(labels); count += len(labels)
        state_probabilities, _ = predict(model, X_val, device, args.batch)
        metrics = V36.metrics_from_confusion(V36.confusion_matrix(y_val, state_probabilities.argmax(axis=1)))
        score = float(metrics["macro_f1"])
        print(f"seed={seed} epoch={epoch:03d} loss={total_loss/max(1,count):.4f} macro_f1={score:.4f}", flush=True)
        if score > best_score + 1e-9:
            best_score, best_epoch, best_state, stale = score, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("no trained state")
    model.load_state_dict(best_state)
    states, transitions = predict(model, X_val, device, args.batch)
    metrics = V36.metrics_from_confusion(V36.confusion_matrix(y_val, states.argmax(axis=1)))
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_type": "observable_state_tcn", "model_state_dict": best_state,
        "feature_dim": int(X_train.shape[2]), "hidden_dim": int(args.hidden),
        "dropout": float(args.dropout), "feature_names": names,
        "absolute_value_features": list(ABSOLUTE_VALUE_FEATURES),
        "class_names": list(CLASS_NAMES), "normalization_mean": mean.astype(np.float32),
        "normalization_std": std.astype(np.float32), "window_steps": int(X_train.shape[1]),
        "target_fps": 10.0, "dilations": list(DEFAULT_DILATIONS),
        "convolutions_per_block": 1, "receptive_field_frames": receptive_field_frames(DEFAULT_DILATIONS),
        "pooling": "0.5_final_timestep_plus_0.5_temporal_mean",
        "early_warning_horizon_sec": float(args.horizon_sec),
        "transition_loss_weight": float(args.transition_loss_weight), "seed": int(seed),
        "best_epoch": int(best_epoch), "validation_metrics": metrics,
        "training_class_counts": np.bincount(y_train, minlength=3).astype(int).tolist(),
        "sampling_audit": sampling_audit,
        "quality_summary": {"train_mean": float(train_quality.mean()), "val_mean": float(val_quality.mean())},
        "state_target_policy": "original_labels_only_no_pre_nvr_relabel",
    }, output)
    return states, transitions, metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True); parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True); parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--ensemble-out", type=Path, required=True); parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=60); parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=48); parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=8e-4); parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--samples-per-epoch", type=int, default=1800); parser.add_argument("--horizon-sec", type=float, default=3.0)
    parser.add_argument("--transition-loss-weight", type=float, default=0.0)
    parser.add_argument("--jitter-std", type=float, default=0.025); parser.add_argument("--time-mask-probability", type=float, default=0.20)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--sampling-target", type=float, nargs=3, default=(0.45, 0.35, 0.20))
    parser.add_argument("--minimum-observation-quality", type=float, default=0.55)
    parser.add_argument(
        "--control-ensemble-path", type=Path, default=None,
        help="optional control-engagement ensemble; logging only until target-camera calibration",
    )
    args = parser.parse_args()
    if not np.isclose(sum(args.sampling_target), 1.0):
        raise ValueError("--sampling-target must sum to 1")
    train_data, val_data = np.load(args.train, allow_pickle=True), np.load(args.val, allow_pickle=True)
    if tuple(map(str, train_data["original_class_names"])) != CLASS_NAMES:
        raise ValueError("unexpected class order")
    if train_data["X"].shape[1:] != val_data["X"].shape[1:]:
        raise ValueError("train/validation input shapes differ")
    if train_data["X"].shape[1] != receptive_field_frames(DEFAULT_DILATIONS):
        raise ValueError("v38 is intentionally fixed to 15 frames")
    args.model_dir.mkdir(parents=True, exist_ok=True); args.analysis_dir.mkdir(parents=True, exist_ok=True)
    state_members, transition_members, paths = [], [], []
    for seed in args.seeds:
        path = args.model_dir / f"observable_state_tcn_1p5s_seed{seed}.pt"
        states, transitions, _ = train_one(args, seed, train_data, val_data, path)
        paths.append(path.resolve()); state_members.append(states); transition_members.append(transitions)
    mean_state, mean_transition = np.mean(np.stack(state_members), axis=0), np.mean(np.stack(transition_members), axis=0)
    truth = val_data["y_original"].astype(np.int64)
    metrics = V36.metrics_from_confusion(V36.confusion_matrix(truth, mean_state.argmax(axis=1)))
    # This selection is reported only as a holdout diagnostic.  It is not
    # promoted to the operational alarm threshold because the same S01 holdout
    # must not both choose a threshold and claim independent performance.
    validation_selected_threshold, validation_selected_metrics = V36.select_threshold(
        (truth == 2).astype(np.int64), mean_state[:, 2], minimum_recall=0.90
    )
    operational_alarm_threshold = 0.50
    operational_alarm_metrics = V36.binary_metrics(
        (truth == 2).astype(np.int64), mean_state[:, 2], operational_alarm_threshold
    )
    transition_truth, eligible = V36.derive_transition_targets(val_data, args.horizon_sec)
    transition_threshold, transition_metrics = V36.select_threshold(transition_truth[eligible.astype(bool)].astype(np.int64), mean_transition[eligible.astype(bool)])
    quality = observation_quality(val_data["X"].astype(np.float32), [str(x) for x in val_data["feature_names"]])
    observed = quality >= args.minimum_observation_quality
    ensemble = {
        "ensemble_type": "mean_observable_state_probability", "model_module": str(Path(__file__).resolve()),
        "model_paths": [str(path) for path in paths], "class_names": list(CLASS_NAMES), "window_steps": 15,
        "target_fps": 10.0, "num_models": len(paths), "selected_threshold": operational_alarm_threshold,
        "early_warning_horizon_sec": float(args.horizon_sec), "transition_threshold": transition_threshold,
        "transition_operational": False,
        # The old three-class head remains logged for audit, but its reduced
        # class has no usable held-out F1.  It must not become an operational
        # reduced decision until the separate control branch is calibrated on
        # target-camera hand/control labels.
        "state_decision": "nvr_argmax_when_observable_else_unobservable",
        "operational_state_policy": "nvr_argmax_then_active_control_branch_required_for_reduced",
        "reduced_head_operational": False,
        "control_ensemble_path": (
            str(args.control_ensemble_path.resolve())
            if args.control_ensemble_path is not None else None
        ),
        "minimum_observation_quality": float(args.minimum_observation_quality),
        "minimum_state_confidence": 0.0, "validation_file": str(args.val.resolve()),
        "validation_metrics": metrics,
        "nvr_gate_validation_metrics_at_fixed_operational_threshold": operational_alarm_metrics,
        "validation_threshold_diagnostic": {
            "selected_on_s01_only_not_operational": validation_selected_threshold,
            "metrics": validation_selected_metrics,
        },
        "transition_validation_metrics": transition_metrics,
        "observation_quality_validation": {"coverage": float(observed.mean()), "mean": float(quality.mean())},
        "architecture": {"dilations": list(DEFAULT_DILATIONS), "convolutions_per_block": 1,
                         "receptive_field_frames": 15, "pooling": "0.5_final_plus_0.5_mean"},
        "limitations": [
            "NVR has six independent training incidents from three subjects; this is an observed no-response proxy, not medically verified loss of consciousness.",
            "The state-head reduced probability is diagnostic-only. A reduced decision requires a separately validated control-engagement branch and target-camera calibration.",
            "transition probability is logged only; its validation evidence is insufficient for warning control.",
        ],
    }
    args.ensemble_out.write_text(json.dumps(ensemble, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.analysis_dir / "ensemble_validation_metrics.json").write_text(json.dumps(ensemble, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.analysis_dir / "ensemble_val_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["video_id", "target_time", "ground_truth", "prediction", "p_active", "p_reduced", "p_no_visible_response", "observation_quality", "observable", "transition_probability"])
        for index in range(len(truth)):
            writer.writerow([str(val_data["video_id"][index]), float(val_data["target_time"][index]), CLASS_NAMES[truth[index]], CLASS_NAMES[int(mean_state[index].argmax())], *map(float, mean_state[index]), float(quality[index]), int(observed[index]), float(mean_transition[index])])
    print(json.dumps(ensemble, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
