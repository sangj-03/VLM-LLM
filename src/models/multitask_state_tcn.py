#!/usr/bin/env python3
"""Train a 1.5 s TCN that directly predicts three driver states.

The shared temporal encoder has two supervised heads:

* state head: active / reduced / no_visible_response
* transition head: NVR begins within the configured future horizon

The transition target is derived only from timestamps in the same source video.
Current NVR windows are excluded from transition-head loss because they are no
longer an *impending* transition.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


CLASS_NAMES = ("active", "reduced", "no_visible_response")
ABSOLUTE_VALUE_FEATURES = (
    "head_pitch", "head_yaw", "head_roll",
    "pitch_velocity", "yaw_velocity", "roll_velocity",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int = 1) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size,
            dilation=dilation, padding=0,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(values, (self.left_padding, 0)))


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, 3, dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = CausalConv1d(channels, channels, 3, dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.dropout(F.relu(self.bn1(self.conv1(values))))
        values = self.dropout(F.relu(self.bn2(self.conv2(values))))
        return F.relu(values + residual)


class MultiTaskDriverTCN(nn.Module):
    """Causal shared encoder with state and impending-NVR heads."""

    def __init__(self, feature_dim: int, hidden_dim: int = 64,
                 num_classes: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(feature_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([
            TCNBlock(hidden_dim, dilation, dropout)
            for dilation in (1, 2, 4, 8, 16, 32)
        ])
        self.state_classifier = nn.Linear(hidden_dim, num_classes)
        self.transition_classifier = nn.Linear(hidden_dim, 1)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.input_projection(values.transpose(1, 2))
        for block in self.blocks:
            values = block(values)
        current = values[:, :, -1]
        return self.state_classifier(current), self.transition_classifier(current).squeeze(1)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for truth, prediction in zip(y_true, y_pred):
        matrix[int(truth), int(prediction)] += 1
    return matrix


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, object]:
    rows: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for index, name in enumerate(CLASS_NAMES):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        rows[name] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support": int(matrix[index, :].sum()),
        }
    total = int(matrix.sum())
    return {
        "accuracy": float(np.trace(matrix) / total) if total else 0.0,
        "macro_f1": float(np.mean(f1_values)),
        "per_class": rows,
        "confusion_matrix": matrix.tolist(),
    }


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray,
                   threshold: float) -> dict[str, float | int]:
    predictions = probabilities >= threshold
    truth = labels.astype(bool)
    tp = int(np.sum(truth & predictions))
    fp = int(np.sum(~truth & predictions))
    fn = int(np.sum(truth & ~predictions))
    tn = int(np.sum(~truth & ~predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def derive_transition_targets(dataset: np.lib.npyio.NpzFile,
                              horizon_sec: float) -> tuple[np.ndarray, np.ndarray]:
    labels = dataset["y_original"].astype(np.int64)
    times = dataset["target_time"].astype(np.float64)
    videos = np.asarray(dataset["video_id"]).astype(str)
    targets = np.zeros(len(labels), dtype=np.float32)
    eligible = labels != 2
    for video in np.unique(videos):
        indices = np.flatnonzero(videos == video)
        nvr_times = times[indices[labels[indices] == 2]]
        if not len(nvr_times):
            continue
        for index in indices[eligible[indices]]:
            now = times[index]
            targets[index] = float(np.any(
                (nvr_times > now) & (nvr_times <= now + horizon_sec + 1e-6)
            ))
    return targets, eligible.astype(np.float32)


def derive_operational_state_targets(
    dataset: np.lib.npyio.NpzFile, horizon_sec: float,
) -> np.ndarray:
    """Apply the declared reduced semantics to the supervised state target.

    Annotated reduced windows stay reduced. A still-responsive window followed
    by NVR within ``horizon_sec`` is also reduced (pre-loss-of-consciousness),
    rather than being trained as active and contradicting the declared class.
    """
    labels = dataset["y_original"].astype(np.int64).copy()
    transition, _ = derive_transition_targets(dataset, horizon_sec)
    labels[(labels != 2) & (transition >= 0.5)] = 1
    return labels


def apply_feature_transform(values: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Remove camera/sign convention from orientation and angular-speed inputs."""
    transformed = values.copy()
    for name in ABSOLUTE_VALUE_FEATURES:
        if name in feature_names:
            transformed[:, :, feature_names.index(name)] = np.abs(
                transformed[:, :, feature_names.index(name)]
            )
    return transformed


@torch.no_grad()
def predict(model: nn.Module, values: np.ndarray, device: torch.device,
            batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values.astype(np.float32))),
        batch_size=batch_size, shuffle=False,
    )
    state_probabilities: list[np.ndarray] = []
    transition_probabilities: list[np.ndarray] = []
    model.eval()
    for (batch,) in loader:
        state_logits, transition_logits = model(batch.to(device))
        state_probabilities.append(torch.softmax(state_logits, dim=1).cpu().numpy())
        transition_probabilities.append(torch.sigmoid(transition_logits).cpu().numpy())
    return np.concatenate(state_probabilities), np.concatenate(transition_probabilities)


def train_one(
    *, seed: int, train_data: np.lib.npyio.NpzFile,
    val_data: np.lib.npyio.NpzFile, output_path: Path,
    horizon_sec: float, epochs: int, batch_size: int, hidden_dim: int,
    dropout: float, learning_rate: float, patience: int,
    samples_per_epoch: int, transition_loss_weight: float,
) -> tuple[Path, np.ndarray, np.ndarray]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_names = [str(value) for value in train_data["feature_names"]]
    X_train = apply_feature_transform(train_data["X"].astype(np.float32), feature_names)
    X_val = apply_feature_transform(val_data["X"].astype(np.float32), feature_names)
    y_train = derive_operational_state_targets(train_data, horizon_sec)
    y_val = derive_operational_state_targets(val_data, horizon_sec)
    transition_train, eligible_train = derive_transition_targets(train_data, horizon_sec)
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    counts = np.bincount(y_train, minlength=3).astype(np.float64)
    sampling_target = np.asarray([0.40, 0.30, 0.30], dtype=np.float64)
    sample_weights = (sampling_target / np.maximum(counts, 1.0))[y_train]
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=samples_per_epoch, replacement=True, generator=generator,
    )
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train.astype(np.float32)),
            torch.from_numpy(y_train),
            torch.from_numpy(transition_train),
            torch.from_numpy(eligible_train),
        ),
        batch_size=batch_size, sampler=sampler,
    )
    model = MultiTaskDriverTCN(
        feature_dim=X_train.shape[2], hidden_dim=hidden_dim, dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    eligible_positive = float(np.sum(transition_train * eligible_train))
    eligible_negative = float(np.sum((1.0 - transition_train) * eligible_train))
    transition_pos_weight = min(30.0, eligible_negative / max(eligible_positive, 1.0))
    transition_criterion = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor(transition_pos_weight, dtype=torch.float32, device=device),
    )

    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        item_count = 0
        for values, labels, transition_labels, transition_mask in loader:
            values = values.to(device)
            labels = labels.to(device)
            transition_labels = transition_labels.to(device)
            transition_mask = transition_mask.to(device)
            optimizer.zero_grad()
            state_logits, transition_logits = model(values)
            state_loss = F.cross_entropy(state_logits, labels)
            transition_losses = transition_criterion(transition_logits, transition_labels)
            transition_loss = (
                (transition_losses * transition_mask).sum()
                / transition_mask.sum().clamp_min(1.0)
            )
            loss = state_loss + transition_loss_weight * transition_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(labels)
            item_count += len(labels)

        state_probs, _ = predict(model, X_val, device, batch_size)
        state_metrics = metrics_from_confusion(
            confusion_matrix(y_val, state_probs.argmax(axis=1))
        )
        score = float(state_metrics["macro_f1"])
        print(
            f"seed={seed} epoch={epoch:03d} loss={loss_sum/max(item_count, 1):.4f} "
            f"val_accuracy={state_metrics['accuracy']:.4f} val_macro_f1={score:.4f}",
            flush=True,
        )
        if score > best_score + 1e-9:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("학습된 TCN state가 없습니다.")
    model.load_state_dict(best_state)
    state_probs, transition_probs = predict(model, X_val, device, batch_size)
    final_metrics = metrics_from_confusion(confusion_matrix(y_val, state_probs.argmax(axis=1)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_type": "multitask_state_transition_tcn",
        "model_state_dict": best_state,
        "feature_dim": int(X_train.shape[2]),
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "feature_names": feature_names,
        "absolute_value_features": list(ABSOLUTE_VALUE_FEATURES),
        "class_names": list(CLASS_NAMES),
        "normalization_mean": mean.astype(np.float32),
        "normalization_std": std.astype(np.float32),
        "window_steps": int(X_train.shape[1]),
        "target_fps": 10.0,
        "early_warning_horizon_sec": float(horizon_sec),
        "transition_loss_weight": float(transition_loss_weight),
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "validation_metrics": final_metrics,
        "training_class_counts": counts.astype(int).tolist(),
        "state_target_policy": (
            "original reduced OR responsive window within early_warning_horizon_sec "
            "before NVR; NVR labels are unchanged"
        ),
        "reduced_semantics": (
            "conscious but not maintaining normal control, or visibly approaching loss "
            "of consciousness; hand-on-wheel is not directly measured by the current 32-D features"
        ),
    }, output_path)
    print(f"saved {output_path} (best epoch={best_epoch}, macro_f1={best_score:.4f})", flush=True)
    return output_path, state_probs, transition_probs


def select_threshold(labels: np.ndarray, probabilities: np.ndarray,
                     minimum_recall: float | None = None) -> tuple[float, dict[str, float | int]]:
    candidates: list[tuple[float, dict[str, float | int]]] = []
    for threshold in np.linspace(0.01, 0.99, 99):
        candidates.append((float(threshold), binary_metrics(labels, probabilities, threshold)))
    eligible = candidates
    if minimum_recall is not None:
        constrained = [item for item in candidates if float(item[1]["recall"]) >= minimum_recall]
        if constrained:
            eligible = constrained
    return max(eligible, key=lambda item: (float(item[1]["f1"]), float(item[1]["precision"])))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--ensemble-out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--horizon-sec", type=float, default=3.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--samples-per-epoch", type=int, default=900)
    parser.add_argument("--transition-loss-weight", type=float, default=0.20)
    args = parser.parse_args()

    train_data = np.load(args.train, allow_pickle=True)
    val_data = np.load(args.val, allow_pickle=True)
    if tuple(str(value) for value in train_data["original_class_names"]) != CLASS_NAMES:
        raise ValueError("원본 3상태 class 순서가 예상과 다릅니다.")
    if train_data["X"].shape[1:] != val_data["X"].shape[1:]:
        raise ValueError("train/val TCN 입력 shape가 다릅니다.")
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)

    state_members: list[np.ndarray] = []
    transition_members: list[np.ndarray] = []
    model_paths: list[Path] = []
    for seed in args.seeds:
        model_path = args.model_dir / f"multitask_state_tcn_1p5s_seed{seed}.pt"
        path, state_probs, transition_probs = train_one(
            seed=seed, train_data=train_data, val_data=val_data,
            output_path=model_path, horizon_sec=args.horizon_sec,
            epochs=args.epochs, batch_size=args.batch, hidden_dim=args.hidden,
            dropout=args.dropout, learning_rate=args.lr, patience=args.patience,
            samples_per_epoch=args.samples_per_epoch,
            transition_loss_weight=args.transition_loss_weight,
        )
        model_paths.append(path.resolve())
        state_members.append(state_probs)
        transition_members.append(transition_probs)

    mean_state = np.mean(np.stack(state_members), axis=0)
    mean_transition = np.mean(np.stack(transition_members), axis=0)
    y_val = derive_operational_state_targets(val_data, args.horizon_sec)
    transition_val, eligible_val = derive_transition_targets(val_data, args.horizon_sec)
    state_metrics = metrics_from_confusion(confusion_matrix(y_val, mean_state.argmax(axis=1)))
    nvr_threshold, nvr_metrics = select_threshold(
        (y_val == 2).astype(np.int64), mean_state[:, 2], minimum_recall=0.90,
    )
    eligible = eligible_val.astype(bool)
    transition_threshold, transition_metrics = select_threshold(
        transition_val[eligible].astype(np.int64), mean_transition[eligible],
    )
    transition_operational = bool(
        float(transition_metrics["f1"]) >= 0.30
        and float(transition_metrics["precision"]) >= 0.20
    )
    minimum_state_f1 = min(
        float(values["f1"])
        for values in state_metrics["per_class"].values()
    )
    ensemble = {
        "ensemble_type": "mean_multitask_state_probability",
        "model_paths": [str(path) for path in model_paths],
        "class_names": list(CLASS_NAMES),
        "window_steps": int(val_data["X"].shape[1]),
        "target_fps": 10.0,
        "num_models": len(model_paths),
        "selected_threshold": nvr_threshold,
        "transition_threshold": transition_threshold,
        "early_warning_horizon_sec": float(args.horizon_sec),
        "validation_file": str(args.val.resolve()),
        "validation_metrics": state_metrics,
        "nvr_gate_validation_metrics": nvr_metrics,
        "transition_validation_metrics": transition_metrics,
        "transition_operational": transition_operational,
        "vlm_state_distribution_operational": bool(minimum_state_f1 >= 0.20),
        "state_decision": "argmax_mean_softmax_no_manual_state_thresholds",
        "limitations": [
            "reduced training examples remain sparse even after adding pre-NVR windows",
            "the current 32-D input has no direct hand-on-wheel feature",
        ],
    }
    args.ensemble_out.parent.mkdir(parents=True, exist_ok=True)
    args.ensemble_out.write_text(
        json.dumps(ensemble, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metrics_path = args.analysis_dir / "ensemble_validation_metrics.json"
    metrics_path.write_text(
        json.dumps(ensemble, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    predictions_path = args.analysis_dir / "ensemble_val_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "video_id", "target_time", "original_ground_truth", "operational_ground_truth", "prediction",
            "p_active", "p_reduced", "p_no_visible_response",
            "transition_target", "transition_probability",
        ])
        for index in range(len(y_val)):
            writer.writerow([
                str(val_data["video_id"][index]), float(val_data["target_time"][index]),
                CLASS_NAMES[int(val_data["y_original"][index])], CLASS_NAMES[y_val[index]],
                CLASS_NAMES[int(mean_state[index].argmax())],
                *[float(value) for value in mean_state[index]],
                int(transition_val[index]), float(mean_transition[index]),
            ])
    print(json.dumps(ensemble, ensure_ascii=False, indent=2), flush=True)
    print(f"ensemble saved: {args.ensemble_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
