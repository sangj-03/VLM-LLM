#!/usr/bin/env python3
"""Train a separate 1.5 s control-engagement TCN.

This is intentionally *not* a no-visible-response model.  Its target is a
weak but auditable geometry label from Drive&Act: whether a wrist is near the
annotated steering-wheel model.  The resulting score is therefore called a
``control_disengaged_candidate`` until it is calibrated on this project's own
2-D cabin videos with direct hand/control labels.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


CLASS_NAMES = ("control_engaged", "control_disengaged")
FEATURE_DIM = 13
WINDOW_STEPS = 15
DEFAULT_DILATIONS = (1, 2, 4)


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CausalConv1d(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(values, (self.padding, 0)))


class ControlBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv = CausalConv1d(channels, dilation)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.conv(values).transpose(1, 2)
        values = self.norm(values).transpose(1, 2)
        return F.relu(residual + self.dropout(F.relu(values)))


class ControlEngagementTCN(nn.Module):
    """Causal 13-D wrist TCN; RF=15 frames, exactly the supplied 1.5 s."""

    def __init__(
        self, feature_dim: int = FEATURE_DIM, hidden_dim: int = 32,
        dropout: float = 0.15, dilations: tuple[int, ...] = DEFAULT_DILATIONS,
    ) -> None:
        super().__init__()
        self.dilations = tuple(int(item) for item in dilations)
        self.input_projection = nn.Conv1d(feature_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([
            ControlBlock(hidden_dim, dilation, dropout) for dilation in self.dilations
        ])
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.input_projection(values.transpose(1, 2))
        for block in self.blocks:
            encoded = block(encoded)
        representation = 0.5 * encoded[:, :, -1] + 0.5 * encoded.mean(dim=2)
        return self.classifier(representation)


def binary_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    tp = int(np.sum((truth == 1) & (prediction == 1)))
    fp = int(np.sum((truth == 0) & (prediction == 1)))
    fn = int(np.sum((truth == 1) & (prediction == 0)))
    tn = int(np.sum((truth == 0) & (prediction == 0)))
    precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    accuracy = (tp + tn) / max(1, len(truth))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision,
            "recall": recall, "f1": f1, "accuracy": accuracy}


@torch.inference_mode()
def predict(model: nn.Module, values: np.ndarray, device: torch.device, batch: int) -> np.ndarray:
    model.eval(); chunks: list[np.ndarray] = []
    for start in range(0, len(values), batch):
        logits = model(torch.from_numpy(values[start:start + batch]).to(device))
        chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(chunks)


def train_one(args: argparse.Namespace, seed: int, train: dict[str, np.ndarray], val: dict[str, np.ndarray], path: Path):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = train["X"].mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = train["X"].std(axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    X_train = ((train["X"].astype(np.float32) - mean) / std).astype(np.float32)
    X_val = ((val["X"].astype(np.float32) - mean) / std).astype(np.float32)
    y_train, y_val = train["y_control"].astype(np.int64), val["y_control"].astype(np.int64)
    counts = np.bincount(y_train, minlength=2).astype(np.float64)
    sample_weights = 1.0 / np.maximum(counts[y_train], 1.0)
    sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), args.samples_per_epoch,
                                    replacement=True, generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                        batch_size=args.batch, sampler=sampler)
    model = ControlEngagementTCN(hidden_dim=args.hidden, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_state, best_score, best_epoch, stale = None, -1.0, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; seen = 0
        for values, labels in loader:
            optimizer.zero_grad(); logits = model(values.to(device))
            loss = F.cross_entropy(logits, labels.to(device), label_smoothing=0.02)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            total += float(loss.item()) * len(labels); seen += len(labels)
        probabilities = predict(model, X_val, device, args.batch)
        metrics = binary_metrics(y_val, probabilities.argmax(axis=1))
        print(f"seed={seed} epoch={epoch:03d} loss={total/max(1, seen):.4f} val_disengaged_f1={metrics['f1']:.4f}", flush=True)
        if metrics["f1"] > best_score + 1e-9:
            best_state, best_score, best_epoch, stale = copy.deepcopy(model.state_dict()), float(metrics["f1"]), epoch, 0
        else:
            stale += 1
            if stale >= args.patience: break
    if best_state is None: raise RuntimeError("no trained state")
    model.load_state_dict(best_state)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_type": "control_engagement_tcn", "model_state_dict": best_state,
        "feature_dim": FEATURE_DIM, "hidden_dim": args.hidden, "dropout": args.dropout,
        "feature_names": train["feature_names"].tolist(), "class_names": list(CLASS_NAMES),
        "normalization_mean": mean, "normalization_std": std, "window_steps": WINDOW_STEPS,
        "target_fps": 10.0, "dilations": list(DEFAULT_DILATIONS), "receptive_field_frames": 15,
        "pooling": "0.5_final_timestep_plus_0.5_temporal_mean", "seed": seed,
        "best_epoch": best_epoch, "validation_metrics": binary_metrics(y_val, predict(model, X_val, device, args.batch).argmax(axis=1)),
        "training_class_counts": counts.astype(int).tolist(),
        "label_scope": "DriveAct_3d_geometry_weak_control_engagement_only",
    }, path)
    return model, mean, std


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True); parser.add_argument("--val", type=Path, required=True); parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True); parser.add_argument("--ensemble-out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True); parser.add_argument("--epochs", type=int, default=45); parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch", type=int, default=128); parser.add_argument("--samples-per-epoch", type=int, default=2400); parser.add_argument("--hidden", type=int, default=32); parser.add_argument("--dropout", type=float, default=0.15); parser.add_argument("--lr", type=float, default=8e-4)
    args = parser.parse_args()
    train = dict(np.load(args.train, allow_pickle=True)); val = dict(np.load(args.val, allow_pickle=True)); test = dict(np.load(args.test, allow_pickle=True))
    for name, data in (("train", train), ("val", val), ("test", test)):
        if data["X"].shape[1:] != (WINDOW_STEPS, FEATURE_DIM): raise ValueError(f"{name} must be [N,15,13]")
        if tuple(data["feature_names"].tolist()) != tuple(train["feature_names"].tolist()): raise ValueError(f"{name} feature order differs")
    paths: list[str] = []; test_members: list[np.ndarray] = []
    for seed in args.seeds:
        path = args.model_dir / f"control_engagement_tcn_1p5s_seed{seed}.pt"
        model, mean, std = train_one(args, seed, train, val, path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        test_members.append(predict(model, ((test["X"].astype(np.float32) - mean) / std).astype(np.float32), device, args.batch))
        paths.append(str(path.resolve()))
    test_probability = np.mean(np.stack(test_members), axis=0)
    artifact = {
        "ensemble_type": "mean_control_engagement_probability", "model_module": str(Path(__file__).resolve()),
        "model_paths": paths, "class_names": list(CLASS_NAMES), "feature_names": train["feature_names"].tolist(),
        "window_steps": WINDOW_STEPS, "target_fps": 10.0, "num_models": len(paths), "selected_threshold": 0.50,
        "control_operational": True, "reduced_confirmation_windows": 5,
        "runtime_decision": "observable_active_and_sustained_control_disengagement_becomes_reduced",
        "test_metrics_at_argmax": binary_metrics(test["y_control"].astype(np.int64), test_probability.argmax(axis=1)),
        "test_metrics_at_fixed_threshold_0p50": binary_metrics(test["y_control"].astype(np.int64), (test_probability[:, 1] >= 0.50).astype(np.int64)),
        "provenance": {"dataset": "Drive&Act", "split": "official split 0 participant-disjoint", "label": "3-D wrist-to-wheel geometry weak label"},
        "limitations": ["This does not measure consciousness or medical impairment.", "Training uses Drive&Act 3-D pose while the live runtime uses 2-D MediaPipe features; reduced therefore means visual control disengagement."],
    }
    args.ensemble_out.parent.mkdir(parents=True, exist_ok=True)
    args.ensemble_out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
