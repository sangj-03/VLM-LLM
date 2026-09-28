import argparse
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (
    DataLoader,
    TensorDataset,
    WeightedRandomSampler,
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Causal TCN
# ============================================================

class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
    ):
        super().__init__()

        self.left_padding = (
            (kernel_size - 1)
            * dilation
        )

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x):

        x = F.pad(
            x,
            (self.left_padding, 0)
        )

        return self.conv(x)


class TCNBlock(nn.Module):
    def __init__(
        self,
        channels,
        dilation,
        kernel_size=3,
        dropout=0.2,
    ):
        super().__init__()

        self.conv1 = CausalConv1d(
            channels,
            channels,
            kernel_size,
            dilation,
        )

        self.bn1 = nn.BatchNorm1d(
            channels
        )

        self.conv2 = CausalConv1d(
            channels,
            channels,
            kernel_size,
            dilation,
        )

        self.bn2 = nn.BatchNorm1d(
            channels
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(self, x):

        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        return F.relu(
            x + residual
        )


class BinaryDriverTCN(nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_dim=64,
        dropout=0.2,
    ):
        super().__init__()

        self.input_projection = nn.Conv1d(
            feature_dim,
            hidden_dim,
            kernel_size=1,
        )

        dilations = [
            1,
            2,
            4,
            8,
            16,
            32,
        ]

        self.blocks = nn.ModuleList([
            TCNBlock(
                hidden_dim,
                dilation=d,
                dropout=dropout,
            )
            for d in dilations
        ])

        # Single logit:
        # sigmoid(logit) = P(non_active)
        self.classifier = nn.Linear(
            hidden_dim,
            1
        )

    def forward(self, x):

        # [B,100,22] -> [B,22,100]
        x = x.transpose(1, 2)

        x = self.input_projection(x)

        for block in self.blocks:
            x = block(x)

        # causal current-time representation
        x = x[:, :, -1]

        return self.classifier(
            x
        ).squeeze(1)


# ============================================================
# Validation prediction
# ============================================================

@torch.no_grad()
def predict_probabilities(
    model,
    X,
    device,
    batch_size,
):

    ds = TensorDataset(
        torch.from_numpy(
            X.astype(np.float32)
        )
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
    )

    model.eval()

    probs = []

    for (xb,) in loader:

        xb = xb.to(device)

        logits = model(xb)

        p = torch.sigmoid(
            logits
        )

        probs.extend(
            p.cpu().numpy()
        )

    return np.asarray(
        probs,
        dtype=np.float64
    )


def binary_metrics(
    y_true,
    probs,
    threshold,
):

    pred = (
        probs >= threshold
    ).astype(np.int64)

    tp = int(
        np.sum(
            (y_true == 1)
            & (pred == 1)
        )
    )

    fn = int(
        np.sum(
            (y_true == 1)
            & (pred == 0)
        )
    )

    fp = int(
        np.sum(
            (y_true == 0)
            & (pred == 1)
        )
    )

    tn = int(
        np.sum(
            (y_true == 0)
            & (pred == 0)
        )
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    f1 = (
        2
        * precision
        * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    accuracy = (
        (tp + tn)
        / max(
            tp + tn + fp + fn,
            1
        )
    )

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        default=(
            "data/windows/"
            "binary_v1/train.npz"
        )
    )

    parser.add_argument(
        "--val",
        default=(
            "data/windows/"
            "binary_v1/val.npz"
        )
    )

    parser.add_argument(
        "--out",
        default=(
            "models/"
            "binary_tcn_seed42.pt"
        )
    )

    parser.add_argument(
        "--threshold-out",
        default=(
            "data/analysis/"
            "binary_threshold_sweep_seed42.csv"
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=64
    )

    parser.add_argument(
        "--hidden",
        type=int,
        default=64
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=8
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=600
    )

    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.90
    )

    args = parser.parse_args()

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    train = np.load(
        args.train,
        allow_pickle=True
    )

    val = np.load(
        args.val,
        allow_pickle=True
    )

    X_train = train["X"].astype(
        np.float32
    )

    y_train = train["y"].astype(
        np.int64
    )

    X_val = val["X"].astype(
        np.float32
    )

    y_val = val["y"].astype(
        np.int64
    )

    feature_names = [
        str(x)
        for x in train[
            "feature_names"
        ]
    ]

    # ========================================================
    # Normalize using TRAIN ONLY
    # ========================================================

    mean = X_train.mean(
        axis=(0, 1),
        keepdims=True
    )

    std = X_train.std(
        axis=(0, 1),
        keepdims=True
    )

    std = np.where(
        std < 1e-6,
        1.0,
        std
    )

    Xtr = (
        X_train - mean
    ) / std

    Xva = (
        X_val - mean
    ) / std

    # ========================================================
    # Balanced sampler
    #
    # 50% active
    # 50% non_active
    # ========================================================

    counts = np.bincount(
        y_train,
        minlength=2
    ).astype(
        np.float64
    )

    target_probs = np.asarray(
        [0.50, 0.50],
        dtype=np.float64
    )

    class_sample_weights = (
        target_probs
        / np.maximum(
            counts,
            1.0
        )
    )

    sample_weights = (
        class_sample_weights[
            y_train
        ]
    )

    generator = torch.Generator()
    generator.manual_seed(
        args.seed
    )

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(
            sample_weights,
            dtype=torch.double
        ),
        num_samples=args.samples_per_epoch,
        replacement=True,
        generator=generator,
    )

    ds = TensorDataset(
        torch.from_numpy(
            Xtr.astype(np.float32)
        ),
        torch.from_numpy(
            y_train.astype(np.float32)
        ),
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch,
        sampler=sampler,
        shuffle=False,
    )

    print("=" * 72)
    print("BINARY TCN TRAINING")
    print("=" * 72)

    print(
        "Device:",
        device
    )

    print(
        "Train:",
        X_train.shape
    )

    print(
        "Val:",
        X_val.shape
    )

    print(
        "Train counts:",
        counts.astype(int)
    )

    print(
        "Sampler target:",
        target_probs
    )

    print(
        "Samples/epoch:",
        args.samples_per_epoch
    )

    print(
        "Expected sampled counts:",
        np.round(
            target_probs
            * args.samples_per_epoch
        ).astype(int)
    )

    print()

    print(
        "IMPORTANT: TEST SET IS NOT USED."
    )

    # ========================================================
    # Model
    # ========================================================

    model = BinaryDriverTCN(
        feature_dim=X_train.shape[2],
        hidden_dim=args.hidden,
        dropout=args.dropout,
    ).to(device)

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=1e-4,
        )
    )

    best_f1 = -1.0
    best_epoch = 0
    best_state = None

    no_improve = 0

    # ========================================================
    # Training
    #
    # Epoch selection uses threshold 0.5 ONLY.
    # Final operational threshold will be chosen later.
    # ========================================================

    for epoch in range(
        1,
        args.epochs + 1
    ):

        model.train()

        total_loss = 0.0
        total_n = 0

        for xb, yb in loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(xb)

            loss = criterion(
                logits,
                yb
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            optimizer.step()

            total_loss += (
                loss.item()
                * len(yb)
            )

            total_n += len(yb)

        probs = predict_probabilities(
            model,
            Xva,
            device,
            args.batch,
        )

        m = binary_metrics(
            y_val,
            probs,
            threshold=0.5,
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs}"
            f" | train_loss="
            f"{total_loss/total_n:.4f}"
            f" | val_acc="
            f"{m['accuracy']:.4f}"
            f" | val_recall="
            f"{m['recall']:.4f}"
            f" | val_precision="
            f"{m['precision']:.4f}"
            f" | val_f1="
            f"{m['f1']:.4f}"
        )

        if m["f1"] > best_f1:

            best_f1 = m["f1"]
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            no_improve = 0

        else:

            no_improve += 1

        if (
            no_improve
            >= args.patience
        ):

            print()
            print(
                "EARLY STOPPING"
            )

            break

    # ========================================================
    # Restore best checkpoint
    # ========================================================

    model.load_state_dict(
        best_state
    )

    val_probs = predict_probabilities(
        model,
        Xva,
        device,
        args.batch,
    )

    # ========================================================
    # Threshold sweep
    # ========================================================

    rows = []

    thresholds = np.arange(
        0.05,
        0.951,
        0.05
    )

    for threshold in thresholds:

        rows.append(
            binary_metrics(
                y_val,
                val_probs,
                float(threshold),
            )
        )

    sweep = pd.DataFrame(
        rows
    )

    # ========================================================
    # Operational threshold selection
    #
    # Requirement:
    # non_active recall >= target
    #
    # Among satisfying thresholds:
    # choose HIGHEST threshold.
    #
    # This tries to reduce unnecessary Qwen calls while
    # maintaining the required detector sensitivity.
    # ========================================================

    candidates = sweep[
        sweep["recall"]
        >= args.target_recall
    ]

    if len(candidates) > 0:

        selected = (
            candidates
            .sort_values(
                "threshold",
                ascending=False
            )
            .iloc[0]
        )

        selection_reason = (
            f"highest threshold with "
            f"recall >= "
            f"{args.target_recall:.2f}"
        )

    else:

        # fallback:
        # maximum recall, then specificity
        selected = (
            sweep
            .sort_values(
                [
                    "recall",
                    "specificity",
                    "threshold",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ]
            )
            .iloc[0]
        )

        selection_reason = (
            "target recall unavailable; "
            "selected maximum recall"
        )

    selected_threshold = float(
        selected["threshold"]
    )

    # ========================================================
    # Print best validation checkpoint
    # ========================================================

    m05 = binary_metrics(
        y_val,
        val_probs,
        0.5,
    )

    print()
    print("=" * 72)
    print("BEST VALIDATION CHECKPOINT")
    print("=" * 72)

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        "Threshold 0.50"
    )

    print(
        f"Accuracy    : {m05['accuracy']:.4f}"
    )

    print(
        f"Precision   : {m05['precision']:.4f}"
    )

    print(
        f"Recall      : {m05['recall']:.4f}"
    )

    print(
        f"Specificity : {m05['specificity']:.4f}"
    )

    print(
        f"F1          : {m05['f1']:.4f}"
    )

    print(
        f"TP={m05['tp']} "
        f"FN={m05['fn']} "
        f"FP={m05['fp']} "
        f"TN={m05['tn']}"
    )

    # ========================================================
    # Print threshold sweep
    # ========================================================

    print()
    print("=" * 72)
    print("THRESHOLD SWEEP")
    print("=" * 72)

    print(
        sweep[
            [
                "threshold",
                "precision",
                "recall",
                "specificity",
                "f1",
                "tp",
                "fn",
                "fp",
                "tn",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    print()
    print("=" * 72)
    print("SELECTED QWEN TRIGGER THRESHOLD")
    print("=" * 72)

    print(
        "Threshold:",
        f"{selected_threshold:.3f}"
    )

    print(
        "Reason:",
        selection_reason
    )

    print(
        "Non-active recall:",
        f"{selected['recall']:.4f}"
    )

    print(
        "Precision:",
        f"{selected['precision']:.4f}"
    )

    print(
        "Specificity:",
        f"{selected['specificity']:.4f}"
    )

    print(
        "F1:",
        f"{selected['f1']:.4f}"
    )

    print(
        f"TP={int(selected['tp'])} "
        f"FN={int(selected['fn'])} "
        f"FP={int(selected['fp'])} "
        f"TN={int(selected['tn'])}"
    )

    # ========================================================
    # Save sweep
    # ========================================================

    threshold_out = Path(
        args.threshold_out
    )

    threshold_out.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sweep.to_csv(
        threshold_out,
        index=False
    )

    # ========================================================
    # Save model
    # ========================================================

    out = Path(
        args.out
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "feature_dim":
                X_train.shape[2],

            "hidden_dim":
                args.hidden,

            "dropout":
                args.dropout,

            "feature_names":
                feature_names,

            "class_names":
                [
                    "active",
                    "non_active",
                ],

            "normalization_mean":
                mean,

            "normalization_std":
                std,

            "best_epoch":
                best_epoch,

            "best_val_f1_at_05":
                best_f1,

            "selected_threshold":
                selected_threshold,

            "target_recall":
                args.target_recall,

            "seed":
                args.seed,
        },
        out,
    )

    print()
    print(
        "Saved model:",
        out
    )

    print(
        "Saved threshold sweep:",
        threshold_out
    )

    print()
    print(
        "TEST SET WAS NOT LOADED."
    )


if __name__ == "__main__":
    main()
