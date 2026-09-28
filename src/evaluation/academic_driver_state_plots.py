#!/usr/bin/env python3
"""Publication-style plots for driver-state evaluation artifacts.

The plotting code is kept separate from inference so an existing experiment can
be re-rendered without running the models again.
"""

from __future__ import annotations

import csv
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


LABELS = ("active", "reduced", "no_visible_response")
DISPLAY_LABELS = ("Active", "Reduced", "No visible\nresponse")
METRIC_COLORS = ("#0072B2", "#E69F00", "#009E73")
SYSTEM_COLORS = ("#0072B2", "#D55E00", "#009E73")


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def read_matched_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def normalize_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize common and legacy fusion CSV columns for plotting."""
    normalized: list[dict[str, Any]] = []
    for row in rows:
        truth = str(row.get("ground_truth", "")).strip().lower()
        prediction = str(
            row.get("prediction", row.get("qwen_prediction", ""))
        ).strip().lower()
        if truth not in LABELS or prediction not in LABELS:
            continue
        normalized.append(
            {
                "ground_truth": truth,
                "prediction": prediction,
                "timestamp_sec": _number(
                    row.get("timestamp_sec", row.get("video_time_sec"))
                ),
                "model_ms": _number(
                    row.get("model_ms", row.get("qwen_inference_ms"))
                ),
                "end_to_end_ms": _number(row.get("end_to_end_ms")),
                "event_id": row.get("event_id"),
            }
        )
    if not normalized:
        raise ValueError("유효한 ground_truth/prediction 행이 없습니다.")
    return normalized


def classification_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    matrix = {
        truth: Counter(
            str(row["prediction"])
            for row in rows
            if str(row["ground_truth"]) == truth
        )
        for truth in LABELS
    }
    per_class: dict[str, dict[str, float | int | None]] = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in LABELS if other != label)
        fn = sum(matrix[label][other] for other in LABELS if other != label)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision is not None
            and recall is not None
            and precision + recall
            else None
        )
        per_class[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    usable_f1 = [item["f1"] for item in per_class.values() if item["f1"] is not None]
    correct = sum(matrix[label][label] for label in LABELS)
    return {
        "per_class": per_class,
        "macro_f1": sum(usable_f1) / len(usable_f1) if usable_f1 else None,
        "accuracy": correct / len(rows),
        "confusion_matrix": [
            [matrix[truth][prediction] for prediction in LABELS]
            for truth in LABELS
        ],
    }


def _confusion_from_rows(rows: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    return classification_summary(rows)["confusion_matrix"]


def _latencies(
    rows: Sequence[Mapping[str, Any]], key: str
) -> list[float]:
    return [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float))
        and float(row[key]) >= 0.0
    ]


def _overall_state_error(summary: Mapping[str, Any]) -> tuple[int | None, float | None, int | None]:
    """Return incorrect predictions, error rate, and evaluated sample count."""
    matrix = summary.get("confusion_matrix")
    if isinstance(matrix, Sequence) and matrix:
        try:
            total = sum(int(value) for row in matrix for value in row)
            correct = sum(int(matrix[index][index]) for index in range(len(matrix)))
        except (TypeError, ValueError, IndexError):
            total = 0
        if total > 0:
            errors = total - correct
            return errors, errors / total, total
    sample_count = _number(summary.get("matched_samples"))
    accuracy = _number(summary.get("accuracy"))
    if sample_count is None or accuracy is None or sample_count <= 0:
        return None, None, None
    total = int(round(sample_count))
    errors = int(round(total * (1.0 - accuracy)))
    return errors, errors / total, total


def _setup_matplotlib() -> tuple[Any, Any]:
    os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK KR", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return plt, np


def _percent_bars(axis: Any, values: Sequence[float | None]) -> None:
    for patch, value in zip(axis.patches, values):
        if value is None:
            continue
        axis.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + 1.4,
            f"{100 * value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _style_score_axis(axis: Any, title: str) -> None:
    axis.set_title(title, loc="left", fontweight="bold", pad=10)
    axis.set_ylabel("Score (%)")
    axis.set_ylim(0, 112)
    axis.set_yticks(range(0, 101, 20))
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.75)
    axis.set_axisbelow(True)


def _label_score_bars(axis: Any, bars: Any, values: Sequence[float | None]) -> None:
    for bar, value in zip(bars, values):
        if value is None:
            label = "N/A"
        else:
            label = f"{100.0 * value:.1f}"
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.6,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


def _label_grouped_score_bars(
    axis: Any, bars: Any, values: Sequence[float | None]
) -> None:
    """Keep labels inside narrow grouped bars so adjacent values do not collide."""
    for bar, value in zip(bars, values):
        if value is None:
            continue
        height = 100.0 * value
        inside = height >= 22.0
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            height - 2.8 if inside else height + 1.6,
            f"{height:.1f}",
            ha="center",
            va="top" if inside else "bottom",
            fontsize=7.8,
            color="white" if inside else "#111111",
            fontweight="bold",
        )


def render_evaluation_figure(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    write_pdf: bool = True,
) -> None:
    """Render the four decision-oriented metrics for one system."""
    rows = normalize_rows(rows)
    computed = classification_summary(rows)
    per_class = computed["per_class"]
    title = str(summary.get("title") or "Driver-state system")
    evaluation_type = str(summary.get("evaluation_type") or "continuous_timeline")
    duration = _number(summary.get("duration_sec"))
    scope = (
        "Candidate-based samples"
        if evaluation_type == "candidate_sampled"
        else "Continuous timeline"
    )

    plt, np = _setup_matplotlib()
    # English panel headings are wider than the prior Korean labels.  Reserve
    # a dedicated header band instead of letting the subtitle compete with the
    # top-row panel titles.
    figure, axes = plt.subplots(2, 2, figsize=(13.8, 8.4))
    figure.subplots_adjust(
        left=0.07, right=0.985, bottom=0.09, top=0.84, hspace=0.35, wspace=0.38
    )

    metric_names = ("Precision", "Recall", "F1 score")
    metric_keys = ("precision", "recall", "f1")

    # (a) Only the safety-critical no-response class is shown.
    state_values = [_number(per_class["no_visible_response"].get(key)) for key in metric_keys]
    state_bars = axes[0, 0].bar(
        metric_names,
        [0.0 if value is None else 100.0 * value for value in state_values],
        color=METRIC_COLORS,
        width=0.62,
        edgecolor="white",
        linewidth=0.8,
    )
    _style_score_axis(axes[0, 0], "(a) No-response state classification")
    _label_score_bars(axes[0, 0], state_bars, state_values)

    # (b) Event-unit detection is intentionally separated from state samples.
    event = summary.get("event_metrics")
    if isinstance(event, Mapping):
        event_values = [_number(event.get(key)) for key in metric_keys]
    else:
        event_values = [None, None, None]
    event_bars = axes[0, 1].bar(
        metric_names,
        [0.0 if value is None else 100.0 * value for value in event_values],
        color=METRIC_COLORS,
        edgecolor="white",
        linewidth=0.8,
        width=0.62,
    )
    _style_score_axis(axes[0, 1], "(b) Event-level detection performance")
    _label_score_bars(axes[0, 1], event_bars, event_values)
    if isinstance(event, Mapping):
        tp, fp, fn = (event.get(key, "N/A") for key in ("tp", "fp", "fn"))
        axes[0, 1].text(
            0.02, 0.96, f"TP {tp}  |  FP {fp}  |  FN {fn}",
            transform=axes[0, 1].transAxes, ha="left", va="top",
            fontsize=8.5, color="#4d4d4d",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
        )

    # (c) Count every wrong Active/Reduced/NVR assignment as one state error.
    error_count, error_rate, sample_count = _overall_state_error(computed)
    error_percent = None if error_rate is None else 100.0 * error_rate
    error_bar = axes[1, 0].barh(
        ["Overall state"], [error_percent or 0.0], color="#D55E00", height=0.42
    )[0]
    axes[1, 0].set_title("(c) Overall state error", loc="left", fontweight="bold", pad=10)
    axes[1, 0].set_xlabel("Overall state error rate (%) ↓")
    axes[1, 0].grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.75)
    axes[1, 0].set_axisbelow(True)
    axes[1, 0].set_xlim(0, max(5.0, (error_percent or 0.0) * 1.36))
    error_label = (
        "N/A"
        if error_percent is None
        else f"{error_percent:.1f}%   (errors {error_count:,}/{sample_count:,})"
    )
    axes[1, 0].text(
        error_bar.get_width() + axes[1, 0].get_xlim()[1] * 0.025,
        error_bar.get_y() + error_bar.get_height() / 2,
        error_label,
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    # (d) Median and p95 provide typical and tail inference latency at a glance.
    model_latencies = _latencies(rows, "model_ms")
    median_value = _percentile(model_latencies, 0.50)
    p95_value = _percentile(model_latencies, 0.95)
    latency_values = [median_value, p95_value]
    latency_bars = axes[1, 1].barh(
        ["Median (p50)", "Tail latency (p95)"],
        [value or 0.0 for value in latency_values],
        color=("#56B4E9", "#0072B2"),
        height=0.48,
    )
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_title("(d) Model inference latency", loc="left", fontweight="bold", pad=10)
    axes[1, 1].set_xlabel("Latency (ms) ↓")
    axes[1, 1].grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.75)
    axes[1, 1].set_axisbelow(True)
    axes[1, 1].set_xlim(0, max(1.0, (p95_value or median_value or 0.0) * 1.25))
    for bar, value in zip(latency_bars, latency_values):
        if value is not None:
            axes[1, 1].text(
                bar.get_width() + axes[1, 1].get_xlim()[1] * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{value:,.1f} ms",
                va="center",
                fontsize=9.5,
                fontweight="bold",
            )
    if not model_latencies:
        axes[1, 1].text(0.5, 0.5, "No latency samples", ha="center")

    duration_note = f", {duration:.1f} s" if duration is not None else ""
    figure.suptitle(
        f"{title}: Safety Performance Summary",
        fontsize=15,
        fontweight="bold",
        y=0.975,
    )
    figure.text(
        0.5,
        0.925,
        f"{scope}  |  Samples={len(rows):,}{duration_note}",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#444444",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    if write_pdf:
        figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def render_system_comparison(
    systems: Sequence[tuple[str, Mapping[str, Any]]],
    output_path: Path,
) -> None:
    """Render the same four evaluation questions with stable system colors."""
    plt, np = _setup_matplotlib()
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 7.8), constrained_layout=True)
    figure.set_constrained_layout_pads(w_pad=0.05, h_pad=0.05, hspace=0.10, wspace=0.08)
    names = [name for name, _ in systems]
    legend_names = names
    metric_names = ("Precision", "Recall", "F1 score")
    metric_keys = ("precision", "recall", "f1")
    metric_x = np.arange(len(metric_names))
    width = 0.23
    offsets = np.linspace(-width, width, len(systems))

    # (a) Sample-level performance for the safety-critical NVR class only.
    for offset, (name, summary), color in zip(offsets, systems, SYSTEM_COLORS):
        nvr = summary["per_class"]["no_visible_response"]
        values = [_number(nvr.get(key)) for key in metric_keys]
        bars = axes[0, 0].bar(
            metric_x + offset,
            [0.0 if value is None else 100.0 * value for value in values],
            width,
            label=name,
            color=color,
            edgecolor="white",
            linewidth=0.6,
        )
        _label_grouped_score_bars(axes[0, 0], bars, values)
    axes[0, 0].set_xticks(metric_x, metric_names)
    _style_score_axis(axes[0, 0], "(a) No-response state classification")
    axes[0, 0].text(
        0.02, 0.04, "Same 1.5 s windows · no response vs. other states",
        transform=axes[0, 0].transAxes, ha="left", va="bottom",
        fontsize=8.2, color="#4d4d4d",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
    )

    # (b) Event-unit detection, using the same system colors as every panel.
    for offset, (name, summary), color in zip(offsets, systems, SYSTEM_COLORS):
        event = summary.get("event_metrics", {})
        values = [_number(event.get(key)) for key in metric_keys]
        bars = axes[0, 1].bar(
            metric_x + offset,
            [0.0 if value is None else 100.0 * value for value in values],
            width,
            label=name,
            color=color,
            edgecolor="white",
            linewidth=0.6,
        )
        _label_grouped_score_bars(axes[0, 1], bars, values)
    axes[0, 1].set_xticks(metric_x, metric_names)
    _style_score_axis(axes[0, 1], "(b) Event-level detection performance")
    axes[0, 1].text(
        0.02, 0.04, "One-to-one event matching · minimum 0.5 s overlap",
        transform=axes[0, 1].transAxes, ha="left", va="bottom",
        fontsize=8.2, color="#4d4d4d",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
    )

    # (c) False NVR alarm events over the exact same observation duration.
    y = np.arange(len(systems))
    false_alarm_counts = [
        _number(summary.get("event_metrics", {}).get("fp"))
        for _, summary in systems
    ]
    false_alarm_rates = [
        _number(summary.get("event_metrics", {}).get("fp_per_hour"))
        for _, summary in systems
    ]
    false_alarm_bars = axes[1, 0].barh(
        y,
        [value or 0.0 for value in false_alarm_counts],
        color=SYSTEM_COLORS[: len(systems)],
        height=0.54,
    )
    axes[1, 0].set_yticks(y, names)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_title("(c) No-response false alarms", loc="left", fontweight="bold", pad=10)
    axes[1, 0].set_xlabel("False-alarm events (same observation period) ↓")
    axes[1, 0].grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.75)
    axes[1, 0].set_axisbelow(True)
    axes[1, 0].set_xlim(
        0, max(1.0, max((value or 0.0) for value in false_alarm_counts) * 1.45)
    )
    for bar, count, rate in zip(
        false_alarm_bars, false_alarm_counts, false_alarm_rates
    ):
        label = (
            "N/A"
            if count is None
            else f"{int(count)}" + (f"  ({rate:.1f}/h)" if rate is not None else "")
        )
        axes[1, 0].text(
            bar.get_width() + axes[1, 0].get_xlim()[1] * 0.018,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    # (d) Model-only latency.  Do not reuse the common 1.5-second scoring grid
    # for this panel: that grid deliberately down-samples TCN predictions to
    # VLM clip boundaries and would inflate its time-to-alert.  The operational
    # event-delay metric remains in the CSV/audit, while this panel reports the
    # like-for-like per-call runtime measured by each system.
    medians = [
        _number(
            summary.get("operational", {})
            .get("model_latency_ms", {})
            .get("p50")
        )
        for _, summary in systems
    ]
    p95_values = [
        _number(
            summary.get("operational", {})
            .get("model_latency_ms", {})
            .get("p95")
        )
        for _, summary in systems
    ]
    for row, (median_value, p95_value, color) in enumerate(
        zip(medians, p95_values, SYSTEM_COLORS)
    ):
        if median_value is None or p95_value is None:
            continue
        axes[1, 1].plot(
            [median_value, p95_value], [row, row], color=color, linewidth=4,
            solid_capstyle="round", zorder=2,
        )
        axes[1, 1].scatter(median_value, row, s=55, color=color, marker="o", zorder=3)
        axes[1, 1].scatter(p95_value, row, s=65, color=color, marker="D", zorder=3)
        axes[1, 1].annotate(
            f"p50 {median_value:,.1f} · p95 {p95_value:,.1f} ms",
            (p95_value, row), xytext=(8, 0), textcoords="offset points",
            ha="left", va="center", fontsize=8.3, fontweight="bold",
        )
    axes[1, 1].set_yticks(y, names)
    axes[1, 1].invert_yaxis()
    available_latencies = [value for value in p95_values if value is not None]
    if available_latencies:
        positive_latencies = [
            value for value in medians + p95_values
            if value is not None and value > 0.0
        ]
        axes[1, 1].set_xscale("log")
        axes[1, 1].set_xlim(
            min(positive_latencies) / 1.8,
            max(available_latencies) * 2.2,
        )
    axes[1, 1].set_title("(d) Model inference latency", loc="left", fontweight="bold", pad=10)
    axes[1, 1].set_xlabel("Model call → response (ms, log scale) ↓   ○ p50   ◆ p95")
    axes[1, 1].grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.75)
    axes[1, 1].set_axisbelow(True)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=label)
        for color, label in zip(SYSTEM_COLORS, legend_names)
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncols=len(handles),
        fontsize=9.5,
    )
    figure.suptitle(
        "Driver-state monitoring: same-condition system comparison",
        fontsize=15,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
