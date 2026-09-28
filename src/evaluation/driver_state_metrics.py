#!/usr/bin/env python3
"""Common, event-aware evaluation for TCN, VLM, and fused driver states."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from academic_driver_state_plots import render_evaluation_figure


LABELS = ("active", "reduced", "no_visible_response")
POSITIVE = "no_visible_response"
CLIP_PATTERN = re.compile(r"(\d+(?:\.\d+)?)s_to_(\d+(?:\.\d+)?)s")


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _parse_ground_truth(path: Path) -> list[tuple[float, float, str, str]]:
    intervals: list[tuple[float, float, str, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            clip_name = (row.get("clip_name") or "").strip()
            label = (row.get("driver_response") or "").strip().lower()
            match = CLIP_PATTERN.search(clip_name)
            if match and label in LABELS:
                intervals.append((float(match.group(1)), float(match.group(2)), clip_name, label))
    if not intervals:
        raise ValueError(f"시간 구간 정답을 읽지 못했습니다: {path}")
    return sorted(intervals)


def _truth_at(timestamp: float, intervals: list[tuple[float, float, str, str]]) -> tuple[str, str] | None:
    matches = [item for item in intervals if item[0] <= timestamp < item[1]]
    if not matches:
        return None
    _, _, clip_name, label = max(matches, key=lambda item: item[0])
    return clip_name, label


def _load_records(
    path: Path, source: str, prediction_field: str = "driver_response",
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if source == "tcn":
                timestamp = record.get("timestamp")
                prediction = record.get(prediction_field)
                model_ms = record.get("tcn_inference_ms")
                end_to_end_ms = model_ms
            else:
                timestamp = record.get("video_time")
                result = record.get("bridge_result")
                state = result.get("driver_state") if isinstance(result, dict) else None
                prediction = state.get("driver_response") if isinstance(state, dict) else None
                model_ms = record.get("qwen_model_inference_ms")
                if not isinstance(model_ms, (int, float)) and isinstance(result, dict):
                    model_ms = result.get("inference_ms")
                end_to_end_sec = record.get("event_end_to_end_sec")
                end_to_end_ms = float(end_to_end_sec) * 1000.0 if isinstance(end_to_end_sec, (int, float)) else None
            if not isinstance(timestamp, (int, float)) or not isinstance(prediction, str):
                continue
            prediction = prediction.strip().lower()
            if prediction not in LABELS:
                continue
            records.append({
                "timestamp_sec": float(timestamp),
                "prediction": prediction,
                "model_ms": float(model_ms) if isinstance(model_ms, (int, float)) else None,
                "end_to_end_ms": end_to_end_ms,
            })
    # Keep append order here: a JSONL may contain several complete runs with
    # overlapping video timestamps. Callers select the latest run by file
    # order, then the selected records are sorted only for time-series work.
    return records


def _source_duration(records: list[dict[str, Any]]) -> float:
    if len(records) < 2:
        return 0.0
    return max(0.0, float(records[-1]["timestamp_sec"]) - float(records[0]["timestamp_sec"]))


def _support_width(records: list[dict[str, Any]]) -> float:
    gaps = [
        float(right["timestamp_sec"]) - float(left["timestamp_sec"])
        for left, right in zip(records, records[1:])
        if 0.0 < float(right["timestamp_sec"]) - float(left["timestamp_sec"]) <= 10.0
    ]
    return median(gaps) if gaps else 1.0


def _positive_events_from_predictions(records: list[dict[str, Any]]) -> list[tuple[float, float]]:
    if not records:
        return []
    support = _support_width(records)
    merge_gap = support * 1.5
    events: list[list[float]] = []
    for row in records:
        if row["prediction"] != POSITIVE:
            continue
        timestamp = float(row["timestamp_sec"])
        if events and timestamp - events[-1][1] <= merge_gap:
            events[-1][1] = timestamp
        else:
            events.append([timestamp, timestamp])
    return [(start - support / 2.0, end + support / 2.0) for start, end in events]


def _positive_events_from_truth(intervals: Iterable[tuple[float, float, str, str]]) -> list[tuple[float, float]]:
    events: list[list[float]] = []
    for start, end, _clip, label in intervals:
        if label != POSITIVE:
            continue
        if events and start <= events[-1][1] + 1e-9:
            events[-1][1] = max(events[-1][1], end)
        else:
            events.append([start, end])
    return [(start, end) for start, end in events]


def _overlap(first: tuple[float, float], second: tuple[float, float]) -> bool:
    return first[0] < second[1] and second[0] < first[1]


def evaluate_jsonl(
    results_path: Path,
    ground_truth_path: Path,
    output_dir: Path,
    *,
    source: str,
    title: str,
    last_records: int | None = None,
    observation_duration_sec: float | None = None,
    coverage_type: str = "continuous_timeline",
    prediction_field: str = "driver_response",
) -> dict[str, Path]:
    """Evaluate a complete timeline or a candidate-event log and draw one figure.

    ``source`` is ``tcn`` for TCN state JSONL and ``vlm`` for VLM/fusion JSONL.
    Event-triggered fusion logs are explicitly marked as candidate-sampled in
    the plot because their macro-F1 is not a whole-video state metric.
    """
    if source not in {"tcn", "vlm"}:
        raise ValueError("source must be 'tcn' or 'vlm'")
    records = _load_records(results_path, source, prediction_field=prediction_field)
    if last_records is not None and last_records > 0:
        records = records[-last_records:]
    records.sort(key=lambda row: float(row["timestamp_sec"]))
    if not records:
        raise ValueError(f"평가할 유효 결과가 없습니다: {results_path}")
    intervals = _parse_ground_truth(ground_truth_path)
    matched: list[dict[str, Any]] = []
    for record in records:
        truth = _truth_at(float(record["timestamp_sec"]), intervals)
        if truth is None:
            continue
        clip_name, label = truth
        matched.append({**record, "ground_truth_clip": clip_name, "ground_truth": label})
    if not matched:
        raise ValueError("결과의 timestamp가 정답 시간 범위와 겹치지 않습니다.")

    matrix = {truth: Counter() for truth in LABELS}
    for row in matched:
        matrix[str(row["ground_truth"])][str(row["prediction"])] += 1
    per_class: dict[str, dict[str, float | int | None]] = {}
    f1_values: list[float] = []
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in LABELS if other != label)
        fn = sum(matrix[label][other] for other in LABELS if other != label)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        # Use the count form so a supported class with zero true positives is
        # scored as F1=0 instead of being omitted from macro-F1. Only a class
        # with neither truth support nor predictions is undefined.
        f1_denominator = 2 * tp + fp + fn
        f1 = 2.0 * tp / f1_denominator if f1_denominator else None
        if f1 is not None:
            f1_values.append(f1)
        per_class[label] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else None

    predicted_events = _positive_events_from_predictions(records)
    # Only score ground-truth events that overlap the evaluated observation
    # window.  The annotation CSV may extend beyond a shortened experiment;
    # counting those future events as false negatives biases event recall.
    support = _support_width(records)
    observation_start = max(0.0, float(records[0]["timestamp_sec"]) - support / 2.0)
    if observation_duration_sec is not None and observation_duration_sec > 0.0:
        # Timestamps are relative to the source video.  ``observation_duration_sec``
        # is therefore the video-time boundary, not a duration to add to the
        # first retained record (which may start late after run de-duplication).
        observation_end = float(observation_duration_sec)
    else:
        observation_end = float(records[-1]["timestamp_sec"]) + support / 2.0
    observed_intervals = [
        item
        for item in intervals
        if item[0] < observation_end and item[1] > observation_start
    ]
    truth_events = _positive_events_from_truth(observed_intervals)
    matched_truth = [any(_overlap(truth, predicted) for predicted in predicted_events) for truth in truth_events]
    matched_predicted = [any(_overlap(predicted, truth) for truth in truth_events) for predicted in predicted_events]
    event_tp, event_fn, event_fp = sum(matched_truth), len(truth_events) - sum(matched_truth), len(predicted_events) - sum(matched_predicted)
    event_precision = _safe_div(event_tp, event_tp + event_fp)
    event_recall = _safe_div(event_tp, event_tp + event_fn)
    event_f1_denominator = 2 * event_tp + event_fp + event_fn
    event_f1 = (
        2.0 * event_tp / event_f1_denominator
        if event_f1_denominator else None
    )
    duration_sec = (
        float(observation_duration_sec)
        if observation_duration_sec is not None and observation_duration_sec > 0.0
        else _source_duration(records)
    )
    fp_per_second = event_fp / duration_sec if duration_sec > 0 else None
    calls_per_min = len(records) / (duration_sec / 60.0) if duration_sec > 0 else None
    model_latencies = [float(row["model_ms"]) for row in records if isinstance(row["model_ms"], (int, float))]
    end_latencies = [float(row["end_to_end_ms"]) for row in records if isinstance(row["end_to_end_ms"], (int, float))]
    correct_samples = sum(matrix[label][label] for label in LABELS)
    incorrect_samples = len(matched) - correct_samples

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", title.lower()).strip("_") or "driver_state"
    matched_path = output_dir / f"{stem}_matched.csv"
    summary_path = output_dir / f"{stem}_metrics.json"
    plot_path = output_dir / f"{stem}_metrics.png"
    with matched_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(matched[0]))
        writer.writeheader()
        writer.writerows(matched)
    summary = {
        "title": title,
        "evaluation_type": coverage_type,
        "matched_samples": len(matched),
        "duration_sec": duration_sec,
        "observation_window_sec": [observation_start, observation_end],
        "no_visible_response": per_class[POSITIVE],
        "macro_f1": macro_f1,
        "overall_state_errors": {
            "count": incorrect_samples,
            "rate": incorrect_samples / len(matched),
        },
        "event_metrics": {
            "tp": event_tp,
            "fp": event_fp,
            "fn": event_fn,
            "precision": event_precision,
            "recall": event_recall,
            "f1": event_f1,
            "fp_per_second": fp_per_second,
        },
        "operational": {
            "calls_per_min": calls_per_min,
            "model_latency_ms": {
                "mean": sum(model_latencies) / len(model_latencies) if model_latencies else None,
                "p50": _percentile(model_latencies, 0.50),
                "p95": _percentile(model_latencies, 0.95),
            },
            "end_to_end_latency_ms": {
                "mean": sum(end_latencies) / len(end_latencies) if end_latencies else None,
                "p50": _percentile(end_latencies, 0.50),
                "p95": _percentile(end_latencies, 0.95),
            },
        },
        "per_class": per_class,
        "accuracy": correct_samples / len(matched),
        "confusion_matrix": [
            [matrix[truth][prediction] for prediction in LABELS]
            for truth in LABELS
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    render_evaluation_figure(summary, matched, plot_path)
    print(
        f"학술 논문형 운전자 상태 평가 그래프 저장: {plot_path} "
        f"(+ {plot_path.with_suffix('.pdf')})",
        flush=True,
    )
    return {"plot": plot_path, "matched": matched_path, "summary": summary_path}
