#!/usr/bin/env python3
"""Compare TCN-only, VLM-only and TCN–VLM–YOLO on a common 1.5 s grid.

All three systems are scored on the same completed 1.5-second VLM-only
windows (NVR vs non-NVR).  Fusion windows without a VLM candidate count as
non-NVR.  Event metrics use one-to-one matching with at least 0.5 s overlap.

Default inputs are the saved paper runs in ``results/paper_20260917``, so
``python src/evaluation/compare_systems.py`` reproduces the reported table
without a GPU, the source video, or model weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Mapping

from academic_driver_state_plots import (
    classification_summary,
    normalize_rows,
    read_matched_csv,
    render_evaluation_figure,
    render_system_comparison,
)
from driver_state_metrics import evaluate_jsonl


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_RUN_DIR = REPO_ROOT / "results" / "paper_20260917"
GT_PATH = REPO_ROOT / "data" / "labels" / "1000013115_ground_truth_nonoverlap_4s.csv"
TCN_STATES_JSONL = PAPER_RUN_DIR / "inputs" / "tcn_only_states.jsonl"
VLM_YOLO_JSONL = PAPER_RUN_DIR / "inputs" / "vlm_only_clips.jsonl"
FUSION_CSV = PAPER_RUN_DIR / "inputs" / "tcn_vlm_yolo_matched.csv"
DEFAULT_OUTPUT_DIR = PAPER_RUN_DIR / "evaluation"
POSITIVE = "no_visible_response"
CLIP_PATTERN = re.compile(r"(\d+(?:\.\d+)?)s_to_(\d+(?:\.\d+)?)s")
COMMON_GRID_SEC = 1.5
EVENT_MIN_OVERLAP_SEC = 0.5


def _safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(values) - 1)
    weight = position - low
    return values[low] * (1.0 - weight) + values[high] * weight


def _ground_truth_events(path: Path, observation_end: float) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if str(row.get("driver_response", "")).strip().lower() != POSITIVE:
                continue
            match = CLIP_PATTERN.search(str(row.get("clip_name", "")))
            if match is None:
                continue
            start, end = (float(value) for value in match.groups())
            if start >= observation_end or end <= 0.0:
                continue
            if intervals and start <= intervals[-1][1] + 1e-9:
                intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end))
            else:
                intervals.append((start, end))
    return intervals


def _overlap(first: tuple[float, float], second: tuple[float, float]) -> bool:
    return first[0] < second[1] and second[0] < first[1]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _latest_complete_vlm_run(
    path: Path, duration_sec: float
) -> list[dict[str, Any]]:
    """Select the latest run that reached the end of the video.

    The result log is append-only and can contain a new, still-running pass.
    A backwards jump in clip end time marks a run boundary.
    """
    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_end: float | None = None
    for record in _read_jsonl(path):
        end = record.get("event_video_end_sec", record.get("video_time"))
        if not isinstance(end, (int, float)):
            continue
        end = float(end)
        if previous_end is not None and end < previous_end - 1e-6:
            if current:
                runs.append(current)
            current = []
        current.append(record)
        previous_end = end
    if current:
        runs.append(current)
    complete = [
        run
        for run in runs
        if float(run[-1].get("event_video_end_sec", run[-1].get("video_time", 0.0)))
        >= duration_sec - COMMON_GRID_SEC
    ]
    if not complete:
        raise ValueError(f"영상 끝까지 완료된 VLM-only 실행이 없습니다: {path}")
    return complete[-1]


def _ground_truth_intervals(path: Path) -> list[tuple[float, float, str]]:
    intervals: list[tuple[float, float, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            label = str(row.get("driver_response", "")).strip().lower()
            match = CLIP_PATTERN.search(str(row.get("clip_name", "")))
            if match is None or label not in {"active", "reduced", POSITIVE}:
                continue
            intervals.append((float(match.group(1)), float(match.group(2)), label))
    return sorted(intervals)


def _truth_at(
    timestamp: float, intervals: list[tuple[float, float, str]]
) -> str | None:
    matches = [item for item in intervals if item[0] <= timestamp < item[1]]
    return max(matches, key=lambda item: item[0])[2] if matches else None


def _binary_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(
        row["ground_truth"] == POSITIVE and row["prediction"] == POSITIVE
        for row in rows
    )
    fp = sum(
        row["ground_truth"] != POSITIVE and row["prediction"] == POSITIVE
        for row in rows
    )
    fn = sum(
        row["ground_truth"] == POSITIVE and row["prediction"] != POSITIVE
        for row in rows
    )
    tn = len(rows) - tp - fp - fn
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / len(rows),
    }


def _prediction_events(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    events: list[dict[str, float]] = []
    for row in rows:
        if row["prediction"] != POSITIVE:
            continue
        start = float(row["window_start_sec"])
        end = float(row["window_end_sec"])
        alert = float(row["alert_time_sec"])
        if events and start <= events[-1]["end"] + 1e-9:
            events[-1]["end"] = max(events[-1]["end"], end)
            events[-1]["alert"] = min(events[-1]["alert"], alert)
        else:
            events.append({"start": start, "end": end, "alert": alert})
    return events


def _score_events_one_to_one(
    predicted: list[dict[str, float]],
    truth: list[tuple[float, float]],
    duration_sec: float,
) -> dict[str, Any]:
    used: set[int] = set()
    delays: list[float] = []
    for truth_start, truth_end in truth:
        candidates: list[tuple[float, int]] = []
        for index, event in enumerate(predicted):
            if index in used:
                continue
            overlap = max(
                0.0,
                min(truth_end, event["end"]) - max(truth_start, event["start"]),
            )
            if overlap >= EVENT_MIN_OVERLAP_SEC and event["alert"] >= truth_start:
                candidates.append((event["alert"], index))
        if candidates:
            alert, index = min(candidates)
            used.add(index)
            delays.append(max(0.0, alert - truth_start))
    tp = len(used)
    fp = len(predicted) - tp
    fn = len(truth) - tp
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp_per_hour": fp / (duration_sec / 3600.0),
        "matching": "one-to-one, minimum temporal overlap 0.5 s",
        "detection_latency_sec": {
            "samples": len(delays),
            "p50": _percentile(delays, 0.50),
            "p95": _percentile(delays, 0.95),
        },
    }


def build_common_grid_comparison(
    duration_sec: float,
    tcn_jsonl: Path,
    vlm_jsonl: Path,
    fusion_csv: Path,
    ground_truth_path: Path,
) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, Any]]:
    """Evaluate all systems on the same completed 1.5-second VLM windows."""
    truth_intervals = _ground_truth_intervals(ground_truth_path)
    truth_events = _ground_truth_events(ground_truth_path, duration_sec)
    tcn_records = sorted(
        _read_jsonl(tcn_jsonl), key=lambda row: float(row.get("timestamp", -1.0))
    )
    vlm_records = _latest_complete_vlm_run(vlm_jsonl, duration_sec)
    fusion_records = normalize_rows(read_matched_csv(fusion_csv))

    # Runtime must be calculated from each system's complete native request
    # stream.  Sampling TCN on VLM clip boundaries is appropriate for the
    # common classification grid below, but it discards most TCN calls and
    # selects only a subset of fusion calls, biasing latency percentiles.
    native_model_latencies: dict[str, list[float]] = {
        "TCN-only": [
            float(row["tcn_inference_ms"])
            for row in tcn_records
            if isinstance(row.get("tcn_inference_ms"), (int, float))
            and float(row["tcn_inference_ms"]) > 0.0
        ],
        "VLM-only": [
            float(row.get("qwen_model_inference_ms") or 0.0)
            for row in vlm_records
            if isinstance(row.get("qwen_model_inference_ms"), (int, float))
            and float(row["qwen_model_inference_ms"]) > 0.0
        ],
        "TCN–VLM–YOLO": [
            float(row["model_ms"])
            for row in fusion_records
            if isinstance(row.get("model_ms"), (int, float))
            and float(row["model_ms"]) > 0.0
        ],
    }

    windows: list[tuple[float, float, dict[str, Any]]] = []
    for record in vlm_records:
        start = record.get("event_video_start_sec")
        end = record.get("event_video_end_sec", record.get("video_time"))
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        start, end = float(start), min(float(end), duration_sec)
        if end <= start or end < float(tcn_records[0]["timestamp"]):
            continue
        windows.append((start, end, record))
    if not windows:
        raise ValueError("세 시스템에 공통으로 사용할 VLM 시간 구간이 없습니다.")

    grid_rows: dict[str, list[dict[str, Any]]] = {
        "TCN-only": [],
        "VLM-only": [],
        "TCN–VLM–YOLO": [],
    }
    for start, end, vlm_record in windows:
        midpoint = (start + end) / 2.0
        truth = _truth_at(midpoint, truth_intervals)
        if truth is None:
            continue
        tcn_candidates = [
            row for row in tcn_records
            if isinstance(row.get("timestamp"), (int, float))
            and start < float(row["timestamp"]) <= end + 1e-9
        ]
        if not tcn_candidates:
            continue
        tcn = tcn_candidates[-1]
        tcn_ms = float(tcn.get("tcn_inference_ms") or 0.0)
        grid_rows["TCN-only"].append({
            "window_start_sec": start,
            "window_end_sec": end,
            "ground_truth": truth,
            "prediction": str(tcn.get("driver_response", "active")),
            "alert_time_sec": float(tcn["timestamp"]) + tcn_ms / 1000.0,
            "model_ms": tcn_ms,
        })

        bridge = vlm_record.get("bridge_result")
        state = bridge.get("driver_state") if isinstance(bridge, dict) else None
        vlm_prediction = (
            str(state.get("driver_response", "active"))
            if isinstance(state, dict) else "active"
        )
        vlm_e2e = float(vlm_record.get("event_end_to_end_sec") or 0.0)
        vlm_ms = float(vlm_record.get("qwen_model_inference_ms") or 0.0)
        grid_rows["VLM-only"].append({
            "window_start_sec": start,
            "window_end_sec": end,
            "ground_truth": truth,
            "prediction": vlm_prediction,
            "alert_time_sec": end + vlm_e2e,
            "model_ms": vlm_ms,
        })

        candidates = [
            row for row in fusion_records
            if isinstance(row.get("timestamp_sec"), (int, float))
            and start < float(row["timestamp_sec"]) <= end + 1e-9
        ]
        positive_candidates = [row for row in candidates if row["prediction"] == POSITIVE]
        if positive_candidates:
            selected = min(positive_candidates, key=lambda row: float(row["timestamp_sec"]))
            fusion_prediction = POSITIVE
            fusion_ms = float(selected.get("model_ms") or 0.0)
            fusion_alert = float(selected["timestamp_sec"]) + fusion_ms / 1000.0
        else:
            fusion_prediction = "active"
            fusion_ms = 0.0
            fusion_alert = end
        grid_rows["TCN–VLM–YOLO"].append({
            "window_start_sec": start,
            "window_end_sec": end,
            "ground_truth": truth,
            "prediction": fusion_prediction,
            "alert_time_sec": fusion_alert,
            "model_ms": fusion_ms,
        })

    observation_start = min(row["window_start_sec"] for row in grid_rows["TCN-only"])
    observation_end = max(row["window_end_sec"] for row in grid_rows["TCN-only"])
    observation_duration = observation_end - observation_start
    observed_truth_events = [
        (max(start, observation_start), min(end, observation_end))
        for start, end in truth_events
        if start < observation_end and end > observation_start
    ]
    systems: list[tuple[str, dict[str, Any]]] = []
    for name, rows in grid_rows.items():
        binary = _binary_summary(rows)
        predicted_events = _prediction_events(rows)
        event = _score_events_one_to_one(
            predicted_events, observed_truth_events, observation_duration
        )
        model_latencies = native_model_latencies[name]
        nvr = {
            key: binary[key] for key in ("tp", "fp", "fn", "precision", "recall", "f1")
        }
        summary: dict[str, Any] = {
            "title": name,
            "evaluation_type": "common_1p5s_binary_grid",
            "comparison_basis": "same 1.5-second windows; NVR vs non-NVR",
            "matched_samples": len(rows),
            "duration_sec": observation_duration,
            "observation_window_sec": [observation_start, observation_end],
            "accuracy": binary["accuracy"],
            "macro_f1": binary["f1"],
            "per_class": {POSITIVE: nvr},
            "no_visible_response": nvr,
            "event_metrics": event,
            "operational": {
                "model_latency_ms": {
                    "mean": sum(model_latencies) / len(model_latencies) if model_latencies else None,
                    "p50": _percentile(model_latencies, 0.50),
                    "p95": _percentile(model_latencies, 0.95),
                }
            },
        }
        systems.append((name, summary))
    audit = {
        "grid_seconds": COMMON_GRID_SEC,
        "sample_count_per_system": {name: len(rows) for name, rows in grid_rows.items()},
        "observation_window_sec": [observation_start, observation_end],
        "event_min_overlap_sec": EVENT_MIN_OVERLAP_SEC,
        "fusion_default_when_no_candidate": "non-NVR",
        "tcn_source": str(tcn_jsonl),
        "vlm_source": str(vlm_jsonl),
        "fusion_source": str(fusion_csv),
        "vlm_completed_run_records": len(vlm_records),
        "model_latency_basis": (
            "native request streams: all TCN forward passes, all completed "
            "VLM-only calls, and all candidate-triggered fusion calls"
        ),
    }
    return systems, audit


def build_fusion_summary(
    raw_rows: list[dict[str, Any]],
    duration_sec: float,
    ground_truth_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Summarize the candidate-triggered fusion CSV without mixing runs."""
    rows = normalize_rows(raw_rows)
    classification = classification_summary(rows)
    incorrect_samples = len(rows) - sum(
        classification["confusion_matrix"][index][index]
        for index in range(len(classification["confusion_matrix"]))
    )

    event_groups: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        event_id = str(row.get("event_id") or index + 1)
        event_groups.setdefault(event_id, []).append(row)
    predicted_events: list[tuple[float, float]] = []
    for group in event_groups.values():
        if not any(row["prediction"] == POSITIVE for row in group):
            continue
        timestamps = [
            float(row["timestamp_sec"])
            for row in group
            if isinstance(row.get("timestamp_sec"), (int, float))
        ]
        if timestamps:
            predicted_events.append((min(timestamps) - 0.5, max(timestamps) + 0.5))

    truth_events = _ground_truth_events(ground_truth_path, duration_sec)
    matched_truth = [
        any(_overlap(truth, prediction) for prediction in predicted_events)
        for truth in truth_events
    ]
    matched_prediction = [
        any(_overlap(prediction, truth) for truth in truth_events)
        for prediction in predicted_events
    ]
    tp = sum(matched_truth)
    fn = len(truth_events) - tp
    fp = len(predicted_events) - sum(matched_prediction)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None
        and recall is not None
        and precision + recall
        else None
    )
    model_latencies = [
        float(row["model_ms"])
        for row in rows
        if isinstance(row.get("model_ms"), (int, float))
    ]
    summary: dict[str, Any] = {
        "title": "TCN–VLM–YOLO",
        "evaluation_type": "candidate_sampled",
        "matched_samples": len(rows),
        "duration_sec": duration_sec,
        "observation_window_sec": [0.0, duration_sec],
        **classification,
        "no_visible_response": classification["per_class"][POSITIVE],
        "overall_state_errors": {
            "count": incorrect_samples,
            "rate": incorrect_samples / len(rows),
        },
        "event_metrics": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fp_per_second": fp / duration_sec,
            "definition": "one decision per unique TCN/YOLO candidate event",
        },
        "operational": {
            "calls_per_min": len(rows) / (duration_sec / 60.0),
            "candidate_events_per_min": len(event_groups) / (duration_sec / 60.0),
            "model_latency_ms": {
                "mean": sum(model_latencies) / len(model_latencies),
                "p50": _percentile(model_latencies, 0.50),
                "p95": _percentile(model_latencies, 0.95),
            },
            "end_to_end_latency_ms": {"mean": None, "p95": None},
        },
    }
    return summary, rows


def _display_path(path: Path) -> str:
    """Show repository files relative to the clone for portable reports."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_comparison_csv(
    systems: list[tuple[str, Mapping[str, Any]]], output_path: Path
) -> None:
    columns = (
        "system",
        "evaluation_scope",
        "n",
        "binary_accuracy",
        "nvr_precision",
        "nvr_recall",
        "nvr_f1",
        "event_precision",
        "event_recall",
        "event_f1",
        "event_false_alarm_count",
        "event_false_alarms_per_hour",
        "detection_latency_p50_sec",
        "detection_latency_p95_sec",
        "model_latency_mean_ms",
        "model_latency_p50_ms",
        "model_latency_p95_ms",
    )
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for name, summary in systems:
            nvr = summary["per_class"][POSITIVE]
            event = summary.get("event_metrics", {})
            operation = summary.get("operational", {})
            latency = operation.get("model_latency_ms", {})
            detection_latency = event.get("detection_latency_sec", {})
            writer.writerow(
                {
                    "system": name,
                    "evaluation_scope": summary.get("evaluation_type"),
                    "n": summary.get("matched_samples"),
                    "binary_accuracy": summary.get("accuracy"),
                    "nvr_precision": nvr.get("precision"),
                    "nvr_recall": nvr.get("recall"),
                    "nvr_f1": nvr.get("f1"),
                    "event_precision": event.get("precision"),
                    "event_recall": event.get("recall"),
                    "event_f1": event.get("f1"),
                    "event_false_alarm_count": event.get("fp"),
                    "event_false_alarms_per_hour": event.get("fp_per_hour"),
                    "detection_latency_p50_sec": detection_latency.get("p50"),
                    "detection_latency_p95_sec": detection_latency.get("p95"),
                    "model_latency_mean_ms": latency.get("mean"),
                    "model_latency_p50_ms": latency.get("p50"),
                    "model_latency_p95_ms": latency.get("p95"),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=286.9736842105263,
        help="evaluated video duration used for rates and event-window clipping",
    )
    parser.add_argument(
        "--tcn-jsonl",
        type=Path,
        default=TCN_STATES_JSONL,
        help="TCN-only state JSONL (tcn_driver_states_*.jsonl)",
    )
    parser.add_argument(
        "--vlm-jsonl", type=Path, default=VLM_YOLO_JSONL,
        help="completed YOLO VLM-only clip JSONL",
    )
    parser.add_argument(
        "--fusion-csv", type=Path, default=FUSION_CSV,
        help="TCN-augmented VLM matched CSV",
    )
    parser.add_argument(
        "--ground-truth", type=Path, default=GT_PATH,
        help="time-indexed driver-response ground truth CSV",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="directory for per-system metrics, figures and the comparison table",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="exit successfully when one of the three runs is not available yet",
    )
    args = parser.parse_args()

    try:
        tcn_jsonl = args.tcn_jsonl.resolve()
        vlm_jsonl = args.vlm_jsonl.resolve()
        fusion_csv = args.fusion_csv.resolve()
        ground_truth = args.ground_truth.resolve()
        for label, path in (
            ("TCN-only JSONL", tcn_jsonl),
            ("VLM-only JSONL", vlm_jsonl),
            ("TCN-augmented fusion CSV", fusion_csv),
            ("ground truth", ground_truth),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} 파일이 없습니다: {path}")
        vlm_records = _latest_complete_vlm_run(vlm_jsonl, args.duration_sec)
    except (OSError, ValueError) as exc:
        if args.allow_missing:
            print(f"종합 그래프 갱신 보류: {exc}")
            return 0
        raise

    output_dir = args.output_dir.resolve()
    tcn_dir = output_dir / "tcn_only"
    vlm_dir = output_dir / "vlm_only"
    fusion_dir = output_dir / "tcn_vlm_yolo"
    for directory in (tcn_dir, vlm_dir, fusion_dir):
        directory.mkdir(parents=True, exist_ok=True)

    evaluate_jsonl(
        tcn_jsonl,
        ground_truth,
        tcn_dir,
        source="tcn",
        title="TCN-only",
        observation_duration_sec=args.duration_sec,
    )
    evaluate_jsonl(
        vlm_jsonl,
        ground_truth,
        vlm_dir,
        source="vlm",
        title="VLM-only",
        last_records=len(vlm_records),
        observation_duration_sec=args.duration_sec,
    )

    fusion_raw_rows = read_matched_csv(fusion_csv)
    fusion_summary, fusion_rows = build_fusion_summary(
        fusion_raw_rows, args.duration_sec, ground_truth
    )
    (fusion_dir / "tcn-vlm-yolo_metrics.json").write_text(
        json.dumps(fusion_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    render_evaluation_figure(
        fusion_summary, fusion_rows, fusion_dir / "tcn-vlm-yolo_metrics.png"
    )

    systems, comparison_audit = build_common_grid_comparison(
        args.duration_sec,
        tcn_jsonl,
        vlm_jsonl,
        fusion_csv,
        ground_truth,
    )
    for key in ("tcn_source", "vlm_source", "fusion_source"):
        comparison_audit[key] = _display_path(Path(comparison_audit[key]))
    (output_dir / "system_comparison_audit.json").write_text(
        json.dumps(
            {
                "audit": comparison_audit,
                "systems": {name: summary for name, summary in systems},
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    render_system_comparison(systems, output_dir / "system_comparison.png")
    _write_comparison_csv(systems, output_dir / "system_comparison.csv")
    def cell(value: Any, width: int, digits: int = 3) -> str:
        # Precision/recall are undefined (None) when a run has no NVR windows.
        return f"{value:>{width}.{digits}f}" if value is not None else f"{'-':>{width}}"

    print()
    print(f"{'system':<14} {'acc':>6} {'P':>6} {'R':>6} {'F1':>6} "
          f"{'event F1':>9} {'event FP':>9} {'model ms':>9}")
    for name, summary in systems:
        nvr = summary["per_class"][POSITIVE]
        event = summary["event_metrics"]
        latency = summary["operational"]["model_latency_ms"]["mean"]
        print(f"{name:<14} {cell(summary['accuracy'], 6)} {cell(nvr['precision'], 6)} "
              f"{cell(nvr['recall'], 6)} {cell(nvr['f1'], 6)} {cell(event['f1'], 9)} "
              f"{event['fp']:>9d} {cell(latency, 9, 1)}")
    print()
    print(f"비교 그래프 저장: {_display_path(output_dir / 'system_comparison.png')}")
    print(f"비교 수치 저장: {_display_path(output_dir / 'system_comparison.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
