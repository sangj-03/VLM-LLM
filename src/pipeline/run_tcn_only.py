#!/usr/bin/env python3
"""TCN이 직접 분류한 운전자 상태를 기록하는 전용 실행기.

``run_tcn_vlm_yolo.py``는 변경하지 않는다. 이 파일은 해당 파일의
TCN 특징 추출·앙상블 추론을 재사용하지만, VLM/브리지 호출 결과는 상태 결정에
사용하지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
BASE_RUNNER = SCRIPT_DIR / "run_tcn_vlm_yolo.py"
EVALUATION_DIR = SRC_DIR / "evaluation"
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))
from driver_state_metrics import evaluate_jsonl


def load_base_runner() -> Any:
    """Load the unmodified legacy runner under an isolated module name."""
    if not BASE_RUNNER.is_file():
        raise FileNotFoundError(f"기준 실행 파일이 없습니다: {BASE_RUNNER}")
    spec = importlib.util.spec_from_file_location("tcn_base_runner", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"기준 실행 파일을 불러올 수 없습니다: {BASE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pose_fields(features: dict[str, float]) -> dict[str, Any]:
    """Expose the head/body features used by the TCN with cautious labels."""
    pitch = features.get("head_pitch", 0.0)
    yaw = features.get("head_yaw", 0.0)
    roll = features.get("head_roll", 0.0)
    face_valid = features.get("face_valid", 0.0) >= 0.5
    pose_reliable = features.get("pose_reliability", 0.0) >= 0.5
    head_to_shoulder_dy = features.get("head_to_shoulder_dy", 0.0)
    upper_body_motion = features.get("upper_body_motion", 0.0)

    # These are display labels derived from the TCN inputs, not extra TCN
    # model classes. Raw values are recorded so the cutoffs remain auditable.
    head_pose = (
        "unknown" if not face_valid
        else "upright" if max(abs(pitch), abs(yaw), abs(roll)) < 30.0
        else "non_upright"
    )
    upper_body_posture = (
        "unknown" if not pose_reliable
        else "upright" if -2.5 <= head_to_shoulder_dy <= -0.15
        else "non_upright"
    )
    return {
        "head_pose": head_pose,
        "head_pitch_deg": round(pitch, 3),
        "head_yaw_deg": round(yaw, 3),
        "head_roll_deg": round(roll, 3),
        "upper_body_posture": upper_body_posture,
        "head_to_shoulder_dy": round(head_to_shoulder_dy, 6),
        "upper_body_motion": round(upper_body_motion, 6),
        "pose_reliability": round(features.get("pose_reliability", 0.0), 6),
    }


def install_feature_snapshot(module: Any) -> dict[str, float]:
    """Keep the latest 32-D feature values without changing the base runner."""
    latest: dict[str, float] = {}
    original_extract = module.StreamingFeatureExtractor.extract

    def extract_with_snapshot(self: Any, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        vector, diagnostics = original_extract(self, *args, **kwargs)
        latest.clear()
        latest.update({
            name: float(vector[index])
            for index, name in enumerate(module.TCN_FEATURE_NAMES)
        })
        return vector, diagnostics

    module.StreamingFeatureExtractor.extract = extract_with_snapshot
    return latest


def install_inference_timer(module: Any) -> dict[str, float]:
    """Measure one primary TCN ensemble prediction without changing its math."""
    latest = {"tcn_inference_ms": 0.0}
    original_predict = module.StreamingTCN.predict

    def predict_with_timer(self: Any, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        result = original_predict(self, *args, **kwargs)
        latest["tcn_inference_ms"] = (time.perf_counter() - started) * 1000.0
        return result

    module.StreamingTCN.predict = predict_with_timer
    return latest


def parse_tcn_only_option() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--tcn-reduced-threshold",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--tcn-ground-truth-file",
        type=Path,
        default=None,
        help="TCN 3상태 정확도 평가용 시간 구간 GT CSV",
    )
    parser.add_argument(
        "--tcn-accuracy-plot-file",
        type=Path,
        default=None,
        help="TCN 정확도·추론시간 그래프 PNG 경로",
    )
    parser.add_argument(
        "--tcn-matched-file",
        type=Path,
        default=None,
        help="TCN 예측/GT 매칭 CSV 경로",
    )
    parser.add_argument(
        "--evaluate-tcn-states-file",
        type=Path,
        default=None,
        help="이미 생성된 TCN 상태 JSONL만 평가하고 그래프/CSV를 생성",
    )
    option, remaining = parser.parse_known_args()
    return option, remaining


def install_tcn_state_recorder(
    module: Any, state_path: Path,
    latest_features: dict[str, float], latest_timing: dict[str, float],
    on_tcn_state: Any | None = None,
) -> Any:
    """Record the direct three-class TCN output without state thresholds."""
    original_writer = csv.DictWriter
    state_stream = state_path.open("w", encoding="utf-8")

    class RecordingDictWriter:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._writer = original_writer(*args, **kwargs)

        def writeheader(self) -> Any:
            return self._writer.writeheader()

        def writerow(self, row: dict[str, Any]) -> Any:
            result = self._writer.writerow(row)
            probability = row.get("tcn_probability")
            timestamp = row.get("timestamp")
            state = row.get("tcn_direct_state")
            if isinstance(probability, (int, float)) and isinstance(state, str) and state:
                record = {
                    "timestamp": round(float(timestamp), 3),
                    "driver_response": state,
                    "raw_argmax_state": row.get("tcn_raw_argmax_state", state),
                    "state_decision": (
                        "argmax_mean_softmax"
                        if state in module.DRIVER_RESPONSE_LABELS
                        else "unobservable_observation_quality_gate"
                    ),
                    "observable": row.get("tcn_observable", ""),
                    "observation_quality": row.get("tcn_observation_quality", ""),
                    "state_probabilities": {
                        "active": row.get("tcn_active_probability"),
                        "reduced": row.get("tcn_reduced_probability"),
                        "no_visible_response": row.get("tcn_nvr_probability"),
                    },
                    "control_engagement": {
                        "engaged_probability": row.get("control_engaged_probability", ""),
                        "disengaged_probability": row.get("control_disengaged_probability", ""),
                        "observable": row.get("control_observable", ""),
                        "observation_quality": row.get("control_observation_quality", ""),
                        "disengaged_candidate": row.get("control_disengaged_candidate", ""),
                        "reduced_confirmed": row.get("control_reduced_confirmed", ""),
                        "confirmation_windows": row.get("control_confirmation_windows", ""),
                        "operational": bool(row.get("control_operational", False)),
                    },
                    "nvr_probability": round(float(probability), 6),
                    "nvr_candidate": row.get("tcn_nvr_candidate", ""),
                    "transition_probability": row.get("tcn_transition_probability"),
                    "transition_threshold": row.get("tcn_transition_threshold"),
                    "transition_horizon_sec": row.get("tcn_transition_horizon_sec"),
                    "transition_operational": bool(row.get("tcn_transition_operational")),
                    "tcn_inference_ms": round(
                        float(latest_timing["tcn_inference_ms"]), 3
                    ),
                    **pose_fields(latest_features),
                }
                state_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                state_stream.flush()
                if on_tcn_state is not None:
                    on_tcn_state(
                        float(record["timestamp"]),
                        state,
                        record["state_probabilities"],
                    )
                print(
                    "TCN 운전자 상태: "
                    f"t={record['timestamp']:.3f}s {state} "
                    f"(active={record['state_probabilities']['active']:.3f}, "
                    f"reduced={record['state_probabilities']['reduced']:.3f}, "
                    f"NVR={record['state_probabilities']['no_visible_response']:.3f}, "
                    f"3초 전이(실험)={record['transition_probability']:.3f}) "
                    f"inference_ms={record['tcn_inference_ms']:.2f} "
                    f"observable={record['observable']} "
                    f"quality={record['observation_quality']} "
                    f"control_candidate={record['control_engagement']['disengaged_candidate']} "
                    f"control_reduced={record['control_engagement']['reduced_confirmed']} "
                    f"head_pose={record['head_pose']} "
                    f"upper_body_posture={record['upper_body_posture']}",
                    flush=True,
                )
            return result

        def __getattr__(self, name: str) -> Any:
            return getattr(self._writer, name)

    module.csv.DictWriter = RecordingDictWriter
    return state_stream


def make_tcn_only_preview(
    module: Any, enabled: bool, *, direct_argmax: bool = True,
) -> Any:
    """Show the three state probabilities and the actual TCN verdict."""

    class TCNOnlyPreview(module.ImagePreview):
        def __init__(self, preview_enabled: bool) -> None:
            super().__init__(preview_enabled)
            self.tcn_live_result: str | None = None
            self.state_history: deque[tuple[float, dict[str, float]]] = deque(maxlen=240)

        def record_tcn_result(
            self, video_time: float, state: str,
            state_probabilities: dict[str, Any],
        ) -> None:
            if self.enabled:
                self.tcn_live_result = str(state)
                probabilities = {
                    name: float(value)
                    for name, value in state_probabilities.items()
                    if name in module.DRIVER_RESPONSE_LABELS
                    and isinstance(value, (int, float))
                }
                self.state_history.append((float(video_time), probabilities))

        # The base video loop still visits its legacy event hooks.  TCN-only
        # does not submit VLM work, so ignore them rather than retaining a
        # synthetic VLM request/result in the preview state.
        def record_vlm_request(self, *_args: Any, **_kwargs: Any) -> None:
            return

        def record_vlm_response(self, *_args: Any, **_kwargs: Any) -> None:
            return

        def finish_vlm_inference(self) -> None:
            return

        def _append_live_dashboard(self, image: Any) -> Any:
            """Render state scores without the shared event-threshold chart."""
            import numpy as np

            cv2 = self.cv2
            width = image.shape[1]
            left, right, top, chart_bottom = 50, max(52, width - 14), 34, 149
            chart_width = max(1, right - left)
            now_time = self.state_history[-1][0] if self.state_history else 0.0
            start_time = max(0.0, now_time - 12.0)
            end_time = start_time + 12.0
            values = [item for item in self.state_history if item[0] >= start_time]
            latest = values[-1][1] if values else {}
            series = (
                ("active", "Active score", (70, 190, 70)),
                ("reduced", "Reduced score", (0, 210, 240)),
                ("no_visible_response", "No visible response score", (45, 45, 230)),
            )
            # Wrap the legend so all three scores remain visible on narrow videos.
            legend_items = []
            legend_x, legend_y = left, 180
            for name, label, color in series:
                score = latest.get(name)
                text = f"{label}: {score:.3f}" if score is not None else f"{label}: --"
                text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0][0]
                item_width = 21 + text_width
                if legend_x > left and legend_x + item_width > right:
                    legend_x, legend_y = left, legend_y + 20
                legend_items.append((legend_x, legend_y, text, color))
                legend_x += item_width + 18

            status_top = legend_y + 15
            panel = np.full((status_top + 61, width, 3), (20, 20, 20), dtype=np.uint8)
            cv2.putText(panel, "TCN STATE SCORES", (left, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (220, 220, 220), 1, cv2.LINE_AA)
            for fraction, label in ((1.0, "1.0"), (0.5, "0.5"), (0.0, "0.0")):
                y = int(round(chart_bottom - fraction * (chart_bottom - top)))
                cv2.line(panel, (left, y), (right, y), (65, 65, 65), 1, cv2.LINE_AA)
                cv2.putText(panel, label, (10, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.38, (190, 190, 190), 1, cv2.LINE_AA)
            for tick_index in range(5):
                fraction = tick_index / 4.0
                x = left + int(round(fraction * chart_width))
                label = f"{start_time + fraction * (end_time - start_time):.1f}s"
                label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)[0][0]
                label_x = max(left, min(right - label_width, x - label_width // 2))
                cv2.putText(panel, label, (label_x, 163), cv2.FONT_HERSHEY_SIMPLEX,
                            0.34, (185, 185, 185), 1, cv2.LINE_AA)

            for name, _label, color in series:
                points = [
                    (
                        left + int(round((timestamp - start_time) / 12.0 * chart_width)),
                        int(round(chart_bottom - max(0.0, min(1.0, probabilities[name]))
                                  * (chart_bottom - top))),
                    )
                    for timestamp, probabilities in values if name in probabilities
                ]
                if len(points) > 1:
                    cv2.polylines(
                        panel, [np.asarray(points, dtype=np.int32)], False,
                        color, 2, cv2.LINE_AA,
                    )
            for x, y, text, color in legend_items:
                cv2.line(panel, (x, y), (x + 16, y), color, 2, cv2.LINE_AA)
                cv2.putText(panel, text, (x + 21, y + 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.40, color, 1, cv2.LINE_AA)

            labels = {
                "active": "ACTIVE",
                "reduced": "REDUCED",
                "no_visible_response": "NO VISIBLE RESPONSE",
                "unobservable": "UNOBSERVABLE",
            }
            colors = {name: color for name, _label, color in series}
            colors["unobservable"] = (170, 170, 170)
            result = self.tcn_live_result
            if result in labels:
                status_label = f"TCN RESULT: {labels[result]}"
                if result in latest:
                    status_label += f" ({latest[result]:.3f})"
                status_color = colors[str(result)]
            else:
                status_label, status_color = "TCN WAITING FOR FIRST RESULT", (190, 190, 190)
            status_width = cv2.getTextSize(status_label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0][0]
            status_scale = 0.62 * min(1.0, max(1, chart_width - 20) / max(1, status_width))
            cv2.rectangle(panel, (left, status_top), (right, status_top + 28), (42, 42, 42), -1)
            cv2.rectangle(panel, (left, status_top), (right, status_top + 28), (85, 85, 85), 1)
            cv2.putText(panel, status_label, (left + 10, status_top + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        status_scale, status_color, 2, cv2.LINE_AA)
            decision_note = (
                "TCN RESULT = highest score"
                if direct_argmax else "TCN RESULT uses visibility and control checks"
            )
            if result == "unobservable":
                decision_note = "Waiting for a reliable observation"
            cv2.putText(panel, decision_note, (left, status_top + 47), cv2.FONT_HERSHEY_SIMPLEX,
                        0.40, (190, 190, 190), 1, cv2.LINE_AA)
            return cv2.vconcat([image, panel])

    return TCNOnlyPreview(enabled)


def evaluate_tcn_states(
    module: Any,
    state_path: Path,
    ground_truth_path: Path,
    plot_path: Path | None,
    matched_path: Path | None,
) -> None:
    """Create accuracy and inference-time artifacts from TCN state JSONL."""
    annotations = module.load_ground_truth(ground_truth_path)
    intervals: list[tuple[float, float, str, str]] = []
    for clip_name, response in annotations.items():
        interval = module.parse_ground_truth_clip_interval(clip_name)
        if interval is not None:
            intervals.append((interval[0], interval[1], clip_name, response))
    if not intervals:
        raise ValueError("TCN 정확도 평가에는 시간 구간 clip_name GT가 필요합니다.")
    intervals.sort()

    rows: list[dict[str, Any]] = []
    all_records = 0
    unobservable_records = 0
    with state_path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            all_records += 1
            timestamp = record.get("timestamp")
            prediction = record.get("driver_response")
            if prediction == "unobservable":
                unobservable_records += 1
                continue
            if not isinstance(timestamp, (int, float)) or prediction not in module.DRIVER_RESPONSE_LABELS:
                continue
            candidates = [item for item in intervals if item[0] <= timestamp < item[1]]
            if not candidates:
                earlier = [item for item in intervals if item[0] <= timestamp]
                candidates = [max(earlier, key=lambda item: item[0])] if earlier else [intervals[0]]
            _, _, clip_name, truth = max(candidates, key=lambda item: item[0])
            rows.append({
                "timestamp_sec": round(float(timestamp), 3),
                "ground_truth_clip": clip_name,
                "ground_truth": truth,
                "tcn_prediction": prediction,
                "correct": int(prediction == truth),
                "tcn_probability": record.get("nvr_probability", ""),
                "tcn_inference_ms": record.get("tcn_inference_ms", ""),
                "head_pose": record.get("head_pose", ""),
                "upper_body_posture": record.get("upper_body_posture", ""),
            })
    if not rows:
        print("TCN 상태가 아직 없어 정확도 그래프 생성을 건너뜁니다.", flush=True)
        return

    print(
        f"TCN 관측 가능 비율: {(all_records - unobservable_records) / max(all_records, 1):.1%} "
        f"({all_records - unobservable_records}/{all_records}); 판단보류={unobservable_records}",
        flush=True,
    )

    output_plot = plot_path or state_path.with_name(state_path.stem + "_accuracy.png")
    output_csv = matched_path or state_path.with_name(state_path.stem + "_matched.csv")
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = tuple(module.DRIVER_RESPONSE_LABELS)
    totals = {label: 0 for label in labels}
    corrects = {label: 0 for label in labels}
    for row in rows:
        truth = str(row["ground_truth"])
        if truth in totals:
            totals[truth] += 1
            corrects[truth] += int(row["correct"])
    overall = 100.0 * sum(int(row["correct"]) for row in rows) / len(rows)
    values = [
        100.0 * corrects[label] / totals[label] if totals[label] else 0.0
        for label in labels
    ] + [overall]
    names = ["정상 반응", "반응 저하", "무반응", "전체"]
    latency_rows = [
        row for row in rows if isinstance(row["tcn_inference_ms"], (int, float))
    ]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (accuracy_axis, latency_axis) = plt.subplots(
        2, 1, figsize=(10, 8), constrained_layout=True
    )
    bars = accuracy_axis.bar(names, values, color="#3b82f6")
    accuracy_axis.set_ylim(0, 100)
    accuracy_axis.set_ylabel("정확도 (%)")
    accuracy_axis.set_title(f"TCN 운전자 상태 정확도 (n={len(rows)})")
    accuracy_axis.grid(axis="y", linestyle="--", alpha=0.35)
    for index, bar in enumerate(bars):
        label = "overall" if index == len(labels) else labels[index]
        note = f"{values[index]:.1f}% ({corrects[label]}/{totals[label]})" if label != "overall" else f"{overall:.1f}%"
        accuracy_axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                           note, ha="center", va="bottom", fontsize=9)

    if latency_rows:
        latency_axis.plot(
            [float(row["timestamp_sec"]) for row in latency_rows],
            [float(row["tcn_inference_ms"]) for row in latency_rows],
            linewidth=1.2, color="#ef4444",
        )
        mean_ms = sum(float(row["tcn_inference_ms"]) for row in latency_rows) / len(latency_rows)
        latency_axis.axhline(mean_ms, color="#111827", linestyle="--", label=f"mean {mean_ms:.2f} ms")
        latency_axis.legend()
    else:
        latency_axis.text(0.5, 0.5, "No inference timing in this JSONL run", ha="center", va="center")
    latency_axis.set_xlabel("영상 시간 (초)")
    latency_axis.set_ylabel("TCN 추론시간 (ms)")
    latency_axis.set_title("TCN 앙상블 추론시간")
    latency_axis.grid(alpha=0.35)
    figure.savefig(output_plot, dpi=160)
    plt.close(figure)
    print(f"TCN 정확도 그래프 저장: {output_plot}", flush=True)
    print(f"TCN 예측/GT CSV 저장: {output_csv}", flush=True)
    evaluate_jsonl(
        state_path,
        ground_truth_path,
        output_plot.parent,
        source="tcn",
        title="TCN-only",
    )
    # The observability gate is reported separately from the underlying state
    # classifier. This applies only to newer records that retain raw argmax.
    with state_path.open(encoding="utf-8") as stream:
        has_raw_argmax = any(
            isinstance(json.loads(line).get("raw_argmax_state"), str)
            for line in stream if line.strip()
        )
    if has_raw_argmax:
        evaluate_jsonl(
            state_path,
            ground_truth_path,
            output_plot.parent,
            source="tcn",
            title="TCN-only raw argmax",
            prediction_field="raw_argmax_state",
        )


def disable_vlm_paths(module: Any) -> None:
    """Ensure legacy event branches cannot call the VLM bridge or create media."""
    base_print = print

    def tcn_only_print(*values: Any, **kwargs: Any) -> None:
        message = str(kwargs.get("sep", " ")).join(str(value) for value in values)
        # These messages belong to the legacy VLM event branch. Its call is
        # replaced below with a local result, so do not imply that Qwen ran.
        if message.startswith("Qwen") or message.startswith("마지막 Qwen"):
            return
        base_print(*values, **kwargs)

    def no_media(_frames: Any, media_path: Path, *_args: Any, **_kwargs: Any) -> Path:
        return media_path

    def no_event_frames(*_args: Any, **_kwargs: Any) -> list[Any]:
        return []

    def local_tcn_result(
        _args: Any, _headers: Any, _source: Any, _media: Any, _observation_id: int,
        _video_time: float, probability: float, threshold: float, *_rest: Any,
    ) -> dict[str, Any]:
        del probability, threshold
        return {
            "driver_state": {"driver_response": "no_visible_response"},
            "tcn_only": True,
            "note": "event branch only; direct states are stored in tcn_driver_states JSONL",
        }

    module.encode_event_video = no_media
    module.encode_event_storyboard = no_media
    module.select_fixed_driver_box_event_frames = no_event_frames
    module.select_event_video_frames = no_event_frames
    module.post_tcn_event = local_tcn_result
    module.print_qwen_event_result = (
        lambda result, _metadata, _print_json: str(
            result.get("driver_state", {}).get("driver_response", "unknown")
        )
    )
    module.print = tcn_only_print


def main() -> int:
    tcn_options, remaining = parse_tcn_only_option()
    ground_truth_path = tcn_options.tcn_ground_truth_file
    if ground_truth_path is not None:
        ground_truth_path = ground_truth_path.resolve()
        if not ground_truth_path.is_file():
            raise FileNotFoundError(f"TCN GT CSV가 없습니다: {ground_truth_path}")
    sys.argv = [sys.argv[0], *remaining]
    module = load_base_runner()
    if tcn_options.evaluate_tcn_states_file is not None:
        if ground_truth_path is None:
            raise ValueError(
                "--evaluate-tcn-states-file에는 --tcn-ground-truth-file이 필요합니다."
            )
        evaluate_path = tcn_options.evaluate_tcn_states_file.resolve()
        if not evaluate_path.is_file():
            raise FileNotFoundError(f"TCN 상태 JSONL이 없습니다: {evaluate_path}")
        evaluate_tcn_states(
            module,
            evaluate_path,
            ground_truth_path,
            tcn_options.tcn_accuracy_plot_file,
            tcn_options.tcn_matched_file,
        )
        return 0
    args = module.parse_args()
    if args.video_path is None:
        raise ValueError("TCN 전용 상태 실행에는 --video-path MP4가 필요합니다.")
    ensemble_metadata = json.loads(
        args.tcn_ensemble_config.resolve().read_text(encoding="utf-8")
    )
    if ensemble_metadata.get("ensemble_type") not in (
        "mean_multitask_state_probability",
        "mean_observable_state_probability",
    ):
        raise ValueError(
            "TCN-only 직접 상태 판정에는 3상태 멀티태스크 ensemble이 필요합니다."
        )

    # The configured primary ensemble alone controls active/reduced/no-visible.
    # Disable optional onset/watchdog VLM paths regardless of legacy defaults.
    args.fast_onset_risk = False
    args.fall_onset_tcn_config = None
    args.fall_onset_tcn_ensemble_config = None
    args.visual_watchdog_sec = 0.0
    args.eye_uncertain_watchdog_sec = 0.0
    module.validate_args(args)

    source_video = args.video_path.resolve()
    run_stamp = f"{int(time.time())}_{int(round(args.video_start_sec * 1000)):010d}"
    state_path = (
        args.event_output_dir.resolve() / source_video.stem
        / f"tcn_driver_states_{run_stamp}.jsonl"
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    latest_features = install_feature_snapshot(module)
    latest_timing = install_inference_timer(module)
    disable_vlm_paths(module)
    direct_argmax = (
        str(ensemble_metadata.get("state_decision", "argmax_mean_softmax")).startswith("argmax")
        and not ensemble_metadata.get("control_ensemble_path")
    )
    preview = make_tcn_only_preview(
        module, args.show_image or args.video_preview, direct_argmax=direct_argmax,
    )
    state_stream = install_tcn_state_recorder(
        module, state_path, latest_features, latest_timing,
        preview.record_tcn_result,
    )
    decision_description = (
        "active/reduced/no_visible_response 평균 softmax 중 가장 큰 값으로 최종 상태를 선택합니다. "
        if direct_argmax else "설정된 관측 품질 및 제어 관여 규칙으로 최종 상태를 선택합니다. "
    )
    print(
        "TCN 전용 상태 판정 시작: VLM/브리지 호출 없이 실행합니다. "
        + decision_description
        + "그래프에는 세 상태의 확률을 표시합니다.",
        flush=True,
    )
    exit_code = 0
    try:
        exit_code = module.run_full_video_realtime(args, {}, preview)
    except KeyboardInterrupt:
        exit_code = 130
        print("TCN 실행을 중단했습니다. 현재까지의 결과를 평가합니다.", flush=True)
    finally:
        state_stream.close()
        print(f"TCN 3상태 결과 저장: {state_path}", flush=True)
    if ground_truth_path is not None:
        evaluate_tcn_states(
            module,
            state_path,
            ground_truth_path,
            tcn_options.tcn_accuracy_plot_file,
            tcn_options.tcn_matched_file,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
