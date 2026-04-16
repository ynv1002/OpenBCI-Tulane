from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import shlex
import subprocess
import sys
import time
import pickle
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.event_utils import EventWindowConfig, JawEventConfig, _linear_slope
    from analysis.jaw_trigger_rules import JawClickTrigger, JawClickTriggerConfig
    from analysis.live_jaw_click_detector import (
        ReplayEvaluationConfig,
        build_jaw_replay_dataset,
        build_replay_score_frame,
        evaluate_clicks_against_reference,
        extract_reference_onsets,
        run_trigger_strategy,
        summarize_strategy_performance,
        train_jaw_click_models,
    )
    from analysis.utils import (
        audit_all_sessions,
        ensure_output_dir,
        estimate_sampling,
        load_openbci_csv,
        preprocess_session_signals,
        write_json,
    )
else:
    from .event_utils import EventWindowConfig, JawEventConfig, _linear_slope
    from .jaw_trigger_rules import JawClickTrigger, JawClickTriggerConfig
    from .live_jaw_click_detector import (
        ReplayEvaluationConfig,
        build_jaw_replay_dataset,
        build_replay_score_frame,
        evaluate_clicks_against_reference,
        extract_reference_onsets,
        run_trigger_strategy,
        summarize_strategy_performance,
        train_jaw_click_models,
    )
    from .utils import (
        audit_all_sessions,
        ensure_output_dir,
        estimate_sampling,
        load_openbci_csv,
        preprocess_session_signals,
        write_json,
    )


DEFAULT_ARTIFACT_PATH = Path(__file__).resolve().parent / "outputs" / "realtime_clench_model.pkl"
DEFAULT_POLL_INTERVAL_SEC = 0.05
DEFAULT_COOLDOWN_SEC = 0.30
DEFAULT_WARMUP_SEC = 1.0
DEFAULT_RUNTIME_CLENCH_THRESHOLD = 0.65
DEFAULT_RUNTIME_REARM_THRESHOLD = 0.55
DEFAULT_RUNTIME_COOLDOWN_SEC = 0.25
DEFAULT_RUNTIME_SMOOTHING_WINDOWS = 1
DEFAULT_REPLAY_FLASH_SEC = 0.25


@dataclass(frozen=True)
class ReplayStep:
    timestamp_sec: float
    clench_probability: float
    smoothed_clench_probability: float
    onset_probability: float
    event_label: str
    emitted_click: bool
    trigger_reason: str


def default_trigger_config() -> JawClickTriggerConfig:
    return JawClickTriggerConfig(
        strategy_name="binary_clench_threshold",
        clench_probability_threshold=0.80,
        onset_probability_threshold=0.45,
        active_probability_threshold=0.55,
        rearm_clench_probability_threshold=0.50,
        cooldown_ms=300,
        minimum_separation_ms=300,
        smoothing_windows=1,
        hold_suppression=True,
        require_transition_from_inactive=False,
        minimum_clench_rise=0.0,
    )


def default_runtime_trigger_overrides() -> Dict[str, Any]:
    cooldown_ms = int(round(DEFAULT_RUNTIME_COOLDOWN_SEC * 1000.0))
    return {
        "strategy_name": "binary_clench_threshold",
        "clench_probability_threshold": DEFAULT_RUNTIME_CLENCH_THRESHOLD,
        "rearm_clench_probability_threshold": DEFAULT_RUNTIME_REARM_THRESHOLD,
        "cooldown_ms": cooldown_ms,
        "minimum_separation_ms": cooldown_ms,
        "smoothing_windows": DEFAULT_RUNTIME_SMOOTHING_WINDOWS,
        "hold_suppression": True,
        "require_transition_from_inactive": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time BrainFlow jaw clench detector for OpenBCI Cyton.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and export the jaw click model artifact.")
    train_parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_PATH, help="Artifact output path.")
    train_parser.add_argument("--window-sec", type=float, default=EventWindowConfig().window_sec)
    train_parser.add_argument("--overlap", type=float, default=EventWindowConfig().overlap)
    train_parser.add_argument("--smoothing-sec", type=float, default=JawEventConfig().smoothing_sec)
    train_parser.add_argument("--onset-sec", type=float, default=JawEventConfig().onset_duration_sec)
    train_parser.add_argument("--offset-sec", type=float, default=JawEventConfig().offset_duration_sec)
    train_parser.add_argument("--minimum-active-sec", type=float, default=JawEventConfig().minimum_active_duration_sec)
    train_parser.add_argument(
        "--minimum-peak-distance-sec",
        type=float,
        default=JawEventConfig().minimum_peak_distance_sec,
    )
    train_parser.add_argument("--inactive-quantile", type=float, default=JawEventConfig().inactive_quantile)
    train_parser.add_argument("--active-quantile", type=float, default=JawEventConfig().active_quantile)
    train_parser.add_argument("--threshold-mix", type=float, default=JawEventConfig().threshold_mix)
    train_parser.add_argument(
        "--release-threshold-mix",
        type=float,
        default=JawEventConfig().release_threshold_mix,
    )
    train_parser.add_argument("--threshold", type=float, default=default_trigger_config().clench_probability_threshold)
    train_parser.add_argument(
        "--rearm-threshold",
        type=float,
        default=default_trigger_config().rearm_clench_probability_threshold,
    )
    train_parser.add_argument("--cooldown-sec", type=float, default=DEFAULT_COOLDOWN_SEC)

    live_parser = subparsers.add_parser("live", help="Run the BrainFlow live clench loop.")
    live_parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_PATH, help="Saved model artifact.")
    live_parser.add_argument(
        "--board",
        type=str,
        default="cyton",
        choices=("cyton", "synthetic", "playback"),
        help="BrainFlow board preset.",
    )
    live_parser.add_argument("--serial-port", type=str, default="", help="Cyton serial port.")
    live_parser.add_argument("--playback-file", type=Path, help="Playback file for BrainFlow playback board.")
    live_parser.add_argument("--threshold", type=float, help="Override trigger threshold.")
    live_parser.add_argument("--rearm-threshold", type=float, help="Override trigger rearm threshold.")
    live_parser.add_argument(
        "--cooldown-sec",
        type=float,
        default=DEFAULT_RUNTIME_COOLDOWN_SEC,
        help="Trigger cooldown.",
    )
    live_parser.add_argument(
        "--poll-interval-sec",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SEC,
        help="Sleep time between loop iterations.",
    )
    live_parser.add_argument("--duration-sec", type=float, help="Optional max runtime for smoke tests.")
    live_parser.add_argument(
        "--on-detect-command",
        type=str,
        help="Optional command to run whenever a clench is detected.",
    )
    live_parser.add_argument("--verbose", action="store_true", help="Print periodic probabilities.")

    replay_parser = subparsers.add_parser("replay", help="Replay the detector on a recorded jaw CSV.")
    replay_parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_PATH, help="Saved model artifact.")
    replay_parser.add_argument("--csv", type=Path, required=True, help="Jaw CSV to replay.")
    replay_parser.add_argument("--threshold", type=float, help="Override trigger threshold.")
    replay_parser.add_argument("--rearm-threshold", type=float, help="Override trigger rearm threshold.")
    replay_parser.add_argument(
        "--cooldown-sec",
        type=float,
        default=DEFAULT_RUNTIME_COOLDOWN_SEC,
        help="Trigger cooldown.",
    )
    replay_parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open a small Tk replay window showing P(clench) and a detection indicator.",
    )
    replay_parser.add_argument("--verbose", action="store_true", help="Print probabilities during replay.")
    return parser.parse_args()


def build_event_config(args: argparse.Namespace) -> JawEventConfig:
    return JawEventConfig(
        smoothing_sec=float(args.smoothing_sec),
        onset_duration_sec=float(args.onset_sec),
        offset_duration_sec=float(args.offset_sec),
        minimum_active_duration_sec=float(args.minimum_active_sec),
        minimum_peak_distance_sec=float(args.minimum_peak_distance_sec),
        inactive_quantile=float(args.inactive_quantile),
        active_quantile=float(args.active_quantile),
        threshold_mix=float(args.threshold_mix),
        release_threshold_mix=float(args.release_threshold_mix),
    )


def build_trigger_config(threshold: float, rearm_threshold: float, cooldown_sec: float) -> JawClickTriggerConfig:
    base = default_trigger_config()
    return JawClickTriggerConfig(
        strategy_name=base.strategy_name,
        clench_probability_threshold=float(threshold),
        onset_probability_threshold=base.onset_probability_threshold,
        active_probability_threshold=base.active_probability_threshold,
        rearm_clench_probability_threshold=float(rearm_threshold),
        cooldown_ms=int(round(float(cooldown_sec) * 1000.0)),
        minimum_separation_ms=int(round(float(cooldown_sec) * 1000.0)),
        smoothing_windows=base.smoothing_windows,
        hold_suppression=base.hold_suppression,
        require_transition_from_inactive=base.require_transition_from_inactive,
        minimum_clench_rise=base.minimum_clench_rise,
        minimum_envelope_uv=base.minimum_envelope_uv,
    )


def _load_artifact(path: Path) -> Dict[str, Any]:
    with path.resolve().open("rb") as handle:
        artifact = pickle.load(handle)
    if not isinstance(artifact, dict) or "model_bundle" not in artifact:
        raise RuntimeError(f"Artifact at {path} is not a supported realtime clench model.")
    return artifact


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy(dtype=float)


def _build_feature_row(
    signal_window: np.ndarray,
    aggregate_rms_window: np.ndarray,
    aggregate_rms_smooth_window: np.ndarray,
    selected_channels: Sequence[str],
) -> Dict[str, float]:
    row: Dict[str, float] = {}
    for channel_index, channel_name in enumerate(selected_channels):
        values = signal_window[:, channel_index]
        feature_prefix = f"signal_{channel_name}"
        row[f"{feature_prefix}_mean"] = float(np.mean(values))
        row[f"{feature_prefix}_std"] = float(np.std(values))
        row[f"{feature_prefix}_rms"] = float(np.sqrt(np.mean(values**2)))
        row[f"{feature_prefix}_ptp"] = float(np.ptp(values))
        row[f"{feature_prefix}_var"] = float(np.var(values))
        row[f"{feature_prefix}_energy"] = float(np.mean(values**2))
        row[f"{feature_prefix}_abs_mean"] = float(np.mean(np.abs(values)))
        row[f"{feature_prefix}_slope"] = _linear_slope(values)

    row["aggregate_rms_mean"] = float(np.mean(aggregate_rms_window))
    row["aggregate_rms_std"] = float(np.std(aggregate_rms_window))
    row["aggregate_rms_max"] = float(np.max(aggregate_rms_window))
    row["aggregate_rms_delta"] = float(aggregate_rms_window[-1] - aggregate_rms_window[0])
    row["aggregate_rms_slope"] = _linear_slope(aggregate_rms_window)
    row["aggregate_rms_smooth_mean"] = float(np.mean(aggregate_rms_smooth_window))
    row["aggregate_rms_smooth_std"] = float(np.std(aggregate_rms_smooth_window))
    row["aggregate_rms_smooth_max"] = float(np.max(aggregate_rms_smooth_window))
    row["aggregate_rms_smooth_delta"] = float(aggregate_rms_smooth_window[-1] - aggregate_rms_smooth_window[0])
    row["aggregate_rms_smooth_slope"] = _linear_slope(aggregate_rms_smooth_window)
    return row


def _feature_vector_from_filtered_matrix(
    filtered_matrix: np.ndarray,
    feature_columns: Sequence[str],
    selected_channels: Sequence[str],
    smoothing_samples: int,
    window_samples: int,
) -> np.ndarray:
    aggregate_rms = np.sqrt(np.mean(filtered_matrix**2, axis=1))
    aggregate_rms_smooth = _rolling_mean(aggregate_rms, smoothing_samples)

    signal_window = filtered_matrix[-window_samples:, :]
    aggregate_window = aggregate_rms[-window_samples:]
    aggregate_smooth_window = aggregate_rms_smooth[-window_samples:]
    feature_row = _build_feature_row(
        signal_window=signal_window,
        aggregate_rms_window=aggregate_window,
        aggregate_rms_smooth_window=aggregate_smooth_window,
        selected_channels=selected_channels,
    )
    return np.asarray([[feature_row[column] for column in feature_columns]], dtype=float)


def _predict_scores(model_bundle: Dict[str, Any], feature_vector: np.ndarray) -> Tuple[Dict[str, float], str]:
    jaw_4state_model = model_bundle["jaw_4state_model"]
    jaw_4state_labels = list(model_bundle["jaw_4state_labels"])
    binary_model = model_bundle["binary_model"]
    binary_labels = list(model_bundle["binary_labels"])

    jaw_4state_probs = jaw_4state_model.predict_proba(feature_vector)[0]
    binary_probs = binary_model.predict_proba(feature_vector)[0]

    jaw_4state_map = {label: float(jaw_4state_probs[index]) for index, label in enumerate(jaw_4state_labels)}
    binary_map = {label: float(binary_probs[index]) for index, label in enumerate(binary_labels)}
    event_label = str(jaw_4state_model.predict(feature_vector)[0])

    scores = {
        "clench_probability": binary_map.get("CLENCH", 0.0),
        "non_clench_probability": binary_map.get("NON_CLENCH", 0.0),
        "inactive_probability": jaw_4state_map.get("INACTIVE", 0.0),
        "onset_probability": jaw_4state_map.get("ONSET", 0.0),
        "active_probability": jaw_4state_map.get("ACTIVE", 0.0),
        "offset_probability": jaw_4state_map.get("OFFSET", 0.0),
    }
    return scores, event_label


def _run_detection_command(command: str) -> None:
    subprocess.run(shlex.split(command), check=False)


def _board_id_from_name(board_name: str, board_ids: Any) -> int:
    if board_name == "cyton":
        return int(board_ids.CYTON_BOARD.value)
    if board_name == "synthetic":
        return int(board_ids.SYNTHETIC_BOARD.value)
    if board_name == "playback":
        return int(board_ids.PLAYBACK_FILE_BOARD.value)
    raise ValueError(f"Unsupported board name: {board_name}")


def _selected_board_rows(selected_channels: Sequence[str], eeg_rows: Sequence[int]) -> List[Tuple[str, int]]:
    rows: List[Tuple[str, int]] = []
    for channel_name in selected_channels:
        channel_index = int(channel_name.split("_", maxsplit=1)[-1]) - 1
        if channel_index < 0 or channel_index >= len(eeg_rows):
            raise RuntimeError(f"{channel_name} is not available on this board.")
        rows.append((channel_name, int(eeg_rows[channel_index])))
    return rows


def _extract_live_feature_vector(
    data: np.ndarray,
    board_rows: Sequence[Tuple[str, int]],
    sampling_rate: int,
    feature_columns: Sequence[str],
    selected_channels: Sequence[str],
    smoothing_samples: int,
    window_samples: int,
) -> np.ndarray:
    try:
        from brainflow.data_filter import DataFilter, FilterTypes, NoiseTypes
    except ModuleNotFoundError as exc:
        raise RuntimeError("brainflow is required for the live detector.") from exc

    filtered_columns: List[np.ndarray] = []
    for _, row_index in board_rows:
        channel = np.ascontiguousarray(np.array(data[row_index], dtype=np.float64))
        channel = channel - float(np.median(channel))
        DataFilter.remove_environmental_noise(channel, sampling_rate, NoiseTypes.SIXTY.value)
        DataFilter.perform_bandpass(
            channel,
            sampling_rate,
            20.0,
            min(100.0, float(sampling_rate) * 0.45),
            2,
            FilterTypes.BUTTERWORTH_ZERO_PHASE.value,
            0.0,
        )
        filtered_columns.append(channel)

    filtered_matrix = np.column_stack(filtered_columns)
    return _feature_vector_from_filtered_matrix(
        filtered_matrix=filtered_matrix,
        feature_columns=feature_columns,
        selected_channels=selected_channels,
        smoothing_samples=smoothing_samples,
        window_samples=window_samples,
    )


def train_and_save_artifact(args: argparse.Namespace) -> Dict[str, Any]:
    artifact_path = args.artifact.resolve()
    ensure_output_dir(artifact_path.parent)

    event_config = build_event_config(args)
    window_config = EventWindowConfig(window_sec=float(args.window_sec), overlap=float(args.overlap))
    trigger_config = build_trigger_config(
        threshold=float(args.threshold),
        rearm_threshold=float(args.rearm_threshold),
        cooldown_sec=float(args.cooldown_sec),
    )

    audits = audit_all_sessions()
    dataset_bundle = build_jaw_replay_dataset(audits, event_config, window_config)
    model_bundle = train_jaw_click_models(dataset_bundle)
    score_frame = build_replay_score_frame(dataset_bundle, model_bundle)
    reference_onsets = extract_reference_onsets(dataset_bundle["event_bundle"]["interval_frame"])
    click_frame = run_trigger_strategy(score_frame, trigger_config)
    evaluation_summary, _ = evaluate_clicks_against_reference(
        click_frame,
        reference_onsets,
        ReplayEvaluationConfig(),
    )
    trigger_summary = summarize_strategy_performance(
        evaluation_summary,
        trigger_config.strategy_name,
        trigger_config,
    )

    feature_columns = model_bundle["feature_columns"]
    window_frame = dataset_bundle["window_bundle"]["window_frame"]
    test_mask = window_frame["split_role"] == "test"
    feature_matrix = window_frame.loc[test_mask, feature_columns].to_numpy(dtype=float)
    jaw_4state_true = window_frame.loc[test_mask, "event_label"].to_numpy()
    jaw_4state_pred = model_bundle["jaw_4state_model"].predict(feature_matrix)
    binary_true = np.where(jaw_4state_true == "INACTIVE", "NON_CLENCH", "CLENCH")
    binary_pred = model_bundle["binary_model"].predict(feature_matrix)

    summary = {
        "artifact_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_channels": model_bundle["selected_channels"],
        "excluded_channels": model_bundle["excluded_channels"],
        "feature_columns": feature_columns,
        "window_config": asdict(window_config),
        "event_config": asdict(event_config),
        "trigger_config": asdict(trigger_config),
        "metrics": {
            "jaw_4state_accuracy": float(accuracy_score(jaw_4state_true, jaw_4state_pred)),
            "jaw_4state_macro_f1": float(f1_score(jaw_4state_true, jaw_4state_pred, average="macro")),
            "binary_accuracy": float(accuracy_score(binary_true, binary_pred)),
            "binary_macro_f1": float(f1_score(binary_true, binary_pred, average="macro")),
            "trigger_summary": trigger_summary,
        },
    }
    artifact = {**summary, "model_bundle": model_bundle}

    with artifact_path.open("wb") as handle:
        pickle.dump(artifact, handle)
    write_json(artifact_path.with_suffix(".json"), summary)
    return summary


def _print_training_summary(summary: Dict[str, Any], artifact_path: Path) -> None:
    trigger_summary = summary["metrics"]["trigger_summary"]
    print(f"Wrote model artifact: {artifact_path.resolve()}")
    print(f"Wrote summary: {(artifact_path.with_suffix('.json')).resolve()}")
    print(f"Selected channels: {', '.join(summary['selected_channels'])}")
    print(
        f"Window: {summary['window_config']['window_sec']:.3f}s | "
        f"overlap: {summary['window_config']['overlap']:.2f}"
    )
    print(
        "Model metrics: "
        f"binary_f1={summary['metrics']['binary_macro_f1']:.3f} "
        f"4state_f1={summary['metrics']['jaw_4state_macro_f1']:.3f}"
    )
    print(
        "Trigger metrics: "
        f"test_f1={trigger_summary['test_weighted_event_f1']:.3f} "
        f"precision={trigger_summary['test_weighted_precision']:.3f} "
        f"recall={trigger_summary['test_weighted_recall']:.3f}"
    )


def _trigger_config_from_artifact(
    artifact: Dict[str, Any],
    threshold_override: float | None,
    rearm_threshold_override: float | None,
    cooldown_sec: float,
) -> JawClickTriggerConfig:
    config = dict(artifact["trigger_config"])
    config.update(default_runtime_trigger_overrides())
    if threshold_override is not None:
        config["clench_probability_threshold"] = float(threshold_override)
    if rearm_threshold_override is not None:
        config["rearm_clench_probability_threshold"] = float(rearm_threshold_override)
    cooldown_ms = int(round(float(cooldown_sec) * 1000.0))
    config["cooldown_ms"] = cooldown_ms
    config["minimum_separation_ms"] = cooldown_ms
    return JawClickTriggerConfig(**config)


def _build_replay_steps(args: argparse.Namespace) -> Tuple[List[ReplayStep], JawClickTriggerConfig, float]:
    artifact = _load_artifact(args.artifact)
    model_bundle = artifact["model_bundle"]
    selected_channels = model_bundle["selected_channels"]
    feature_columns = model_bundle["feature_columns"]
    raw_df = load_openbci_csv(args.csv.resolve())
    fs_hz = float(estimate_sampling(raw_df)["fs_used_hz"])
    filtered = preprocess_session_signals(raw_df, "jaw", selected_channels, fs_hz).to_numpy(dtype=float)

    window_sec = float(artifact["window_config"]["window_sec"])
    overlap = float(artifact["window_config"]["overlap"])
    smoothing_sec = float(artifact["event_config"]["smoothing_sec"])
    window_samples = max(1, int(round(window_sec * fs_hz)))
    hop_samples = max(1, int(round(window_samples * (1.0 - overlap))))
    smoothing_samples = max(1, int(round(smoothing_sec * fs_hz)))
    warmup_sec = max(DEFAULT_WARMUP_SEC, window_sec, smoothing_sec)
    frame_rows: List[Tuple[float, Dict[str, float], str]] = []

    for start in range(0, len(filtered) - window_samples + 1, hop_samples):
        end = start + window_samples
        center = start + window_samples // 2
        timestamp_sec = float(center / fs_hz)
        if timestamp_sec < warmup_sec:
            continue
        feature_vector = _feature_vector_from_filtered_matrix(
            filtered_matrix=filtered[start:end, :],
            feature_columns=feature_columns,
            selected_channels=selected_channels,
            smoothing_samples=smoothing_samples,
            window_samples=window_samples,
        )
        scores, event_label = _predict_scores(model_bundle, feature_vector)
        scores["envelope_uv"] = float(feature_vector[0, feature_columns.index("aggregate_rms_smooth_mean")])
        frame_rows.append((timestamp_sec, scores, event_label))

    trigger_config = _trigger_config_from_artifact(
        artifact,
        threshold_override=args.threshold,
        rearm_threshold_override=args.rearm_threshold,
        cooldown_sec=float(args.cooldown_sec),
    )
    detector = JawClickTrigger(trigger_config)
    detector.reset()

    replay_steps: List[ReplayStep] = []
    for timestamp_sec, scores, event_label in frame_rows:
        decision = detector.step(timestamp_sec=timestamp_sec, scores=scores, event_label=event_label)
        replay_steps.append(
            ReplayStep(
                timestamp_sec=timestamp_sec,
                clench_probability=float(scores["clench_probability"]),
                smoothed_clench_probability=float(decision.smoothed_scores["clench_probability"]),
                onset_probability=float(decision.smoothed_scores["onset_probability"]),
                event_label=event_label,
                emitted_click=bool(decision.emitted_click),
                trigger_reason=str(decision.reason),
            )
        )

    step_interval_sec = float(hop_samples / fs_hz)
    return replay_steps, trigger_config, step_interval_sec


def _run_replay_console(steps: Sequence[ReplayStep], csv_name: str, verbose: bool) -> None:
    detections = 0
    for step in steps:
        if verbose:
            print(
                f"t={step.timestamp_sec:7.3f}s "
                f"p_clench={step.smoothed_clench_probability:.3f} "
                f"p_onset={step.onset_probability:.3f} "
                f"state={step.event_label}"
            )
        if step.emitted_click:
            detections += 1
            print(f"{step.timestamp_sec:7.3f}s CLENCH DETECTED")

    print(f"Replay complete: {detections} clench event(s) from {csv_name}")


def _run_replay_visualization(
    steps: Sequence[ReplayStep],
    csv_path: Path,
    trigger_config: JawClickTriggerConfig,
    step_interval_sec: float,
) -> None:
    try:
        import tkinter as tk
    except ModuleNotFoundError as exc:
        raise SystemExit("Tkinter is unavailable in this Python environment.") from exc

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise SystemExit(f"Unable to open Tk replay window: {exc}") from exc
    root.title(f"Clench Replay | {csv_path.name}")
    root.resizable(False, False)

    frame = tk.Frame(root, padx=16, pady=16)
    frame.pack(fill="both", expand=True)

    tk.Label(frame, text=csv_path.name, font=("Helvetica", 14, "bold")).pack(pady=(0, 10))
    probability_label = tk.Label(frame, text="P(clench): 0.000", font=("Helvetica", 18))
    probability_label.pack(pady=(0, 12))
    tk.Label(
        frame,
        text=(
            f"Threshold: {trigger_config.clench_probability_threshold:.3f} | "
            f"Cooldown: {trigger_config.cooldown_ms} ms | "
            f"Smoothing: {trigger_config.smoothing_windows}"
        ),
        font=("Helvetica", 10),
    ).pack(pady=(0, 12))

    canvas = tk.Canvas(frame, width=120, height=120, highlightthickness=0)
    canvas.pack()
    indicator = canvas.create_oval(10, 10, 110, 110, fill="gray70", outline="black", width=2)

    state_label = tk.Label(frame, text="Idle", font=("Helvetica", 12))
    state_label.pack(pady=(12, 4))
    time_label = tk.Label(frame, text="t = 0.000 s", font=("Helvetica", 10))
    time_label.pack()
    completion_label = tk.Label(frame, text="", font=("Helvetica", 10))
    completion_label.pack(pady=(8, 0))

    flash_until_sec = float("-inf")
    current_index = 0
    detections = 0

    def tick() -> None:
        nonlocal flash_until_sec, current_index, detections
        if current_index >= len(steps):
            completion_label.config(text=f"Replay complete: {detections} clench event(s)")
            return

        step = steps[current_index]
        if step.emitted_click:
            flash_until_sec = step.timestamp_sec + DEFAULT_REPLAY_FLASH_SEC
            detections += 1
            print(f"{step.timestamp_sec:7.3f}s CLENCH DETECTED")

        is_lit = step.timestamp_sec <= flash_until_sec
        canvas.itemconfigure(indicator, fill="green3" if is_lit else "gray70")
        probability_label.config(text=f"P(clench): {step.smoothed_clench_probability:.3f}")
        state_label.config(text="CLENCH DETECTED" if is_lit else "Idle")
        time_label.config(text=f"t = {step.timestamp_sec:.3f} s")
        current_index += 1

        if current_index < len(steps):
            next_time_sec = steps[current_index].timestamp_sec
            delay_ms = max(1, int(round((next_time_sec - step.timestamp_sec) * 1000.0)))
        else:
            delay_ms = max(1, int(round(step_interval_sec * 1000.0)))
        root.after(delay_ms, tick)

    tick()
    root.mainloop()
    print(f"Replay complete: {detections} clench event(s) from {csv_path.name}")


def run_replay(args: argparse.Namespace) -> None:
    replay_steps, trigger_config, step_interval_sec = _build_replay_steps(args)
    print(
        f"Replay threshold={trigger_config.clench_probability_threshold:.3f} "
        f"rearm={trigger_config.rearm_clench_probability_threshold:.3f} "
        f"cooldown_ms={trigger_config.cooldown_ms} "
        f"smoothing={trigger_config.smoothing_windows}"
    )
    if args.visualize:
        _run_replay_visualization(
            replay_steps,
            csv_path=args.csv.resolve(),
            trigger_config=trigger_config,
            step_interval_sec=step_interval_sec,
        )
    else:
        _run_replay_console(replay_steps, csv_name=args.csv.name, verbose=args.verbose)


def run_live_detector(args: argparse.Namespace) -> None:
    try:
        from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "brainflow is not installed. Install it with `pip install -r requirements.txt` and try again."
        ) from exc

    artifact = _load_artifact(args.artifact)
    model_bundle = artifact["model_bundle"]
    trigger_config = _trigger_config_from_artifact(
        artifact,
        threshold_override=args.threshold,
        rearm_threshold_override=args.rearm_threshold,
        cooldown_sec=float(args.cooldown_sec),
    )
    detector = JawClickTrigger(trigger_config)
    selected_channels = model_bundle["selected_channels"]
    feature_columns = model_bundle["feature_columns"]

    params = BrainFlowInputParams()
    if args.serial_port:
        params.serial_port = args.serial_port
    if args.board == "playback":
        if args.playback_file is None:
            raise SystemExit("--playback-file is required when --board playback is used.")
        params.file = str(args.playback_file.resolve())
        params.master_board = int(BoardIds.CYTON_BOARD.value)

    board_id = _board_id_from_name(args.board, BoardIds)
    board = BoardShim(board_id, params)
    started_at = time.monotonic()
    last_status_time = float("-inf")

    try:
        board.prepare_session()
        board.start_stream()
        resolved_board_id = int(board.get_board_id())
        sampling_rate = int(BoardShim.get_sampling_rate(resolved_board_id))
        eeg_rows = BoardShim.get_eeg_channels(resolved_board_id)
        board_rows = _selected_board_rows(selected_channels, eeg_rows)

        window_sec = float(artifact["window_config"]["window_sec"])
        smoothing_sec = float(artifact["event_config"]["smoothing_sec"])
        window_samples = max(1, int(round(window_sec * sampling_rate)))
        smoothing_samples = max(1, int(round(smoothing_sec * sampling_rate)))
        context_samples = max(window_samples + 2 * smoothing_samples, int(round(1.0 * sampling_rate)))

        print(
            f"Streaming from {args.board} at {sampling_rate} Hz using {', '.join(selected_channels)} "
            f"with a {window_sec:.3f}s feature window."
        )

        while True:
            if args.duration_sec is not None and time.monotonic() - started_at >= float(args.duration_sec):
                break

            data = board.get_current_board_data(context_samples)
            if data.shape[1] < context_samples:
                time.sleep(max(0.0, float(args.poll_interval_sec)))
                continue

            feature_vector = _extract_live_feature_vector(
                data=data,
                board_rows=board_rows,
                sampling_rate=sampling_rate,
                feature_columns=feature_columns,
                selected_channels=selected_channels,
                smoothing_samples=smoothing_samples,
                window_samples=window_samples,
            )
            scores, event_label = _predict_scores(model_bundle, feature_vector)
            scores["envelope_uv"] = float(feature_vector[0, feature_columns.index("aggregate_rms_smooth_mean")])

            now_sec = time.monotonic()
            if now_sec - started_at < max(DEFAULT_WARMUP_SEC, window_sec, smoothing_sec):
                time.sleep(max(0.0, float(args.poll_interval_sec)))
                continue
            decision = detector.step(timestamp_sec=now_sec, scores=scores, event_label=event_label)
            if args.verbose and now_sec - last_status_time >= 1.0:
                print(
                    f"p_clench={scores['clench_probability']:.3f} "
                    f"p_onset={scores['onset_probability']:.3f} "
                    f"state={event_label}"
                )
                last_status_time = now_sec

            if decision.emitted_click:
                print("CLENCH DETECTED")
                if args.on_detect_command:
                    _run_detection_command(args.on_detect_command)

            time.sleep(max(0.0, float(args.poll_interval_sec)))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            board.stop_stream()
        except Exception:
            pass
        try:
            board.release_session()
        except Exception:
            pass


def main() -> None:
    args = parse_args()
    if args.command == "train":
        summary = train_and_save_artifact(args)
        _print_training_summary(summary, args.artifact)
    elif args.command == "live":
        run_live_detector(args)
    else:
        run_replay(args)


if __name__ == "__main__":
    main()
