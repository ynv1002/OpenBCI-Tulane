from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import random
import sys
import time
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.jaw_trigger_rules import JawClickTrigger
    from analysis.lr_event_validation import review_yaniv_lr as lr_validation
    from analysis.realtime_clench_detector import (
        DEFAULT_RUNTIME_COOLDOWN_SEC,
        _feature_vector_from_filtered_matrix,
        _load_artifact,
        _predict_scores,
        _trigger_config_from_artifact,
    )
    from analysis.utils import (
        PROJECT_ROOT,
        ensure_output_dir,
        audit_session,
        collapse_marker_events,
        compute_channel_quality,
        estimate_sampling,
        extract_window_features,
        load_openbci_csv,
        pair_marker_events,
        preprocess_session_signals,
        write_json,
    )
    from bci_pipeline.realtime_decoder import conservative_decoder_config
else:
    from .jaw_trigger_rules import JawClickTrigger
    from .lr_event_validation import review_yaniv_lr as lr_validation
    from .realtime_clench_detector import (
        DEFAULT_RUNTIME_COOLDOWN_SEC,
        _feature_vector_from_filtered_matrix,
        _load_artifact,
        _predict_scores,
        _trigger_config_from_artifact,
    )
    from .utils import (
        PROJECT_ROOT,
        ensure_output_dir,
        audit_session,
        collapse_marker_events,
        compute_channel_quality,
        estimate_sampling,
        extract_window_features,
        load_openbci_csv,
        pair_marker_events,
        preprocess_session_signals,
        write_json,
    )
    from bci_pipeline.realtime_decoder import conservative_decoder_config


LEFT = "LEFT"
RIGHT = "RIGHT"
JAW = "JAW"
REST = "REST"
GROUND_TRUTH_UNKNOWN = "UNKNOWN"
REPLAY = "REPLAY"
LIVE = "LIVE"
OPENBCI_RUNS_ROOT = PROJECT_ROOT.parent / "OPENBCI_runs"
YANIV_LR_ROOT = OPENBCI_RUNS_ROOT / "Yaniv"
if (YANIV_LR_ROOT / "EEG_LR").exists():
    YANIV_LR_ROOT = YANIV_LR_ROOT / "EEG_LR"

DEFAULT_JAW_ARTIFACT = PROJECT_ROOT / "models" / "realtime_clench_model.pkl"
DEFAULT_DIRECTION_ARTIFACT = PROJECT_ROOT / "models" / "clean_left_right_window_model.pkl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "live_runs" / "all_runs" / "hybrid_bci_tester"
DEFAULT_REPLAY_FILES = [
    YANIV_LR_ROOT / "LR-2-27-26-(01).csv",
    YANIV_LR_ROOT / "LR-3-15-26-(04).csv",
]
DEFAULT_CALIBRATION_SEC = 45.0
DEFAULT_PROMPT_TIMEOUT_SEC = 5.0
DEFAULT_REPLAY_STEP_SEC = 0.10
DEFAULT_TIMER_INTERVAL_MS = 50
DEFAULT_FLASH_SEC = 0.30
DEFAULT_LR_EVENT_CONFIRMATION_SEC = 0.20
DEFAULT_SEQUENCE_REPETITIONS = 3


@dataclass(frozen=True)
class HybridTesterConfig:
    jaw_artifact_path: Path = DEFAULT_JAW_ARTIFACT
    direction_artifact_path: Path = DEFAULT_DIRECTION_ARTIFACT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    replay_files: tuple[Path, ...] = tuple(DEFAULT_REPLAY_FILES)
    calibration_sec: float = DEFAULT_CALIBRATION_SEC
    prompt_timeout_sec: float = DEFAULT_PROMPT_TIMEOUT_SEC
    replay_step_sec: float = DEFAULT_REPLAY_STEP_SEC
    timer_interval_ms: int = DEFAULT_TIMER_INTERVAL_MS
    flash_sec: float = DEFAULT_FLASH_SEC
    lr_event_confirmation_sec: float = DEFAULT_LR_EVENT_CONFIRMATION_SEC
    sequence_repetitions: int = DEFAULT_SEQUENCE_REPETITIONS
    prompt_seed: int = 42


@dataclass
class DirectionPrediction:
    left_probability: float
    right_probability: float
    best_label: str
    best_probability: float
    margin: float
    agreed: bool
    notes: str = ""


@dataclass
class RuntimeSnapshot:
    run_time_sec: float
    top_label: str
    status_text: str
    left_confidence: float
    right_confidence: float
    jaw_confidence: float
    ground_truth_state: str
    active_prompt_id: int | None
    target_class: str | None
    detections: list[dict[str, Any]] = field(default_factory=list)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_std(series: pd.Series) -> pd.Series:
    out = series.replace(0.0, 1.0).fillna(1.0)
    return out


def _ground_truth_display_label(raw_label: str) -> str:
    if raw_label in {LEFT, RIGHT}:
        return raw_label
    if raw_label in {"HOLD", "REPEATED"}:
        return JAW
    if raw_label in {"BASELINE", "REST"}:
        return REST
    return GROUND_TRUTH_UNKNOWN


def _family_for_csv(csv_path: Path) -> str:
    stem = csv_path.name.upper()
    if stem.startswith("HR-"):
        return "jaw"
    return "left_right"


def _selected_board_rows(selected_channels: Sequence[str], eeg_rows: Sequence[int]) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for channel_name in selected_channels:
        channel_index = int(channel_name.split("_", maxsplit=1)[-1]) - 1
        if channel_index < 0 or channel_index >= len(eeg_rows):
            raise RuntimeError(f"{channel_name} is not available on this board.")
        rows.append((channel_name, int(eeg_rows[channel_index])))
    return rows


def _smooth_values(values: np.ndarray, window_samples: int) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(window=window_samples, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def _create_run_dir(output_root: Path, mode: str, tag: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in tag)
    return ensure_output_dir(output_root / f"{timestamp}_{mode.lower()}_{safe_tag}")


def _build_prompt_sequence(sequence_type: str, repetitions: int, seed: int) -> list[str]:
    active_classes = [LEFT, RIGHT, JAW]
    if sequence_type == "fixed":
        sequence: list[str] = []
        for _ in range(repetitions):
            for label in active_classes:
                sequence.extend([label, REST])
        return sequence
    rng = random.Random(seed)
    active_sequence = active_classes * repetitions
    rng.shuffle(active_sequence)
    sequence = []
    for label in active_sequence:
        sequence.extend([label, REST])
    return sequence


class HybridRunLogger:
    def __init__(self, output_dir: Path, mode: str, metadata: Dict[str, Any]) -> None:
        self.output_dir = ensure_output_dir(output_dir)
        self.mode = mode
        self.metadata = metadata
        self.trace_rows: list[dict[str, Any]] = []
        self.detection_rows: list[dict[str, Any]] = []
        self.prompt_rows: list[dict[str, Any]] = []
        self.ground_truth_rows: list[dict[str, Any]] = []

    def log_trace(self, row: Dict[str, Any]) -> None:
        self.trace_rows.append(row)

    def log_detection(self, row: Dict[str, Any]) -> None:
        self.detection_rows.append(row)

    def log_prompt(self, row: Dict[str, Any]) -> None:
        self.prompt_rows.append(row)

    def log_ground_truth_segments(self, rows: Sequence[Dict[str, Any]]) -> None:
        self.ground_truth_rows.extend(dict(row) for row in rows)

    def save(self) -> dict[str, Path]:
        metadata_path = self.output_dir / "run_metadata.json"
        detections_path = self.output_dir / "detections.csv"
        trace_path = self.output_dir / "confidence_trace.csv"
        prompts_path = self.output_dir / "trial_log.csv"
        ground_truth_path = self.output_dir / "ground_truth_segments.csv"

        write_json(metadata_path, self.metadata)
        detection_columns = [
            "run_time_sec",
            "event_time_sec",
            "event_time_relative_sec",
            "predicted_class",
            "accepted",
            "source",
            "confidence",
            "left_confidence",
            "right_confidence",
            "jaw_confidence",
            "peak_value",
            "peak_prominence",
            "assigned_block_id",
            "assigned_side",
            "inside_marker_block",
            "ground_truth_state",
            "prompt_id",
            "target_class",
            "notes",
        ]
        trace_columns = [
            "mode",
            "run_time_sec",
            "wall_time_utc",
            "prompt_id",
            "target_class",
            "ground_truth_state",
            "left_confidence",
            "right_confidence",
            "jaw_confidence",
            "direction_prediction",
            "direction_margin",
            "direction_agreed",
        ]
        prompt_columns = [
            "prompt_id",
            "target_class",
            "start_time_sec",
            "end_time_sec",
            "start_time_utc",
            "end_time_utc",
            "success",
            "success_time_sec",
            "latency",
            "wrong_detections_count",
            "timeout_flag",
            "detections_json",
        ]
        ground_truth_columns = [
            "label",
            "raw_label",
            "start_time_sec",
            "end_time_sec",
            "start_sample",
            "end_sample",
            "segment_kind",
            "segment_source",
        ]

        pd.DataFrame(self.detection_rows, columns=detection_columns).to_csv(detections_path, index=False)
        pd.DataFrame(self.trace_rows, columns=trace_columns).to_csv(trace_path, index=False)
        pd.DataFrame(self.prompt_rows, columns=prompt_columns).to_csv(prompts_path, index=False)
        pd.DataFrame(self.ground_truth_rows, columns=ground_truth_columns).to_csv(ground_truth_path, index=False)
        return {
            "metadata_path": metadata_path,
            "detections_path": detections_path,
            "trace_path": trace_path,
            "prompts_path": prompts_path,
            "ground_truth_path": ground_truth_path,
        }


class DirectionRuntimeModel:
    def __init__(self, artifact_path: Path) -> None:
        with artifact_path.resolve().open("rb") as handle:
            artifact = pickle.load(handle)
        if not isinstance(artifact, dict) or "model" not in artifact:
            raise RuntimeError(f"Direction artifact at {artifact_path} is not supported.")

        self.artifact_path = artifact_path.resolve()
        self.model = artifact["model"]
        self.model_family = str(artifact.get("model_family", type(self.model).__name__))
        self.selected_channels = list(artifact["selected_channels"])
        self.feature_columns = list(artifact["feature_columns"])
        self.window_sec = float(artifact["window_sec"])
        self.overlap = float(artifact["overlap"])
        self.feature_mode = str(artifact["feature_mode"])
        self.with_asymmetry = bool(artifact["with_asymmetry"])
        self.labels = list(artifact.get("labels", [LEFT, RIGHT]))
        self.calibration_mean: pd.Series | None = None
        self.calibration_std: pd.Series | None = None
        self.decoder_cfg = conservative_decoder_config()

    def build_feature_series(self, filtered_window: np.ndarray, fs_hz: float) -> pd.Series:
        feature_map = extract_window_features(
            filtered_window,
            channel_columns=self.selected_channels,
            fs_hz=fs_hz,
            feature_mode=self.feature_mode,
            with_asymmetry=self.with_asymmetry,
        )
        return pd.Series({column: float(feature_map[column]) for column in self.feature_columns}, dtype=float)

    def fit_calibration(self, calibration_rows: Sequence[pd.Series]) -> None:
        if not calibration_rows:
            raise RuntimeError("No calibration feature rows were available for the direction model.")
        frame = pd.DataFrame(calibration_rows).reset_index(drop=True)
        self.calibration_mean = frame.mean(axis=0)
        self.calibration_std = _safe_std(frame.std(axis=0, ddof=0))

    def predict_from_filtered_window(self, filtered_window: np.ndarray, fs_hz: float) -> DirectionPrediction:
        if self.calibration_mean is None or self.calibration_std is None:
            return DirectionPrediction(0.0, 0.0, GROUND_TRUTH_UNKNOWN, 0.0, 0.0, False, "direction_not_calibrated")
        feature_row = self.build_feature_series(filtered_window, fs_hz)
        normalized = ((feature_row - self.calibration_mean) / self.calibration_std).to_frame().T
        probabilities = self.model.predict_proba(normalized.to_numpy(dtype=float))[0]
        class_order = list(self.model.classes_)
        probability_map = {
            str(label): float(probabilities[index]) for index, label in enumerate(class_order)
        }
        left_probability = float(probability_map.get(LEFT, 0.0))
        right_probability = float(probability_map.get(RIGHT, 0.0))
        best_label = LEFT if left_probability >= right_probability else RIGHT
        best_probability = max(left_probability, right_probability)
        margin = abs(left_probability - right_probability)
        agreed = (
            best_probability >= float(self.decoder_cfg.direction_min_confidence)
            and margin >= float(self.decoder_cfg.direction_margin)
        )
        notes = ""
        if best_probability < float(self.decoder_cfg.direction_min_confidence):
            notes = "direction_confidence_low"
        elif margin < float(self.decoder_cfg.direction_margin):
            notes = "direction_margin_low"
        return DirectionPrediction(
            left_probability=left_probability,
            right_probability=right_probability,
            best_label=best_label,
            best_probability=best_probability,
            margin=margin,
            agreed=agreed,
            notes=notes,
        )


class JawRuntimeModel:
    def __init__(self, artifact_path: Path, cooldown_sec: float = DEFAULT_RUNTIME_COOLDOWN_SEC) -> None:
        self.artifact_path = artifact_path.resolve()
        self.artifact = _load_artifact(self.artifact_path)
        self.model_bundle = self.artifact["model_bundle"]
        self.selected_channels = list(self.model_bundle["selected_channels"])
        self.feature_columns = list(self.model_bundle["feature_columns"])
        self.window_sec = float(self.artifact["window_config"]["window_sec"])
        self.smoothing_sec = float(self.artifact["event_config"]["smoothing_sec"])
        self.cooldown_sec = cooldown_sec
        self.reset()

    def reset(self) -> None:
        trigger_config = _trigger_config_from_artifact(
            self.artifact,
            threshold_override=None,
            rearm_threshold_override=None,
            cooldown_sec=self.cooldown_sec,
        )
        self.detector = JawClickTrigger(trigger_config)
        self.detector.reset()

    def step_from_filtered_tail(self, filtered_tail: np.ndarray, fs_hz: float, timestamp_sec: float) -> dict[str, Any]:
        window_samples = max(1, int(round(self.window_sec * fs_hz)))
        smoothing_samples = max(1, int(round(self.smoothing_sec * fs_hz)))
        feature_vector = _feature_vector_from_filtered_matrix(
            filtered_matrix=filtered_tail,
            feature_columns=self.feature_columns,
            selected_channels=self.selected_channels,
            smoothing_samples=smoothing_samples,
            window_samples=window_samples,
        )
        scores, event_label = _predict_scores(self.model_bundle, feature_vector)
        decision = self.detector.step(timestamp_sec=timestamp_sec, scores=scores, event_label=event_label)
        return {
            "jaw_probability": float(decision.smoothed_scores["clench_probability"]),
            "jaw_onset_probability": float(decision.smoothed_scores["onset_probability"]),
            "jaw_active_probability": float(decision.smoothed_scores["active_probability"]),
            "jaw_inactive_probability": float(decision.smoothed_scores["inactive_probability"]),
            "jaw_offset_probability": float(decision.smoothed_scores["offset_probability"]),
            "jaw_event_label": str(event_label),
            "emitted_click": bool(decision.emitted_click),
            "trigger_reason": str(decision.reason),
        }


class PromptLocalLREventDetector:
    def __init__(self, fs_hz: float, config: HybridTesterConfig) -> None:
        self.fs_hz = float(fs_hz)
        self.config = config
        self.confirmation_samples = max(1, int(round(config.lr_event_confirmation_sec * fs_hz)))
        self.emitted_peak_samples: set[int] = set()

    def reset(self) -> None:
        self.emitted_peak_samples.clear()

    def detect_new_events(self, prompt_raw_df: pd.DataFrame, count_channels: list[str]) -> list[dict[str, Any]]:
        if prompt_raw_df.empty:
            return []
        count_signal, _ = lr_validation.FROZEN_LR6._build_count_signal(prompt_raw_df, count_channels, self.fs_hz)
        event_rows, _ = lr_validation._detect_events_for_window(
            count_signal,
            start_sample=0,
            end_sample=len(count_signal) - 1,
            fs_hz=self.fs_hz,
            start_time_sec=0.0,
            relative_origin_time_sec=0.0,
        )
        new_rows: list[dict[str, Any]] = []
        for row in event_rows:
            peak_sample = int(row["peak_sample"])
            if not bool(row["kept_for_count"]):
                continue
            if peak_sample > len(count_signal) - 1 - self.confirmation_samples:
                continue
            if peak_sample in self.emitted_peak_samples:
                continue
            self.emitted_peak_samples.add(peak_sample)
            new_rows.append(row)
        return new_rows


class ReplaySessionBundle:
    def __init__(self, csv_path: Path, config: HybridTesterConfig) -> None:
        self.csv_path = csv_path.resolve()
        self.config = config
        self.family = _family_for_csv(self.csv_path)
        self.raw_df = load_openbci_csv(self.csv_path)
        self.sampling = estimate_sampling(self.raw_df)
        self.fs_hz = float(self.sampling["fs_used_hz"])
        self.audit = audit_session(self.csv_path, self.family)
        self.ground_truth_segments = self._build_ground_truth_segments()

        self.direction_model = DirectionRuntimeModel(config.direction_artifact_path)
        self.jaw_model = JawRuntimeModel(config.jaw_artifact_path)

        self.direction_filtered = preprocess_session_signals(
            self.raw_df,
            "left_right",
            self.direction_model.selected_channels,
            self.fs_hz,
        ).to_numpy(dtype=float)
        self.jaw_filtered = preprocess_session_signals(
            self.raw_df,
            "jaw",
            self.jaw_model.selected_channels,
            self.fs_hz,
        ).to_numpy(dtype=float)

        self.direction_model.fit_calibration(self._collect_direction_calibration_rows())
        self.step_rows = self._build_step_rows()
        self.lr_detection_rows = self._build_replay_lr_detection_rows()
        self.jaw_detection_rows = self._build_replay_jaw_detection_rows()
        self.detection_rows = sorted(
            self.lr_detection_rows + self.jaw_detection_rows,
            key=lambda row: (float(row["run_time_sec"]), int(not bool(row["accepted"]))),
        )

    def _build_ground_truth_segments(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for segment in self.audit["segments"]:
            rows.append(
                {
                    "label": _ground_truth_display_label(str(segment["label"])),
                    "raw_label": str(segment["label"]),
                    "start_time_sec": float(segment["start_time_sec"]),
                    "end_time_sec": float(segment["end_time_sec"]),
                    "start_sample": int(segment["start_sample"]),
                    "end_sample": int(segment["end_sample"]),
                    "segment_kind": str(segment["segment_kind"]),
                    "segment_source": str(segment["segment_source"]),
                }
            )
        return rows

    def state_at_time(self, time_sec: float) -> str:
        for row in self.ground_truth_segments:
            if float(row["start_time_sec"]) <= time_sec <= float(row["end_time_sec"]):
                return str(row["label"])
        return GROUND_TRUTH_UNKNOWN

    def _collect_direction_calibration_rows(self) -> list[pd.Series]:
        rows: list[pd.Series] = []
        window_samples = max(1, int(round(self.direction_model.window_sec * self.fs_hz)))
        step_samples = max(1, int(round(self.config.replay_step_sec * self.fs_hz)))
        max_sample = min(len(self.direction_filtered), int(round(self.config.calibration_sec * self.fs_hz)))
        for end_sample in range(window_samples, max_sample + 1, step_samples):
            window = self.direction_filtered[end_sample - window_samples : end_sample, :]
            rows.append(self.direction_model.build_feature_series(window, self.fs_hz))
        return rows

    def _build_step_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        jaw_min_samples = max(
            int(round(self.jaw_model.window_sec * self.fs_hz)),
            int(round(self.jaw_model.smoothing_sec * self.fs_hz)) * 2,
        )
        direction_window_samples = max(1, int(round(self.direction_model.window_sec * self.fs_hz)))
        step_samples = max(1, int(round(self.config.replay_step_sec * self.fs_hz)))
        self.jaw_model.reset()

        start_sample = max(jaw_min_samples, direction_window_samples)
        for end_sample in range(start_sample, len(self.raw_df) + 1, step_samples):
            timestamp_sec = float((end_sample - 1) / self.fs_hz)
            direction_window = self.direction_filtered[end_sample - direction_window_samples : end_sample, :]
            direction_prediction = self.direction_model.predict_from_filtered_window(direction_window, self.fs_hz)

            jaw_start = max(0, end_sample - jaw_min_samples)
            jaw_tail = self.jaw_filtered[jaw_start:end_sample, :]
            jaw_step = self.jaw_model.step_from_filtered_tail(jaw_tail, self.fs_hz, timestamp_sec)

            rows.append(
                {
                    "run_time_sec": timestamp_sec,
                    "sample_index": end_sample - 1,
                    "left_confidence": direction_prediction.left_probability,
                    "right_confidence": direction_prediction.right_probability,
                    "direction_probability": direction_prediction.best_probability,
                    "direction_margin": direction_prediction.margin,
                    "direction_prediction": direction_prediction.best_label,
                    "direction_agreed": bool(direction_prediction.agreed),
                    "jaw_confidence": jaw_step["jaw_probability"],
                    "jaw_event_label": jaw_step["jaw_event_label"],
                    "jaw_emitted_click": bool(jaw_step["emitted_click"]),
                    "jaw_trigger_reason": jaw_step["trigger_reason"],
                    "ground_truth_state": self.state_at_time(timestamp_sec),
                }
            )
        return rows

    def _direction_prediction_at_sample(self, sample_index: int) -> DirectionPrediction:
        window_samples = max(1, int(round(self.direction_model.window_sec * self.fs_hz)))
        end_sample = sample_index + 1
        if end_sample < window_samples:
            return DirectionPrediction(0.0, 0.0, GROUND_TRUTH_UNKNOWN, 0.0, 0.0, False, "insufficient_history")
        window = self.direction_filtered[end_sample - window_samples : end_sample, :]
        return self.direction_model.predict_from_filtered_window(window, self.fs_hz)

    def _build_replay_lr_detection_rows(self) -> list[dict[str, Any]]:
        if self.family != "left_right" or "Marker" not in self.raw_df.columns:
            return []

        marker_events = collapse_marker_events(self.raw_df["Marker"])
        blocks_df, _ = lr_validation._build_block_table(marker_events, self.fs_hz)
        count_signal, _ = lr_validation.FROZEN_LR6._build_count_signal(
            self.raw_df,
            self.direction_model.selected_channels,
            self.fs_hz,
        )
        rows: list[dict[str, Any]] = []

        def append_event_rows(event_rows: Sequence[dict[str, Any]], block: dict[str, Any] | None) -> None:
            for row in event_rows:
                prediction = self._direction_prediction_at_sample(int(row["peak_sample"]))
                accepted = bool(row["kept_for_count"]) and prediction.agreed
                assigned_side = str(block["side"]) if block is not None else ""
                block_id = int(block["block_id"]) if block is not None else ""
                rows.append(
                    {
                        "run_time_sec": float(row["event_time_sec"]),
                        "event_time_sec": float(row["event_time_sec"]),
                        "event_time_relative_sec": float(row["event_time_relative_sec"]),
                        "predicted_class": prediction.best_label if accepted else "",
                        "accepted": accepted,
                        "source": "lr_event",
                        "confidence": float(prediction.best_probability),
                        "left_confidence": float(prediction.left_probability),
                        "right_confidence": float(prediction.right_probability),
                        "jaw_confidence": np.nan,
                        "peak_value": float(row["peak_value"]),
                        "peak_prominence": float(row["peak_prominence"]),
                        "assigned_block_id": block_id,
                        "assigned_side": assigned_side,
                        "inside_marker_block": "yes" if block is not None else "no",
                        "ground_truth_state": self.state_at_time(float(row["event_time_sec"])),
                        "notes": prediction.notes if not accepted else "",
                    }
                )

        for block in blocks_df.to_dict(orient="records"):
            block_events, _ = lr_validation._detect_events_for_window(
                count_signal,
                start_sample=int(block["start_sample"]),
                end_sample=int(block["end_sample"]),
                fs_hz=self.fs_hz,
                start_time_sec=float(block["start_time_sec"]),
                relative_origin_time_sec=float(block["start_time_sec"]),
            )
            append_event_rows(block_events, block)

        full_events, _ = lr_validation._detect_events_for_window(
            count_signal,
            start_sample=0,
            end_sample=len(count_signal) - 1,
            fs_hz=self.fs_hz,
            start_time_sec=0.0,
            relative_origin_time_sec=0.0,
        )
        for row in full_events:
            inside = blocks_df[
                (blocks_df["start_sample"] <= int(row["peak_sample"]))
                & (blocks_df["end_sample"] >= int(row["peak_sample"]))
            ]
            if not inside.empty:
                continue
            append_event_rows([row], None)
        return rows

    def _build_replay_jaw_detection_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for step in self.step_rows:
            if not bool(step["jaw_emitted_click"]):
                continue
            rows.append(
                {
                    "run_time_sec": float(step["run_time_sec"]),
                    "event_time_sec": float(step["run_time_sec"]),
                    "event_time_relative_sec": np.nan,
                    "predicted_class": JAW,
                    "accepted": True,
                    "source": "jaw_trigger",
                    "confidence": float(step["jaw_confidence"]),
                    "left_confidence": float(step["left_confidence"]),
                    "right_confidence": float(step["right_confidence"]),
                    "jaw_confidence": float(step["jaw_confidence"]),
                    "peak_value": np.nan,
                    "peak_prominence": np.nan,
                    "assigned_block_id": "",
                    "assigned_side": "",
                    "inside_marker_block": "",
                    "ground_truth_state": str(step["ground_truth_state"]),
                    "notes": str(step["jaw_trigger_reason"]),
                }
            )
        return rows


class LiveSampleStore:
    def __init__(self, channels: Sequence[str], fs_hz: float) -> None:
        self.channels = list(channels)
        self.fs_hz = float(fs_hz)
        self.values: dict[str, list[float]] = {channel: [] for channel in self.channels}
        self.total_samples = 0

    def append_batch(self, batch_by_channel: Dict[str, Sequence[float]]) -> None:
        batch_len = 0
        for channel in self.channels:
            values = [float(value) for value in batch_by_channel.get(channel, [])]
            self.values[channel].extend(values)
            batch_len = max(batch_len, len(values))
        self.total_samples += batch_len

    def frame_slice(self, start_sample: int = 0, end_sample: int | None = None) -> pd.DataFrame:
        if end_sample is None:
            end_sample = self.total_samples
        data = {
            channel: np.asarray(self.values[channel][start_sample:end_sample], dtype=float)
            for channel in self.channels
        }
        return pd.DataFrame(data)

    def tail_frame(self, sample_count: int) -> pd.DataFrame:
        start_sample = max(0, self.total_samples - sample_count)
        return self.frame_slice(start_sample=start_sample, end_sample=self.total_samples)


class ReplayController:
    def __init__(
        self,
        config: HybridTesterConfig,
        replay_csv: Path,
        speed: float,
        status_callback: Callable[[RuntimeSnapshot], None] | None = None,
    ) -> None:
        self.config = config
        self.bundle = ReplaySessionBundle(replay_csv, config)
        self.status_callback = status_callback
        self.speed = float(speed)
        self.playing = False
        self.current_index = 0
        self.index_accumulator = 0.0
        self.light_until = {LEFT: float("-inf"), RIGHT: float("-inf"), JAW: float("-inf")}
        self.last_status = "waiting"
        self.pending_detection_index = 0
        self.logger = HybridRunLogger(
            output_dir=_create_run_dir(config.output_root, REPLAY, replay_csv.stem),
            mode=REPLAY,
            metadata={
                "mode": REPLAY,
                "created_at_utc": _utc_now(),
                "csv_path": str(replay_csv.resolve()),
                "jaw_artifact_path": str(config.jaw_artifact_path.resolve()),
                "direction_artifact_path": str(config.direction_artifact_path.resolve()),
                "fs_hz": self.bundle.fs_hz,
                "sampling_note": self.bundle.sampling,
            },
        )
        self.logger.log_ground_truth_segments(self.bundle.ground_truth_segments)
        for row in self.bundle.detection_rows:
            self.logger.log_detection(row)
        for row in self.bundle.step_rows:
            self.logger.log_trace(
                {
                    "mode": REPLAY,
                    "run_time_sec": float(row["run_time_sec"]),
                    "wall_time_utc": "",
                    "prompt_id": "",
                    "target_class": "",
                    "ground_truth_state": str(row["ground_truth_state"]),
                    "left_confidence": float(row["left_confidence"]),
                    "right_confidence": float(row["right_confidence"]),
                    "jaw_confidence": float(row["jaw_confidence"]),
                    "direction_prediction": str(row["direction_prediction"]),
                    "direction_margin": float(row["direction_margin"]),
                    "direction_agreed": bool(row["direction_agreed"]),
                }
            )

    def set_speed(self, speed: float) -> None:
        self.speed = float(speed)

    def set_playing(self, playing: bool) -> None:
        self.playing = bool(playing)

    def snapshot(self) -> RuntimeSnapshot:
        row = self.bundle.step_rows[self.current_index]
        run_time_sec = float(row["run_time_sec"])
        while self.pending_detection_index < len(self.bundle.detection_rows):
            detection = self.bundle.detection_rows[self.pending_detection_index]
            if float(detection["run_time_sec"]) > run_time_sec + 1e-9:
                break
            if bool(detection["accepted"]):
                detected_class = str(detection["predicted_class"])
                if detected_class in self.light_until:
                    self.light_until[detected_class] = float(detection["run_time_sec"]) + self.config.flash_sec
                self.last_status = f"detected {detected_class}"
            self.pending_detection_index += 1

        active_detections = [
            detection
            for detection in self.bundle.detection_rows
            if bool(detection["accepted"])
            and float(detection["run_time_sec"]) <= run_time_sec
            and float(detection["run_time_sec"]) + self.config.flash_sec >= run_time_sec
        ]
        return RuntimeSnapshot(
            run_time_sec=run_time_sec,
            top_label=str(row["ground_truth_state"]),
            status_text=self.last_status if active_detections else "waiting",
            left_confidence=float(row["left_confidence"]),
            right_confidence=float(row["right_confidence"]),
            jaw_confidence=float(row["jaw_confidence"]),
            ground_truth_state=str(row["ground_truth_state"]),
            active_prompt_id=None,
            target_class=None,
            detections=active_detections,
        )

    def tick(self) -> RuntimeSnapshot:
        if self.playing and self.current_index < len(self.bundle.step_rows) - 1:
            step_ratio = self.speed * (self.config.timer_interval_ms / 1000.0) / self.config.replay_step_sec
            self.index_accumulator += step_ratio
            advance = int(self.index_accumulator)
            if advance > 0:
                self.index_accumulator -= advance
                self.current_index = min(len(self.bundle.step_rows) - 1, self.current_index + advance)
        snapshot = self.snapshot()
        if self.status_callback is not None:
            self.status_callback(snapshot)
        return snapshot

    def run_headless(self) -> dict[str, Path]:
        self.current_index = len(self.bundle.step_rows) - 1
        self.pending_detection_index = len(self.bundle.detection_rows)
        return self.logger.save()


@dataclass
class LivePromptState:
    prompt_id: int
    target_class: str
    start_run_time_sec: float
    start_wall_time_utc: str
    start_sample: int
    detections_json: list[dict[str, Any]] = field(default_factory=list)
    wrong_detections_count: int = 0


class LiveController:
    def __init__(
        self,
        config: HybridTesterConfig,
        sequence_type: str,
        board: str,
        serial_port: str = "",
        playback_file: Path | None = None,
        max_live_sec: float | None = None,
        status_callback: Callable[[RuntimeSnapshot], None] | None = None,
    ) -> None:
        try:
            from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
        except ModuleNotFoundError as exc:
            raise RuntimeError("brainflow is required for live mode.") from exc

        self.BoardIds = BoardIds
        self.BoardShim = BoardShim
        self.BrainFlowInputParams = BrainFlowInputParams
        self.config = config
        self.sequence_type = sequence_type
        self.board_name = board
        self.serial_port = serial_port
        self.playback_file = playback_file
        self.max_live_sec = max_live_sec
        self.status_callback = status_callback

        self.direction_model = DirectionRuntimeModel(config.direction_artifact_path)
        self.jaw_model = JawRuntimeModel(config.jaw_artifact_path)
        self.union_channels = sorted(set(self.direction_model.selected_channels) | set(self.jaw_model.selected_channels))
        self.sample_store = LiveSampleStore(self.union_channels, fs_hz=250.0)
        self.sequence = _build_prompt_sequence(sequence_type, config.sequence_repetitions, config.prompt_seed)
        self.current_prompt_index = -1
        self.current_prompt: LivePromptState | None = None
        self.prompt_event_detector: PromptLocalLREventDetector | None = None
        self.calibration_rows: list[pd.Series] = []
        self.live_started = False
        self.run_complete = False
        self.last_status = "waiting"
        self.last_direction_prediction = DirectionPrediction(0.0, 0.0, GROUND_TRUTH_UNKNOWN, 0.0, 0.0, False)
        self.latest_jaw_confidence = 0.0
        self.current_tick_detections: list[dict[str, Any]] = []

        self.logger = HybridRunLogger(
            output_dir=_create_run_dir(config.output_root, LIVE, sequence_type),
            mode=LIVE,
            metadata={
                "mode": LIVE,
                "created_at_utc": _utc_now(),
                "sequence_type": sequence_type,
                "sequence": self.sequence,
                "jaw_artifact_path": str(config.jaw_artifact_path.resolve()),
                "direction_artifact_path": str(config.direction_artifact_path.resolve()),
                "board": board,
                "serial_port": serial_port,
                "playback_file": str(playback_file.resolve()) if playback_file else "",
                "calibration_sec": config.calibration_sec,
                "prompt_timeout_sec": config.prompt_timeout_sec,
            },
        )
        self._prepare_board()

    def _prepare_board(self) -> None:
        params = self.BrainFlowInputParams()
        if self.serial_port:
            params.serial_port = self.serial_port
        if self.board_name == "playback":
            if self.playback_file is None:
                raise RuntimeError("--playback-file is required for playback board mode.")
            params.file = str(self.playback_file.resolve())
            params.master_board = int(self.BoardIds.CYTON_BOARD.value)

        if self.board_name == "cyton":
            board_id = int(self.BoardIds.CYTON_BOARD.value)
        elif self.board_name == "synthetic":
            board_id = int(self.BoardIds.SYNTHETIC_BOARD.value)
        elif self.board_name == "playback":
            board_id = int(self.BoardIds.PLAYBACK_FILE_BOARD.value)
        else:
            raise RuntimeError(f"Unsupported live board '{self.board_name}'.")

        self.board = self.BoardShim(board_id, params)
        self.board.prepare_session()
        self.board.start_stream()
        resolved_board_id = int(self.board.get_board_id())
        self.fs_hz = float(self.BoardShim.get_sampling_rate(resolved_board_id))
        self.sample_store.fs_hz = self.fs_hz
        eeg_rows = self.BoardShim.get_eeg_channels(resolved_board_id)
        self.board_rows = _selected_board_rows(self.union_channels, eeg_rows)
        self.prompt_event_detector = PromptLocalLREventDetector(self.fs_hz, self.config)
        self.live_started = True

    def stop(self) -> dict[str, Path]:
        if self.live_started:
            try:
                self.board.stop_stream()
            except Exception:
                pass
            try:
                self.board.release_session()
            except Exception:
                pass
            self.live_started = False
        self.board = None
        return self.logger.save()

    def _append_new_samples(self) -> int:
        data = self.board.get_board_data()
        if data.shape[1] == 0:
            return 0
        batch_by_channel = {name: np.asarray(data[row_index], dtype=float) for name, row_index in self.board_rows}
        batch_len = len(next(iter(batch_by_channel.values()))) if batch_by_channel else 0
        self.sample_store.append_batch(batch_by_channel)
        return batch_len

    def _current_run_time_sec(self) -> float:
        if self.sample_store.total_samples <= 0:
            return 0.0
        return float((self.sample_store.total_samples - 1) / self.fs_hz)

    def _direction_prediction_from_tail(self) -> DirectionPrediction:
        window_samples = max(1, int(round(self.direction_model.window_sec * self.fs_hz)))
        if self.sample_store.total_samples < window_samples:
            return DirectionPrediction(0.0, 0.0, GROUND_TRUTH_UNKNOWN, 0.0, 0.0, False, "insufficient_history")
        raw_tail = self.sample_store.tail_frame(max(window_samples * 2, window_samples))
        filtered_tail = preprocess_session_signals(
            raw_tail,
            "left_right",
            self.direction_model.selected_channels,
            self.fs_hz,
        ).to_numpy(dtype=float)
        if len(filtered_tail) < window_samples:
            return DirectionPrediction(0.0, 0.0, GROUND_TRUTH_UNKNOWN, 0.0, 0.0, False, "insufficient_history")
        window = filtered_tail[-window_samples:, :]
        return self.direction_model.predict_from_filtered_window(window, self.fs_hz)

    def _jaw_step_from_tail(self, run_time_sec: float) -> dict[str, Any]:
        window_samples = max(1, int(round(self.jaw_model.window_sec * self.fs_hz)))
        context_samples = max(
            window_samples,
            int(round(self.jaw_model.smoothing_sec * self.fs_hz)) * 2,
        )
        if self.sample_store.total_samples < context_samples:
            return {
                "jaw_probability": 0.0,
                "jaw_onset_probability": 0.0,
                "jaw_event_label": GROUND_TRUTH_UNKNOWN,
                "emitted_click": False,
                "trigger_reason": "insufficient_history",
            }
        raw_tail = self.sample_store.tail_frame(context_samples)
        filtered_tail = preprocess_session_signals(
            raw_tail,
            "jaw",
            self.jaw_model.selected_channels,
            self.fs_hz,
        ).to_numpy(dtype=float)
        return self.jaw_model.step_from_filtered_tail(filtered_tail, self.fs_hz, run_time_sec)

    def _collect_calibration_row(self) -> None:
        window_samples = max(1, int(round(self.direction_model.window_sec * self.fs_hz)))
        if self.sample_store.total_samples < window_samples:
            return
        raw_tail = self.sample_store.tail_frame(max(window_samples * 2, window_samples))
        filtered_tail = preprocess_session_signals(
            raw_tail,
            "left_right",
            self.direction_model.selected_channels,
            self.fs_hz,
        ).to_numpy(dtype=float)
        if len(filtered_tail) < window_samples:
            return
        window = filtered_tail[-window_samples:, :]
        self.calibration_rows.append(self.direction_model.build_feature_series(window, self.fs_hz))

    def _start_next_prompt(self, run_time_sec: float) -> None:
        self.current_prompt_index += 1
        if self.current_prompt_index >= len(self.sequence):
            self.current_prompt = None
            self.run_complete = True
            self.last_status = "sequence complete"
            return
        target_class = self.sequence[self.current_prompt_index]
        self.current_prompt = LivePromptState(
            prompt_id=self.current_prompt_index + 1,
            target_class=target_class,
            start_run_time_sec=run_time_sec,
            start_wall_time_utc=_utc_now(),
            start_sample=self.sample_store.total_samples,
        )
        if self.prompt_event_detector is not None:
            self.prompt_event_detector.reset()
        self.last_status = "waiting"

    def _close_prompt(self, run_time_sec: float, success: bool, success_time_sec: float | None, timeout_flag: bool) -> None:
        if self.current_prompt is None:
            return
        target_class = self.current_prompt.target_class
        latency = None
        if success_time_sec is not None:
            latency = float(success_time_sec - self.current_prompt.start_run_time_sec)
        self.logger.log_prompt(
            {
                "prompt_id": int(self.current_prompt.prompt_id),
                "target_class": target_class,
                "start_time_sec": float(self.current_prompt.start_run_time_sec),
                "end_time_sec": float(run_time_sec),
                "start_time_utc": self.current_prompt.start_wall_time_utc,
                "end_time_utc": _utc_now(),
                "success": bool(success),
                "success_time_sec": float(success_time_sec) if success_time_sec is not None else np.nan,
                "latency": float(latency) if latency is not None else np.nan,
                "wrong_detections_count": int(self.current_prompt.wrong_detections_count),
                "timeout_flag": bool(timeout_flag),
                "detections_json": json.dumps(self.current_prompt.detections_json),
            }
        )
        self._start_next_prompt(run_time_sec)

    def _log_accepted_detection(
        self,
        run_time_sec: float,
        predicted_class: str,
        confidence: float,
        source: str,
        notes: str,
    ) -> None:
        prompt_id = self.current_prompt.prompt_id if self.current_prompt is not None else ""
        target_class = self.current_prompt.target_class if self.current_prompt is not None else ""
        row = {
            "run_time_sec": float(run_time_sec),
            "event_time_sec": float(run_time_sec),
            "event_time_relative_sec": (
                float(run_time_sec - self.current_prompt.start_run_time_sec) if self.current_prompt is not None else np.nan
            ),
            "predicted_class": predicted_class,
            "accepted": True,
            "source": source,
            "confidence": float(confidence),
            "left_confidence": float(self.last_direction_prediction.left_probability),
            "right_confidence": float(self.last_direction_prediction.right_probability),
            "jaw_confidence": float(self.latest_jaw_confidence),
            "peak_value": np.nan,
            "peak_prominence": np.nan,
            "assigned_block_id": "",
            "assigned_side": "",
            "inside_marker_block": "",
            "ground_truth_state": "",
            "prompt_id": prompt_id,
            "target_class": target_class,
            "notes": notes,
        }
        self.logger.log_detection(row)
        self.current_tick_detections.append(row)
        if self.current_prompt is not None:
            self.current_prompt.detections_json.append(
                {
                    "timestamp": float(run_time_sec),
                    "predicted_class": predicted_class,
                    "confidence": float(confidence),
                    "source": source,
                }
            )

    def _predict_direction_at_global_sample(self, sample_index: int) -> DirectionPrediction:
        window_samples = max(1, int(round(self.direction_model.window_sec * self.fs_hz)))
        if sample_index + 1 < window_samples:
            return DirectionPrediction(0.0, 0.0, GROUND_TRUTH_UNKNOWN, 0.0, 0.0, False, "insufficient_history")
        context_start = max(0, sample_index + 1 - max(window_samples * 2, window_samples))
        raw_frame = self.sample_store.frame_slice(context_start, sample_index + 1)
        filtered = preprocess_session_signals(
            raw_frame,
            "left_right",
            self.direction_model.selected_channels,
            self.fs_hz,
        ).to_numpy(dtype=float)
        if len(filtered) < window_samples:
            return DirectionPrediction(0.0, 0.0, GROUND_TRUTH_UNKNOWN, 0.0, 0.0, False, "insufficient_history")
        return self.direction_model.predict_from_filtered_window(filtered[-window_samples:, :], self.fs_hz)

    def _process_prompt_detections(self, run_time_sec: float, jaw_step: dict[str, Any]) -> None:
        if self.current_prompt is None or self.prompt_event_detector is None:
            return

        prompt_raw = self.sample_store.frame_slice(self.current_prompt.start_sample, self.sample_store.total_samples)
        lr_events = self.prompt_event_detector.detect_new_events(prompt_raw, self.direction_model.selected_channels)
        for event in lr_events:
            global_sample = self.current_prompt.start_sample + int(event["peak_sample"])
            prediction = self._predict_direction_at_global_sample(global_sample)
            if not prediction.agreed:
                self.logger.log_detection(
                    {
                        "run_time_sec": float(run_time_sec),
                        "event_time_sec": float(run_time_sec),
                        "event_time_relative_sec": float(event["event_time_relative_sec"]),
                        "predicted_class": "",
                        "accepted": False,
                        "source": "lr_event",
                        "confidence": float(prediction.best_probability),
                        "left_confidence": float(prediction.left_probability),
                        "right_confidence": float(prediction.right_probability),
                        "jaw_confidence": float(self.latest_jaw_confidence),
                        "peak_value": float(event["peak_value"]),
                        "peak_prominence": float(event["peak_prominence"]),
                        "assigned_block_id": "",
                        "assigned_side": "",
                        "inside_marker_block": "",
                        "ground_truth_state": "",
                        "prompt_id": int(self.current_prompt.prompt_id),
                        "target_class": self.current_prompt.target_class,
                        "notes": prediction.notes,
                    }
                )
                continue
            self.last_status = f"detected {prediction.best_label}"
            self._log_accepted_detection(
                run_time_sec=run_time_sec,
                predicted_class=prediction.best_label,
                confidence=prediction.best_probability,
                source="lr_event",
                notes="event_detector_and_direction_agreement",
            )
            if prediction.best_label == self.current_prompt.target_class:
                self._close_prompt(run_time_sec, success=True, success_time_sec=run_time_sec, timeout_flag=False)
                return
            self.current_prompt.wrong_detections_count += 1
            self.last_status = f"wrong: {prediction.best_label}"

        if bool(jaw_step["emitted_click"]):
            self.last_status = "detected JAW"
            self._log_accepted_detection(
                run_time_sec=run_time_sec,
                predicted_class=JAW,
                confidence=float(jaw_step["jaw_probability"]),
                source="jaw_trigger",
                notes=str(jaw_step["trigger_reason"]),
            )
            if self.current_prompt.target_class == JAW:
                self._close_prompt(run_time_sec, success=True, success_time_sec=run_time_sec, timeout_flag=False)
                return
            self.current_prompt.wrong_detections_count += 1
            self.last_status = "wrong: JAW"

    def _trace_row(self, run_time_sec: float) -> dict[str, Any]:
        return {
            "mode": LIVE,
            "run_time_sec": float(run_time_sec),
            "wall_time_utc": _utc_now(),
            "prompt_id": int(self.current_prompt.prompt_id) if self.current_prompt is not None else "",
            "target_class": self.current_prompt.target_class if self.current_prompt is not None else "",
            "ground_truth_state": "",
            "left_confidence": float(self.last_direction_prediction.left_probability),
            "right_confidence": float(self.last_direction_prediction.right_probability),
            "jaw_confidence": float(self.latest_jaw_confidence),
            "direction_prediction": str(self.last_direction_prediction.best_label),
            "direction_margin": float(self.last_direction_prediction.margin),
            "direction_agreed": bool(self.last_direction_prediction.agreed),
        }

    def tick(self) -> RuntimeSnapshot:
        self.current_tick_detections = []
        if self.run_complete:
            snapshot = RuntimeSnapshot(
                run_time_sec=self._current_run_time_sec(),
                top_label="COMPLETE",
                status_text=self.last_status,
                left_confidence=float(self.last_direction_prediction.left_probability),
                right_confidence=float(self.last_direction_prediction.right_probability),
                jaw_confidence=float(self.latest_jaw_confidence),
                ground_truth_state="",
                active_prompt_id=None,
                target_class=None,
                detections=self.current_tick_detections,
            )
            if self.status_callback is not None:
                self.status_callback(snapshot)
            return snapshot

        self._append_new_samples()
        run_time_sec = self._current_run_time_sec()
        if self.max_live_sec is not None and run_time_sec >= float(self.max_live_sec):
            self.run_complete = True
            self.last_status = "stopped"

        jaw_step = self._jaw_step_from_tail(run_time_sec)
        self.latest_jaw_confidence = float(jaw_step["jaw_probability"])

        if run_time_sec < self.config.calibration_sec:
            self._collect_calibration_row()
            self.last_direction_prediction = DirectionPrediction(0.0, 0.0, GROUND_TRUTH_UNKNOWN, 0.0, 0.0, False)
            self.last_status = "waiting"
            top_label = f"CALIBRATING {max(0.0, self.config.calibration_sec - run_time_sec):.1f}s"
        else:
            if self.direction_model.calibration_mean is None:
                if not self.calibration_rows:
                    self._collect_calibration_row()
                if not self.calibration_rows:
                    top_label = "CALIBRATING BUFFER"
                    self.last_status = "waiting"
                    trace_row = self._trace_row(run_time_sec)
                    self.logger.log_trace(trace_row)
                    snapshot = RuntimeSnapshot(
                        run_time_sec=run_time_sec,
                        top_label=top_label,
                        status_text=self.last_status,
                        left_confidence=0.0,
                        right_confidence=0.0,
                        jaw_confidence=float(self.latest_jaw_confidence),
                        ground_truth_state="",
                        active_prompt_id=None,
                        target_class=None,
                        detections=self.current_tick_detections,
                    )
                    if self.status_callback is not None:
                        self.status_callback(snapshot)
                    return snapshot
                self.direction_model.fit_calibration(self.calibration_rows)
                channel_quality_df = compute_channel_quality(self.sample_store.frame_slice(0, self.sample_store.total_samples))
                warnings = channel_quality_df[channel_quality_df["status"] != "safe"].to_dict(orient="records")
                self.logger.metadata["live_channel_quality"] = warnings
                self._start_next_prompt(run_time_sec)

            self.last_direction_prediction = self._direction_prediction_from_tail()
            if self.current_prompt is not None:
                self._process_prompt_detections(run_time_sec, jaw_step)
                if self.current_prompt is not None:
                    elapsed = run_time_sec - self.current_prompt.start_run_time_sec
                    if self.current_prompt.target_class == REST:
                        if self.current_prompt.wrong_detections_count > 0:
                            self._close_prompt(run_time_sec, success=False, success_time_sec=None, timeout_flag=False)
                        elif elapsed >= self.config.prompt_timeout_sec:
                            self._close_prompt(run_time_sec, success=True, success_time_sec=run_time_sec, timeout_flag=False)
                    elif elapsed >= self.config.prompt_timeout_sec:
                        self.last_status = "timeout"
                        self._close_prompt(run_time_sec, success=False, success_time_sec=None, timeout_flag=True)
            top_label = (
                f"DO {self.current_prompt.target_class}" if self.current_prompt is not None else "COMPLETE"
            )

        trace_row = self._trace_row(run_time_sec)
        self.logger.log_trace(trace_row)
        snapshot = RuntimeSnapshot(
            run_time_sec=run_time_sec,
            top_label=top_label,
            status_text=self.last_status,
            left_confidence=float(self.last_direction_prediction.left_probability),
            right_confidence=float(self.last_direction_prediction.right_probability),
            jaw_confidence=float(self.latest_jaw_confidence),
            ground_truth_state="",
            active_prompt_id=int(self.current_prompt.prompt_id) if self.current_prompt is not None else None,
            target_class=self.current_prompt.target_class if self.current_prompt is not None else None,
            detections=self.current_tick_detections,
        )
        if self.status_callback is not None:
            self.status_callback(snapshot)
        return snapshot


class HybridTesterApp:
    def __init__(
        self,
        config: HybridTesterConfig,
        initial_mode: str,
        replay_csv: Path | None,
        replay_speed: float,
        sequence_type: str,
        board: str,
        serial_port: str,
        playback_file: Path | None,
        max_live_sec: float | None,
    ) -> None:
        try:
            import tkinter as tk
            from tkinter import ttk
        except ModuleNotFoundError as exc:
            raise RuntimeError("Tkinter is required for the tester GUI.") from exc

        self.tk = tk
        self.ttk = ttk
        self.config = config
        self.replay_speed = replay_speed
        self.board = board
        self.serial_port = serial_port
        self.playback_file = playback_file
        self.max_live_sec = max_live_sec
        self.controller: ReplayController | LiveController | None = None
        self.last_saved_paths: dict[str, Path] | None = None
        self.flash_until = {LEFT: float("-inf"), RIGHT: float("-inf"), JAW: float("-inf")}

        self.root = tk.Tk()
        self.root.title("Hybrid BCI Tester")
        self.root.geometry("980x720")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.mode_var = tk.StringVar(value=initial_mode)
        self.replay_file_var = tk.StringVar(value=str((replay_csv or config.replay_files[0]).resolve()))
        self.sequence_var = tk.StringVar(value=sequence_type)
        self.speed_var = tk.StringVar(value=f"{replay_speed:g}x")
        self.status_var = tk.StringVar(value="waiting")
        self.top_label_var = tk.StringVar(value="READY")
        self.left_conf_var = tk.StringVar(value="0%")
        self.right_conf_var = tk.StringVar(value="0%")
        self.jaw_conf_var = tk.StringVar(value="0%")

        self._build_ui()
        self._set_mode_ui_state()
        self.root.after(config.timer_interval_ms, self._tick)

    def _build_ui(self) -> None:
        header = self.ttk.Frame(self.root, padding=12)
        header.pack(fill="x")
        self.ttk.Label(header, textvariable=self.top_label_var, font=("Helvetica", 24, "bold")).pack()

        controls = self.ttk.LabelFrame(self.root, text="Controls", padding=12)
        controls.pack(fill="x", padx=12, pady=(0, 12))

        self.ttk.Label(controls, text="Mode").grid(row=0, column=0, sticky="w")
        self.mode_combo = self.ttk.Combobox(controls, textvariable=self.mode_var, state="readonly", values=[LIVE, REPLAY], width=10)
        self.mode_combo.grid(row=0, column=1, sticky="w", padx=(6, 16))
        self.mode_combo.bind("<<ComboboxSelected>>", lambda _event: self._switch_mode())

        self.ttk.Label(controls, text="Replay file").grid(row=0, column=2, sticky="w")
        replay_values = [str(path.resolve()) for path in self.config.replay_files]
        self.replay_combo = self.ttk.Combobox(controls, textvariable=self.replay_file_var, state="readonly", values=replay_values, width=48)
        self.replay_combo.grid(row=0, column=3, sticky="we", padx=(6, 16))

        self.ttk.Label(controls, text="Replay speed").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.speed_combo = self.ttk.Combobox(
            controls,
            textvariable=self.speed_var,
            state="readonly",
            values=["0.5x", "1x", "2x", "4x"],
            width=10,
        )
        self.speed_combo.grid(row=1, column=1, sticky="w", padx=(6, 16), pady=(8, 0))
        self.speed_combo.bind("<<ComboboxSelected>>", lambda _event: self._apply_speed())

        self.ttk.Label(controls, text="Prompt sequence").grid(row=1, column=2, sticky="w", pady=(8, 0))
        self.sequence_combo = self.ttk.Combobox(
            controls,
            textvariable=self.sequence_var,
            state="readonly",
            values=["random_balanced", "fixed"],
            width=20,
        )
        self.sequence_combo.grid(row=1, column=3, sticky="w", padx=(6, 0), pady=(8, 0))

        self.start_button = self.ttk.Button(controls, text="Start", command=self._start_selected_mode)
        self.start_button.grid(row=0, column=4, padx=(16, 6))
        self.play_pause_button = self.ttk.Button(controls, text="Play", command=self._toggle_replay_play)
        self.play_pause_button.grid(row=0, column=5)
        self.stop_button = self.ttk.Button(controls, text="Stop", command=self._stop_controller)
        self.stop_button.grid(row=0, column=6, padx=(6, 0))
        controls.columnconfigure(3, weight=1)

        lights = self.ttk.LabelFrame(self.root, text="Detections", padding=12)
        lights.pack(fill="x", padx=12, pady=(0, 12))
        self.light_canvas = self.tk.Canvas(lights, width=920, height=180, highlightthickness=0)
        self.light_canvas.pack(fill="x")

        self.light_items = {}
        positions = [(150, LEFT, "#1d4ed8"), (460, JAW, "#16a34a"), (770, RIGHT, "#b91c1c")]
        for x_pos, label, color in positions:
            self.light_canvas.create_text(x_pos, 20, text=label, font=("Helvetica", 16, "bold"))
            oval = self.light_canvas.create_oval(x_pos - 60, 40, x_pos + 60, 160, fill="gray75", outline=color, width=3)
            self.light_items[label] = oval

        confidence_frame = self.ttk.Frame(self.root, padding=(12, 0))
        confidence_frame.pack(fill="x")
        for column, (label, var) in enumerate(
            [(LEFT, self.left_conf_var), (JAW, self.jaw_conf_var), (RIGHT, self.right_conf_var)]
        ):
            inner = self.ttk.LabelFrame(confidence_frame, text=f"{label} confidence", padding=12)
            inner.grid(row=0, column=column, sticky="nsew", padx=6)
            self.ttk.Label(inner, textvariable=var, font=("Helvetica", 18)).pack()
        confidence_frame.columnconfigure(0, weight=1)
        confidence_frame.columnconfigure(1, weight=1)
        confidence_frame.columnconfigure(2, weight=1)

        status = self.ttk.LabelFrame(self.root, text="Status", padding=12)
        status.pack(fill="both", expand=True, padx=12, pady=(12, 12))
        self.status_label = self.ttk.Label(status, textvariable=self.status_var, font=("Helvetica", 16))
        self.status_label.pack(anchor="w")
        self.info_text = self.tk.Text(status, height=10, wrap="word")
        self.info_text.pack(fill="both", expand=True, pady=(12, 0))
        self.info_text.insert("1.0", "Logs will be saved under live_runs/all_runs/hybrid_bci_tester/\n")
        self.info_text.configure(state="disabled")

    def _append_info(self, message: str) -> None:
        self.info_text.configure(state="normal")
        self.info_text.insert("end", message.rstrip() + "\n")
        self.info_text.see("end")
        self.info_text.configure(state="disabled")

    def _set_mode_ui_state(self) -> None:
        is_replay = self.mode_var.get() == REPLAY
        self.replay_combo.configure(state="readonly" if is_replay else "disabled")
        self.speed_combo.configure(state="readonly" if is_replay else "disabled")
        self.play_pause_button.configure(state="normal" if is_replay and isinstance(self.controller, ReplayController) else "disabled")
        self.sequence_combo.configure(state="readonly" if not is_replay else "disabled")

    def _switch_mode(self) -> None:
        self._stop_controller()
        self._set_mode_ui_state()

    def _apply_speed(self) -> None:
        if isinstance(self.controller, ReplayController):
            self.controller.set_speed(float(self.speed_var.get().rstrip("x")))

    def _start_selected_mode(self) -> None:
        self._stop_controller()
        mode = self.mode_var.get()
        if mode == REPLAY:
            replay_csv = Path(self.replay_file_var.get())
            self.controller = ReplayController(
                config=self.config,
                replay_csv=replay_csv,
                speed=float(self.speed_var.get().rstrip("x")),
                status_callback=self._apply_snapshot,
            )
            self.controller.set_playing(False)
            self.play_pause_button.configure(state="normal")
            self.play_pause_button.configure(text="Play")
            self._append_info(f"Loaded replay: {replay_csv.name}")
        else:
            self.controller = LiveController(
                config=self.config,
                sequence_type=self.sequence_var.get(),
                board=self.board,
                serial_port=self.serial_port,
                playback_file=self.playback_file,
                max_live_sec=self.max_live_sec,
                status_callback=self._apply_snapshot,
            )
            self.play_pause_button.configure(state="disabled")
            self._append_info("Live mode started.")
        self._set_mode_ui_state()

    def _toggle_replay_play(self) -> None:
        if not isinstance(self.controller, ReplayController):
            return
        self.controller.set_playing(not self.controller.playing)
        self.play_pause_button.configure(text="Pause" if self.controller.playing else "Play")

    def _stop_controller(self) -> None:
        if self.controller is None:
            return
        if isinstance(self.controller, LiveController):
            self.last_saved_paths = self.controller.stop()
            self._append_info(f"Saved live logs to {self.last_saved_paths['metadata_path'].parent}")
        elif isinstance(self.controller, ReplayController):
            self.last_saved_paths = self.controller.logger.save()
            self._append_info(f"Saved replay logs to {self.last_saved_paths['metadata_path'].parent}")
        self.controller = None
        self.play_pause_button.configure(text="Play", state="disabled")

    def _apply_snapshot(self, snapshot: RuntimeSnapshot) -> None:
        self.top_label_var.set(snapshot.top_label)
        self.status_var.set(snapshot.status_text)
        self.left_conf_var.set(f"{snapshot.left_confidence * 100:.0f}%")
        self.right_conf_var.set(f"{snapshot.right_confidence * 100:.0f}%")
        self.jaw_conf_var.set(f"{snapshot.jaw_confidence * 100:.0f}%")

        for label in (LEFT, RIGHT, JAW):
            lit = False
            for detection in snapshot.detections:
                if str(detection.get("predicted_class", "")) == label:
                    self.flash_until[label] = float(snapshot.run_time_sec) + self.config.flash_sec
            if snapshot.run_time_sec <= self.flash_until[label]:
                lit = True
            fill = {"LEFT": "#60a5fa", "RIGHT": "#f87171", "JAW": "#4ade80"}[label] if lit else "gray75"
            self.light_canvas.itemconfigure(self.light_items[label], fill=fill)

    def _tick(self) -> None:
        if self.controller is not None:
            snapshot = self.controller.tick()
            self._apply_snapshot(snapshot)
            if isinstance(self.controller, LiveController) and self.controller.run_complete:
                self._append_info("Live sequence complete.")
            elif isinstance(self.controller, ReplayController) and self.controller.current_index >= len(self.controller.bundle.step_rows) - 1:
                self.play_pause_button.configure(text="Play")
        self.root.after(self.config.timer_interval_ms, self._tick)

    def _on_close(self) -> None:
        self._stop_controller()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def run_headless_replay(config: HybridTesterConfig, replay_csv: Path, speed: float) -> dict[str, Path]:
    controller = ReplayController(config=config, replay_csv=replay_csv, speed=speed)
    return controller.run_headless()
