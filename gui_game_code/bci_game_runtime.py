from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, Sequence

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.jaw_trigger_rules import JawClickTrigger
    from analysis.hybrid_bci_tester import (
        DEFAULT_DIRECTION_ARTIFACT,
        DEFAULT_JAW_ARTIFACT,
        GROUND_TRUTH_UNKNOWN,
        JAW,
        LEFT,
        LIVE,
        REPLAY,
        RIGHT,
        DirectionPrediction,
        DirectionRuntimeModel,
        JawRuntimeModel,
        LiveSampleStore,
    )
    from analysis.lr_event_validation import review_yaniv_lr as lr_validation
    from analysis.lrj_dataset import reconstruct_lrj_trials
    from analysis.realtime_clench_detector import _trigger_config_from_artifact
    from analysis.stepwise_protocol_registry import FileContract, all_file_contracts
    from analysis.utils import (
        PROJECT_ROOT,
        audit_session,
        compute_channel_quality,
        ensure_output_dir,
        estimate_sampling,
        load_openbci_csv,
        preprocess_session_signals,
        write_json,
    )
else:
    from .jaw_trigger_rules import JawClickTrigger
    from .hybrid_bci_tester import (
        DEFAULT_DIRECTION_ARTIFACT,
        DEFAULT_JAW_ARTIFACT,
        GROUND_TRUTH_UNKNOWN,
        JAW,
        LEFT,
        LIVE,
        REPLAY,
        RIGHT,
        DirectionPrediction,
        DirectionRuntimeModel,
        JawRuntimeModel,
        LiveSampleStore,
    )
    from .lr_event_validation import review_yaniv_lr as lr_validation
    from .lrj_dataset import reconstruct_lrj_trials
    from .realtime_clench_detector import _trigger_config_from_artifact
    from .stepwise_protocol_registry import FileContract, all_file_contracts
    from .utils import (
        PROJECT_ROOT,
        audit_session,
        compute_channel_quality,
        ensure_output_dir,
        estimate_sampling,
        load_openbci_csv,
        preprocess_session_signals,
        write_json,
    )


ACTION_CLICK = "click"
ACTION_HOLD_START = "hold_start"
ACTION_HOLD_END = "hold_end"
ACTION_LEFT = "left"
ACTION_RIGHT = "right"
HAND_NEUTRAL = "NEUTRAL"
STATE_CLICK_HOLD = "CLICK + HOLD"
STATE_CLICK = "CLICK"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "live_runs" / "all_runs"
DEFAULT_TIMER_INTERVAL_MS = 50
DEFAULT_REPLAY_STEP_SEC = 0.10
DEFAULT_CALIBRATION_SEC = 45.0
DEFAULT_HAND_EVENT_CONFIRMATION_SEC = 0.20
DEFAULT_HAND_ACTION_LATCH_SEC = 0.45
DEFAULT_HAND_SWITCH_COOLDOWN_SEC = 0.20
DEFAULT_HAND_EVENT_LOOKBACK_SEC = 8.0
DEFAULT_JAW_HOLD_PROBABILITY_THRESHOLD = 0.70
DEFAULT_JAW_HOLD_ONSET_SEC = 0.45
DEFAULT_JAW_HOLD_RELEASE_SEC = 0.18
DEFAULT_JAW_SUPPRESS_CLICKS_DURING_HOLD = False
DEFAULT_STALE_STREAM_WARNING_SEC = 1.0
DEFAULT_BASELINE_SEC = 45.0
DEFAULT_GUIDED_PREP_SEC = 1.2
DEFAULT_GUIDED_REST_SEC = 1.3
DEFAULT_GUIDED_TAP_BASE_SEC = 1.2
DEFAULT_GUIDED_TAP_PER_COUNT_SEC = 0.80
DEFAULT_GUIDED_HOLD_SEC = 1.5
DEFAULT_GUIDED_REPETITIONS = 6
DEFAULT_GUIDED_HOLD_TRIALS = 6
DEFAULT_GAME_NOTE_CYCLES = 3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _create_run_dir(output_root: Path, mode: str, tag: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_output_dir(output_root / f"{timestamp}_{mode.lower()}_{_safe_tag(tag)}")


def _default_replay_files() -> tuple[Path, ...]:
    ordered_keys = [
        "yaniv_lrj",
        "ben_lrj",
        "yaniv_hr_2026_03_15",
        "yaniv_lr_2026_02_27",
        "yaniv_lr_2026_03_15",
    ]
    contracts = {contract.key: contract for contract in all_file_contracts()}
    return tuple(contracts[key].csv_path.resolve() for key in ordered_keys if key in contracts)


def _contract_for_csv(csv_path: Path) -> FileContract | None:
    resolved = csv_path.resolve()
    for contract in all_file_contracts():
        if contract.csv_path.resolve() == resolved:
            return contract
    return None


def _label_for_audit_segment(raw_label: str) -> str:
    if raw_label in {LEFT, RIGHT}:
        return raw_label
    if raw_label in {"HOLD", "REPEATED"}:
        return JAW
    if raw_label in {"BASELINE", "REST"}:
        return "REST"
    return raw_label or GROUND_TRUTH_UNKNOWN


def _selected_board_rows(selected_channels: Sequence[str], eeg_rows: Sequence[int]) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for channel_name in selected_channels:
        channel_index = int(channel_name.split("_", maxsplit=1)[-1]) - 1
        if channel_index < 0 or channel_index >= len(eeg_rows):
            raise RuntimeError(f"{channel_name} is not available on this board.")
        rows.append((channel_name, int(eeg_rows[channel_index])))
    return rows


def _validate_live_board_request(board: str, serial_port: str, playback_file: Path | None) -> None:
    if board == "cyton" and not serial_port.strip():
        raise RuntimeError(
            "Cyton diagnostics require a serial port. Select the OpenBCI serial port before running diagnostics."
        )
    if board == "playback" and playback_file is None:
        raise RuntimeError("--playback-file is required for playback board mode.")


def _board_error_message(board: str, exc: Exception) -> str:
    board_label = {"cyton": "Cyton", "synthetic": "Synthetic board", "playback": "Playback board"}.get(board, board)
    hint = ""
    if board == "cyton":
        hint = " Check that the correct OpenBCI serial port is selected and that no other process is using it."
    elif board == "playback":
        hint = " Check that the playback file exists and matches the expected Cyton master board format."
    return f"{board_label} probe failed: {exc}.{hint}"


def probe_live_board_connection(
    *,
    board: str,
    serial_port: str = "",
    playback_file: Path | None = None,
) -> dict[str, Any]:
    try:
        from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
    except ModuleNotFoundError as exc:
        raise RuntimeError("brainflow is required for live mode.") from exc

    _validate_live_board_request(board, serial_port, playback_file)

    params = BrainFlowInputParams()
    if serial_port:
        params.serial_port = serial_port
    if board == "playback":
        params.file = str(playback_file.resolve())
        params.master_board = int(BoardIds.CYTON_BOARD.value)

    if board == "cyton":
        board_id = int(BoardIds.CYTON_BOARD.value)
    elif board == "synthetic":
        board_id = int(BoardIds.SYNTHETIC_BOARD.value)
    elif board == "playback":
        board_id = int(BoardIds.PLAYBACK_FILE_BOARD.value)
    else:
        raise RuntimeError(f"Unsupported live board '{board}'.")

    probe_board = BoardShim(board_id, params)
    live_started = False
    try:
        probe_board.prepare_session()
        probe_board.start_stream()
        live_started = True
        resolved_board_id = int(probe_board.get_board_id())
        eeg_rows = BoardShim.get_eeg_channels(resolved_board_id)
        return {
            "board": board,
            "board_id": resolved_board_id,
            "sampling_rate_hz": int(BoardShim.get_sampling_rate(resolved_board_id)),
            "eeg_channel_count": len(eeg_rows),
            "serial_port": serial_port,
            "playback_file": str(playback_file.resolve()) if playback_file else "",
        }
    except Exception as exc:
        raise RuntimeError(_board_error_message(board, exc)) from exc
    finally:
        if live_started:
            try:
                probe_board.stop_stream()
            except Exception:
                pass
        try:
            probe_board.release_session()
        except Exception:
            pass


def _config_metadata(config: BCITrackingGameConfig) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value.resolve())
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        return value

    return {key: convert(value) for key, value in asdict(config).items()}


@dataclass(frozen=True)
class BCITrackingGameConfig:
    jaw_artifact_path: Path = DEFAULT_JAW_ARTIFACT
    direction_artifact_path: Path = DEFAULT_DIRECTION_ARTIFACT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    replay_files: tuple[Path, ...] = field(default_factory=_default_replay_files)
    calibration_sec: float = DEFAULT_CALIBRATION_SEC
    timer_interval_ms: int = DEFAULT_TIMER_INTERVAL_MS
    replay_step_sec: float = DEFAULT_REPLAY_STEP_SEC
    hand_event_confirmation_sec: float = DEFAULT_HAND_EVENT_CONFIRMATION_SEC
    hand_event_lookback_sec: float = DEFAULT_HAND_EVENT_LOOKBACK_SEC
    hand_action_latch_sec: float = DEFAULT_HAND_ACTION_LATCH_SEC
    hand_switch_cooldown_sec: float = DEFAULT_HAND_SWITCH_COOLDOWN_SEC
    jaw_hold_probability_threshold: float = DEFAULT_JAW_HOLD_PROBABILITY_THRESHOLD
    jaw_hold_onset_sec: float = DEFAULT_JAW_HOLD_ONSET_SEC
    jaw_hold_release_sec: float = DEFAULT_JAW_HOLD_RELEASE_SEC
    jaw_suppress_clicks_during_hold: bool = DEFAULT_JAW_SUPPRESS_CLICKS_DURING_HOLD
    stale_stream_warning_sec: float = DEFAULT_STALE_STREAM_WARNING_SEC
    baseline_sec: float = DEFAULT_BASELINE_SEC
    guided_prep_sec: float = DEFAULT_GUIDED_PREP_SEC
    guided_rest_sec: float = DEFAULT_GUIDED_REST_SEC
    guided_tap_base_sec: float = DEFAULT_GUIDED_TAP_BASE_SEC
    guided_tap_per_count_sec: float = DEFAULT_GUIDED_TAP_PER_COUNT_SEC
    guided_hold_sec: float = DEFAULT_GUIDED_HOLD_SEC
    guided_repetitions: int = DEFAULT_GUIDED_REPETITIONS
    guided_hold_trials: int = DEFAULT_GUIDED_HOLD_TRIALS
    game_note_cycles: int = DEFAULT_GAME_NOTE_CYCLES


@dataclass
class ControlEvent:
    timestamp_sec: float
    action: str
    source: str
    accepted: bool
    confidence: float
    jaw_confidence: float
    left_confidence: float
    right_confidence: float
    notes: str = ""

    def as_row(self, mode: str, source_name: str) -> dict[str, Any]:
        return {
            "mode": mode,
            "source_name": source_name,
            "timestamp_sec": float(self.timestamp_sec),
            "action": self.action,
            "source": self.source,
            "accepted": bool(self.accepted),
            "confidence": float(self.confidence),
            "jaw_confidence": float(self.jaw_confidence),
            "left_confidence": float(self.left_confidence),
            "right_confidence": float(self.right_confidence),
            "notes": self.notes,
        }


@dataclass
class TrackingSnapshot:
    mode: str
    source_name: str
    run_time_sec: float
    sample_count: int
    decoded_state: str
    status_text: str
    jaw_confidence: float
    jaw_onset_confidence: float
    jaw_active_confidence: float
    jaw_offset_confidence: float
    jaw_event_label: str
    jaw_hold_active: bool
    jaw_enabled: bool
    hand_left_confidence: float
    hand_right_confidence: float
    hand_prediction: str
    hand_margin: float
    hand_agreed: bool
    hand_notes: str
    hand_state: str
    hand_ready: bool
    hand_enabled: bool
    ground_truth_state: str
    stale_stream: bool
    calibration_remaining_sec: float
    detections: list[dict[str, Any]] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


class TrackingGameLogger:
    def __init__(self, output_dir: Path, metadata: Dict[str, Any]) -> None:
        self.output_dir = ensure_output_dir(output_dir)
        self.metadata = dict(metadata)
        self.control_trace_checkpoint_rows = 50
        self.game_trace_checkpoint_rows = 50
        self.control_trace_rows: list[dict[str, Any]] = []
        self.control_event_rows: list[dict[str, Any]] = []
        self.game_trace_rows: list[dict[str, Any]] = []
        self.game_event_rows: list[dict[str, Any]] = []
        self.ground_truth_rows: list[dict[str, Any]] = []
        self.marker_rows: list[dict[str, Any]] = []
        self.protocol_rows: list[dict[str, Any]] = []
        self.phase_rows: list[dict[str, Any]] = []
        self.participant_payload: dict[str, Any] = {}
        self.feedback_payload: dict[str, Any] = {}
        self.classifier_provenance_payload: dict[str, Any] = {}
        self.adaptation_payload: dict[str, Any] = {}
        self.signal_trace_frame: pd.DataFrame | None = None
        self.metadata_path = self.output_dir / "run_metadata.json"
        self.control_trace_path = self.output_dir / "control_trace.csv"
        self.control_events_path = self.output_dir / "control_events.csv"
        self.game_trace_path = self.output_dir / "game_trace.csv"
        self.game_events_path = self.output_dir / "game_events.csv"
        self.ground_truth_path = self.output_dir / "ground_truth_segments.csv"
        self.marker_log_path = self.output_dir / "marker_log.csv"
        self.protocol_truth_path = self.output_dir / "protocol_truth.csv"
        self.phase_log_path = self.output_dir / "phase_log.csv"
        self.participant_path = self.output_dir / "participant.json"
        self.feedback_path = self.output_dir / "session_feedback.json"
        self.classifier_provenance_path = self.output_dir / "classifier_provenance.json"
        self.adaptation_path = self.output_dir / "adaptation_summary.json"
        self.signal_trace_path = self.output_dir / "session_signal_trace.csv"
        self._initialize_output_files()

    def _write_rows(self, path: Path, rows: Sequence[dict[str, Any]]) -> None:
        pd.DataFrame(rows).to_csv(path, index=False)

    def _initialize_output_files(self) -> None:
        write_json(self.metadata_path, self.metadata)
        self._write_rows(self.control_trace_path, [])
        self._write_rows(self.control_events_path, [])
        self._write_rows(self.game_trace_path, [])
        self._write_rows(self.game_events_path, [])
        self._write_rows(self.ground_truth_path, [])
        self._write_rows(self.marker_log_path, [])
        self._write_rows(self.protocol_truth_path, [])
        self._write_rows(self.phase_log_path, [])
        write_json(self.participant_path, self.participant_payload)
        write_json(self.feedback_path, self.feedback_payload)
        write_json(self.classifier_provenance_path, self.classifier_provenance_payload)
        write_json(self.adaptation_path, self.adaptation_payload)

    def log_control_trace(self, row: dict[str, Any]) -> None:
        self.control_trace_rows.append(dict(row))
        if len(self.control_trace_rows) == 1 or len(self.control_trace_rows) % self.control_trace_checkpoint_rows == 0:
            self._write_rows(self.control_trace_path, self.control_trace_rows)

    def log_control_event(self, row: dict[str, Any]) -> None:
        self.control_event_rows.append(dict(row))
        self._write_rows(self.control_events_path, self.control_event_rows)

    def log_game_trace(self, row: dict[str, Any]) -> None:
        self.game_trace_rows.append(dict(row))
        if len(self.game_trace_rows) == 1 or len(self.game_trace_rows) % self.game_trace_checkpoint_rows == 0:
            self._write_rows(self.game_trace_path, self.game_trace_rows)

    def log_game_event(self, row: dict[str, Any]) -> None:
        self.game_event_rows.append(dict(row))
        self._write_rows(self.game_events_path, self.game_event_rows)

    def log_ground_truth_segments(self, rows: Sequence[dict[str, Any]]) -> None:
        self.ground_truth_rows.extend(dict(row) for row in rows)
        self._write_rows(self.ground_truth_path, self.ground_truth_rows)

    def log_marker(self, row: dict[str, Any]) -> None:
        self.marker_rows.append(dict(row))
        self._write_rows(self.marker_log_path, self.marker_rows)

    def log_protocol_row(self, row: dict[str, Any]) -> None:
        self.protocol_rows.append(dict(row))
        self._write_rows(self.protocol_truth_path, self.protocol_rows)

    def log_phase_row(self, row: dict[str, Any]) -> None:
        self.phase_rows.append(dict(row))
        self._write_rows(self.phase_log_path, self.phase_rows)

    def set_participant(self, payload: dict[str, Any]) -> None:
        self.participant_payload = dict(payload)
        write_json(self.participant_path, self.participant_payload)

    def set_feedback(self, payload: dict[str, Any]) -> None:
        self.feedback_payload = dict(payload)
        write_json(self.feedback_path, self.feedback_payload)

    def set_classifier_provenance(self, payload: dict[str, Any]) -> None:
        self.classifier_provenance_payload = dict(payload)
        write_json(self.classifier_provenance_path, self.classifier_provenance_payload)

    def set_adaptation_summary(self, payload: dict[str, Any]) -> None:
        self.adaptation_payload = dict(payload)
        write_json(self.adaptation_path, self.adaptation_payload)

    def set_signal_trace(self, frame: pd.DataFrame) -> None:
        self.signal_trace_frame = frame.copy()

    def save(self) -> dict[str, Path]:
        write_json(self.metadata_path, self.metadata)
        self._write_rows(self.control_trace_path, self.control_trace_rows)
        self._write_rows(self.control_events_path, self.control_event_rows)
        self._write_rows(self.game_trace_path, self.game_trace_rows)
        self._write_rows(self.game_events_path, self.game_event_rows)
        self._write_rows(self.ground_truth_path, self.ground_truth_rows)
        self._write_rows(self.marker_log_path, self.marker_rows)
        self._write_rows(self.protocol_truth_path, self.protocol_rows)
        self._write_rows(self.phase_log_path, self.phase_rows)
        write_json(self.participant_path, self.participant_payload)
        write_json(self.feedback_path, self.feedback_payload)
        write_json(self.classifier_provenance_path, self.classifier_provenance_payload)
        write_json(self.adaptation_path, self.adaptation_payload)
        if self.signal_trace_frame is not None:
            self.signal_trace_frame.to_csv(self.signal_trace_path, index=False)
        else:
            pd.DataFrame().to_csv(self.signal_trace_path, index=False)
        return {
            "metadata_path": self.metadata_path,
            "control_trace_path": self.control_trace_path,
            "control_events_path": self.control_events_path,
            "game_trace_path": self.game_trace_path,
            "game_events_path": self.game_events_path,
            "ground_truth_path": self.ground_truth_path,
            "marker_log_path": self.marker_log_path,
            "protocol_truth_path": self.protocol_truth_path,
            "phase_log_path": self.phase_log_path,
            "participant_path": self.participant_path,
            "feedback_path": self.feedback_path,
            "classifier_provenance_path": self.classifier_provenance_path,
            "adaptation_path": self.adaptation_path,
            "signal_trace_path": self.signal_trace_path,
        }


class GroundTruthResolver:
    def __init__(
        self,
        csv_path: Path | None,
        raw_df: pd.DataFrame | None = None,
        fs_hz: float | None = None,
        contract: FileContract | None = None,
    ) -> None:
        self.csv_path = csv_path.resolve() if csv_path is not None else None
        self.raw_df = raw_df
        self.fs_hz = float(fs_hz) if fs_hz is not None else None
        self.contract = contract or (_contract_for_csv(self.csv_path) if self.csv_path is not None else None)
        self.segments = self._build_segments()

    def _build_segments(self) -> list[dict[str, Any]]:
        if self.csv_path is None or self.raw_df is None or self.fs_hz is None:
            return []

        if self.contract is not None and self.contract.protocol_family == "LRJ":
            _, trials_df, _, _ = reconstruct_lrj_trials(
                raw_df=self.raw_df,
                fs_hz=self.fs_hz,
                dataset_display_name=self.contract.subject if self.contract is not None else self.csv_path.stem,
            )
            rows = []
            for trial in trials_df.to_dict(orient="records"):
                rows.append(
                    {
                        "label": str(trial["label"]),
                        "raw_label": str(trial["label"]),
                        "start_time_sec": float(trial["start_time_sec"]),
                        "end_time_sec": float(trial["end_time_sec"]),
                        "start_sample": int(trial["start_sample"]),
                        "end_sample": int(trial["end_sample"]),
                        "segment_kind": "interval",
                        "segment_source": "lrj_marker_pairs",
                    }
                )
            return rows

        family = "jaw" if self.contract is not None and self.contract.processing_family == "jaw" else "left_right"
        try:
            audit = audit_session(self.csv_path, family)
        except Exception:
            return []

        rows = []
        for segment in audit.get("segments", []):
            raw_label = str(segment["label"])
            rows.append(
                {
                    "label": _label_for_audit_segment(raw_label),
                    "raw_label": raw_label,
                    "start_time_sec": float(segment["start_time_sec"]),
                    "end_time_sec": float(segment["end_time_sec"]),
                    "start_sample": int(segment["start_sample"]),
                    "end_sample": int(segment["end_sample"]),
                    "segment_kind": str(segment.get("segment_kind", "")),
                    "segment_source": str(segment.get("segment_source", "")),
                }
            )
        return rows

    def state_at_time(self, time_sec: float) -> str:
        for segment in self.segments:
            if float(segment["start_time_sec"]) <= time_sec <= float(segment["end_time_sec"]):
                return str(segment["label"])
        return GROUND_TRUTH_UNKNOWN


class StreamingLREventDetector:
    def __init__(self, fs_hz: float, config: BCITrackingGameConfig) -> None:
        self.fs_hz = float(fs_hz)
        self.confirmation_samples = max(1, int(round(config.hand_event_confirmation_sec * fs_hz)))
        self.lookback_samples = max(1, int(round(config.hand_event_lookback_sec * fs_hz)))
        self._emitted_peak_samples: list[int] = []

    def reset(self) -> None:
        self._emitted_peak_samples.clear()

    def detect_new_events(self, sample_store: LiveSampleStore, count_channels: Sequence[str]) -> list[dict[str, Any]]:
        if sample_store.total_samples <= self.confirmation_samples + 2:
            return []
        tail_frame = sample_store.tail_frame(self.lookback_samples)
        if tail_frame.empty:
            return []

        tail_start_sample = max(0, sample_store.total_samples - len(tail_frame))
        count_signal, _ = lr_validation.FROZEN_LR6._build_count_signal(tail_frame, count_channels, self.fs_hz)
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
            if not bool(row["kept_for_count"]):
                continue
            global_peak_sample = tail_start_sample + int(row["peak_sample"])
            if global_peak_sample > sample_store.total_samples - 1 - self.confirmation_samples:
                continue
            if any(abs(global_peak_sample - previous) <= 2 for previous in self._emitted_peak_samples):
                continue
            self._emitted_peak_samples.append(global_peak_sample)
            new_rows.append(
                {
                    **row,
                    "global_peak_sample": global_peak_sample,
                    "event_time_sec": float(global_peak_sample / self.fs_hz),
                    "event_time_relative_sec": float(global_peak_sample / self.fs_hz),
                }
            )
        return new_rows


class JawHoldInterpreter:
    def __init__(self, config: BCITrackingGameConfig) -> None:
        self.hold_probability_threshold = float(config.jaw_hold_probability_threshold)
        self.hold_onset_sec = float(config.jaw_hold_onset_sec)
        self.hold_release_sec = float(config.jaw_hold_release_sec)
        self.suppress_clicks_during_hold = bool(config.jaw_suppress_clicks_during_hold)
        self.reset()

    def reset(self) -> None:
        self.hold_active = False
        self.engaged_since_sec: float | None = None
        self.release_candidate_sec: float | None = None

    def apply_runtime_tuning(
        self,
        *,
        hold_probability_threshold: float | None = None,
        hold_onset_sec: float | None = None,
        hold_release_sec: float | None = None,
        suppress_clicks_during_hold: bool | None = None,
    ) -> None:
        if hold_probability_threshold is not None:
            self.hold_probability_threshold = float(hold_probability_threshold)
        if hold_onset_sec is not None:
            self.hold_onset_sec = float(hold_onset_sec)
        if hold_release_sec is not None:
            self.hold_release_sec = float(hold_release_sec)
        if suppress_clicks_during_hold is not None:
            self.suppress_clicks_during_hold = bool(suppress_clicks_during_hold)

    def update(self, timestamp_sec: float, jaw_step: dict[str, Any]) -> tuple[bool, list[ControlEvent]]:
        jaw_confidence = float(jaw_step.get("jaw_probability", 0.0))
        jaw_active_confidence = float(jaw_step.get("jaw_active_probability", 0.0))
        jaw_onset_confidence = float(jaw_step.get("jaw_onset_probability", 0.0))
        event_label = str(jaw_step.get("jaw_event_label", GROUND_TRUTH_UNKNOWN))
        engaged = (
            jaw_active_confidence >= self.hold_probability_threshold
            or jaw_confidence >= self.hold_probability_threshold
            or event_label in {"ONSET", "ACTIVE"}
        )

        events: list[ControlEvent] = []
        click_allowed = not self.hold_active or not self.suppress_clicks_during_hold
        if bool(jaw_step.get("emitted_click", False)) and click_allowed:
            events.append(
                ControlEvent(
                    timestamp_sec=timestamp_sec,
                    action=ACTION_CLICK,
                    source="jaw_trigger",
                    accepted=True,
                    confidence=jaw_confidence,
                    jaw_confidence=jaw_confidence,
                    left_confidence=0.0,
                    right_confidence=0.0,
                    notes=str(jaw_step.get("trigger_reason", "")),
                )
            )

        if engaged:
            if self.engaged_since_sec is None:
                self.engaged_since_sec = timestamp_sec
            self.release_candidate_sec = None
            if not self.hold_active and timestamp_sec - self.engaged_since_sec >= self.hold_onset_sec:
                self.hold_active = True
                events.append(
                    ControlEvent(
                        timestamp_sec=timestamp_sec,
                        action=ACTION_HOLD_START,
                        source="jaw_hold",
                        accepted=True,
                        confidence=jaw_active_confidence,
                        jaw_confidence=jaw_confidence,
                        left_confidence=0.0,
                        right_confidence=0.0,
                        notes="sustained_active",
                    )
                )
        else:
            self.engaged_since_sec = None
            if self.hold_active:
                if self.release_candidate_sec is None:
                    self.release_candidate_sec = timestamp_sec
                elif timestamp_sec - self.release_candidate_sec >= self.hold_release_sec:
                    self.hold_active = False
                    self.release_candidate_sec = None
                    events.append(
                        ControlEvent(
                            timestamp_sec=timestamp_sec,
                            action=ACTION_HOLD_END,
                            source="jaw_hold",
                            accepted=True,
                            confidence=jaw_active_confidence,
                            jaw_confidence=jaw_confidence,
                            left_confidence=0.0,
                            right_confidence=0.0,
                            notes="released",
                        )
                    )
            else:
                self.release_candidate_sec = None
        return self.hold_active, events


class HandCommandInterpreter:
    def __init__(self, config: BCITrackingGameConfig) -> None:
        self.action_latch_sec = float(config.hand_action_latch_sec)
        self.switch_cooldown_sec = float(config.hand_switch_cooldown_sec)
        self.reset()

    def reset(self) -> None:
        self.current_state = HAND_NEUTRAL
        self.state_until_sec = float("-inf")
        self.last_switch_sec = float("-inf")

    def apply_runtime_tuning(
        self,
        *,
        action_latch_sec: float | None = None,
        switch_cooldown_sec: float | None = None,
    ) -> None:
        if action_latch_sec is not None:
            self.action_latch_sec = float(action_latch_sec)
        if switch_cooldown_sec is not None:
            self.switch_cooldown_sec = float(switch_cooldown_sec)

    def register_detection(self, timestamp_sec: float, label: str) -> bool:
        if label not in {LEFT, RIGHT}:
            return False
        if (
            self.current_state in {LEFT, RIGHT}
            and self.current_state != label
            and timestamp_sec - self.last_switch_sec < self.switch_cooldown_sec
        ):
            return False
        self.current_state = label
        self.state_until_sec = timestamp_sec + self.action_latch_sec
        self.last_switch_sec = timestamp_sec
        return True

    def state_at(self, timestamp_sec: float) -> str:
        if timestamp_sec > self.state_until_sec:
            self.current_state = HAND_NEUTRAL
        return self.current_state


class BCITrackingDecoder:
    def __init__(
        self,
        config: BCITrackingGameConfig,
        fs_hz: float,
        *,
        mode: str,
        source_name: str,
        csv_path: Path | None = None,
        contract: FileContract | None = None,
        hand_enabled: bool = True,
        jaw_enabled: bool = True,
    ) -> None:
        self.config = config
        self.fs_hz = float(fs_hz)
        self.mode = mode
        self.source_name = source_name
        self.csv_path = csv_path.resolve() if csv_path is not None else None
        self.contract = contract or (_contract_for_csv(self.csv_path) if self.csv_path is not None else None)
        self.hand_enabled = bool(hand_enabled)
        self.jaw_enabled = bool(jaw_enabled)

        self.direction_model = DirectionRuntimeModel(config.direction_artifact_path)
        self.jaw_model = JawRuntimeModel(config.jaw_artifact_path)
        self.union_channels = sorted(set(self.direction_model.selected_channels) | set(self.jaw_model.selected_channels))
        self.sample_store = LiveSampleStore(self.union_channels, fs_hz=self.fs_hz)
        self.ground_truth = GroundTruthResolver(self.csv_path, raw_df=None, fs_hz=None, contract=self.contract)

        self.calibration_rows: list[pd.Series] = []
        self.hand_ready = False
        self.last_direction_prediction = DirectionPrediction(0.0, 0.0, HAND_NEUTRAL, 0.0, 0.0, False)
        self.latest_jaw_step = {
            "jaw_probability": 0.0,
            "jaw_onset_probability": 0.0,
            "jaw_active_probability": 0.0,
            "jaw_offset_probability": 0.0,
            "jaw_event_label": GROUND_TRUTH_UNKNOWN,
            "emitted_click": False,
            "trigger_reason": "not_started",
        }
        self.lr_detector = StreamingLREventDetector(self.fs_hz, config) if self.hand_enabled else None
        self.hand_interpreter = HandCommandInterpreter(config)
        self.jaw_interpreter = JawHoldInterpreter(config)
        self._calibration_announced = False

    def export_signal_frame(self) -> pd.DataFrame:
        frame = self.sample_store.frame_slice(0, self.sample_store.total_samples)
        if frame.empty:
            return frame
        frame = frame.reset_index(drop=True)
        frame.insert(0, "sample_index", np.arange(len(frame), dtype=int))
        frame.insert(1, "run_time_sec", frame["sample_index"].to_numpy(dtype=float) / self.fs_hz)
        return frame

    def classifier_provenance(self) -> dict[str, Any]:
        jaw_trigger_cfg = self.jaw_model.detector.config.as_dict()
        return {
            "mode": self.mode,
            "source_name": self.source_name,
            "hand_branch": {
                "artifact_path": str(self.direction_model.artifact_path),
                "model_family": self.direction_model.model_family,
                "selected_channels": list(self.direction_model.selected_channels),
                "feature_mode": self.direction_model.feature_mode,
                "window_sec": float(self.direction_model.window_sec),
                "decision_style": "event_gated_direction_vote",
                "runtime_thresholds": {
                    "direction_min_confidence": float(self.direction_model.decoder_cfg.direction_min_confidence),
                    "direction_margin": float(self.direction_model.decoder_cfg.direction_margin),
                    "hand_action_latch_sec": float(self.hand_interpreter.action_latch_sec),
                    "hand_switch_cooldown_sec": float(self.hand_interpreter.switch_cooldown_sec),
                },
                "calibration_ready": bool(self.hand_ready),
                "calibration_rows": int(len(self.calibration_rows)),
            },
            "jaw_branch": {
                "artifact_path": str(self.jaw_model.artifact_path),
                "model_family": {
                    "jaw_4state_model": type(self.jaw_model.model_bundle["jaw_4state_model"]).__name__,
                    "binary_model": type(self.jaw_model.model_bundle["binary_model"]).__name__,
                },
                "selected_channels": list(self.jaw_model.selected_channels),
                "window_sec": float(self.jaw_model.window_sec),
                "smoothing_sec": float(self.jaw_model.smoothing_sec),
                "decision_style": "single_jaw_artifact_with_click_hold_outputs",
                "trigger_config": jaw_trigger_cfg,
                "hold_runtime": {
                    "hold_probability_threshold": float(self.jaw_interpreter.hold_probability_threshold),
                    "hold_onset_sec": float(self.jaw_interpreter.hold_onset_sec),
                    "hold_release_sec": float(self.jaw_interpreter.hold_release_sec),
                    "suppress_clicks_during_hold": bool(self.jaw_interpreter.suppress_clicks_during_hold),
                },
            },
        }

    def apply_session_adaptation(self, adaptation_summary: dict[str, Any]) -> dict[str, Any]:
        hand_settings = dict(adaptation_summary.get("hand", {}).get("settings", {}))
        jaw_settings = dict(adaptation_summary.get("jaw", {}).get("settings", {}))

        if hand_settings:
            updates = {}
            if "direction_min_confidence" in hand_settings:
                updates["direction_min_confidence"] = float(hand_settings["direction_min_confidence"])
            if "direction_margin" in hand_settings:
                updates["direction_margin"] = float(hand_settings["direction_margin"])
            if updates:
                from dataclasses import replace
                self.direction_model.decoder_cfg = replace(self.direction_model.decoder_cfg, **updates)
            self.hand_interpreter.apply_runtime_tuning(
                action_latch_sec=hand_settings.get("hand_action_latch_sec"),
                switch_cooldown_sec=hand_settings.get("hand_switch_cooldown_sec"),
            )

        if jaw_settings:
            trigger_cfg = _trigger_config_from_artifact(
                self.jaw_model.artifact,
                threshold_override=jaw_settings.get("click_threshold"),
                rearm_threshold_override=jaw_settings.get("rearm_threshold"),
                cooldown_sec=self.jaw_model.cooldown_sec,
            )
            trigger_cfg = trigger_cfg.__class__(
                strategy_name=trigger_cfg.strategy_name,
                clench_probability_threshold=float(jaw_settings.get("click_threshold", trigger_cfg.clench_probability_threshold)),
                onset_probability_threshold=float(jaw_settings.get("onset_probability_threshold", trigger_cfg.onset_probability_threshold)),
                active_probability_threshold=float(jaw_settings.get("active_probability_threshold", trigger_cfg.active_probability_threshold)),
                rearm_clench_probability_threshold=float(jaw_settings.get("rearm_threshold", trigger_cfg.rearm_clench_probability_threshold)),
                cooldown_ms=int(trigger_cfg.cooldown_ms),
                minimum_separation_ms=int(trigger_cfg.minimum_separation_ms),
                smoothing_windows=int(trigger_cfg.smoothing_windows),
                hold_suppression=bool(trigger_cfg.hold_suppression),
                require_transition_from_inactive=bool(trigger_cfg.require_transition_from_inactive),
                minimum_clench_rise=float(trigger_cfg.minimum_clench_rise),
                minimum_envelope_uv=float(trigger_cfg.minimum_envelope_uv),
            )
            self.jaw_model.detector = JawClickTrigger(trigger_cfg)
            self.jaw_model.detector.reset()
            self.jaw_interpreter.apply_runtime_tuning(
                hold_probability_threshold=jaw_settings.get("hold_probability_threshold"),
                hold_onset_sec=jaw_settings.get("hold_onset_sec"),
                hold_release_sec=jaw_settings.get("hold_release_sec"),
                suppress_clicks_during_hold=jaw_settings.get("suppress_clicks_during_hold"),
            )

        return self.classifier_provenance()

    def attach_ground_truth(self, raw_df: pd.DataFrame) -> None:
        if self.csv_path is None:
            return
        self.ground_truth = GroundTruthResolver(
            self.csv_path,
            raw_df=raw_df,
            fs_hz=self.fs_hz,
            contract=self.contract,
        )

    def append_batch(self, batch_by_channel: Dict[str, Sequence[float]]) -> int:
        previous_total = self.sample_store.total_samples
        self.sample_store.append_batch(batch_by_channel)
        return self.sample_store.total_samples - previous_total

    def _current_run_time_sec(self) -> float:
        if self.sample_store.total_samples <= 0:
            return 0.0
        return float((self.sample_store.total_samples - 1) / self.fs_hz)

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

    def _direction_prediction_from_tail(self) -> DirectionPrediction:
        window_samples = max(1, int(round(self.direction_model.window_sec * self.fs_hz)))
        if self.sample_store.total_samples < window_samples:
            return DirectionPrediction(0.0, 0.0, HAND_NEUTRAL, 0.0, 0.0, False, "insufficient_history")
        raw_tail = self.sample_store.tail_frame(max(window_samples * 2, window_samples))
        filtered_tail = preprocess_session_signals(
            raw_tail,
            "left_right",
            self.direction_model.selected_channels,
            self.fs_hz,
        ).to_numpy(dtype=float)
        if len(filtered_tail) < window_samples:
            return DirectionPrediction(0.0, 0.0, HAND_NEUTRAL, 0.0, 0.0, False, "insufficient_history")
        window = filtered_tail[-window_samples:, :]
        return self.direction_model.predict_from_filtered_window(window, self.fs_hz)

    def _predict_direction_at_global_sample(self, sample_index: int) -> DirectionPrediction:
        window_samples = max(1, int(round(self.direction_model.window_sec * self.fs_hz)))
        if sample_index + 1 < window_samples:
            return DirectionPrediction(0.0, 0.0, HAND_NEUTRAL, 0.0, 0.0, False, "insufficient_history")
        context_start = max(0, sample_index + 1 - max(window_samples * 2, window_samples))
        raw_frame = self.sample_store.frame_slice(context_start, sample_index + 1)
        filtered = preprocess_session_signals(
            raw_frame,
            "left_right",
            self.direction_model.selected_channels,
            self.fs_hz,
        ).to_numpy(dtype=float)
        if len(filtered) < window_samples:
            return DirectionPrediction(0.0, 0.0, HAND_NEUTRAL, 0.0, 0.0, False, "insufficient_history")
        window = filtered[-window_samples:, :]
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
                "jaw_active_probability": 0.0,
                "jaw_offset_probability": 0.0,
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

    def _decoded_state(self, hand_state: str, hold_active: bool, click_events: list[ControlEvent]) -> str:
        parts: list[str] = []
        if hold_active:
            parts.append(STATE_CLICK_HOLD)
        elif click_events:
            parts.append(STATE_CLICK)
        if hand_state in {LEFT, RIGHT}:
            parts.append(hand_state)
        return " | ".join(parts) if parts else HAND_NEUTRAL

    def tick(self, stale_stream: bool = False) -> TrackingSnapshot:
        run_time_sec = self._current_run_time_sec()
        detections: list[ControlEvent] = []
        messages: list[str] = []

        if self.hand_enabled and not self.hand_ready:
            if run_time_sec < self.config.calibration_sec:
                self._collect_calibration_row()
            elif self.direction_model.calibration_mean is None:
                if not self.calibration_rows:
                    self._collect_calibration_row()
                if self.calibration_rows:
                    self.direction_model.fit_calibration(self.calibration_rows)
                    self.hand_ready = True
                    if not self._calibration_announced:
                        messages.append("Hand branch calibrated.")
                        self._calibration_announced = True
        elif not self.hand_enabled:
            self.hand_ready = False

        if self.hand_enabled and self.hand_ready:
            self.last_direction_prediction = self._direction_prediction_from_tail()
            if self.lr_detector is not None:
                lr_events = self.lr_detector.detect_new_events(self.sample_store, self.direction_model.selected_channels)
                for event in lr_events:
                    prediction = self._predict_direction_at_global_sample(int(event["global_peak_sample"]))
                    event_label = prediction.best_label if prediction.best_label in {LEFT, RIGHT} else HAND_NEUTRAL
                    accepted = bool(prediction.agreed and event_label in {LEFT, RIGHT})
                    detection = ControlEvent(
                        timestamp_sec=float(event["event_time_sec"]),
                        action=event_label,
                        source="hand_event",
                        accepted=accepted,
                        confidence=float(prediction.best_probability),
                        jaw_confidence=float(self.latest_jaw_step.get("jaw_probability", 0.0)),
                        left_confidence=float(prediction.left_probability),
                        right_confidence=float(prediction.right_probability),
                        notes=prediction.notes,
                    )
                    detections.append(detection)
                    if accepted and self.hand_interpreter.register_detection(float(event["event_time_sec"]), event_label):
                        messages.append(f"Hand event: {event_label}")
        else:
            self.last_direction_prediction = DirectionPrediction(0.0, 0.0, HAND_NEUTRAL, 0.0, 0.0, False)

        if self.jaw_enabled:
            self.latest_jaw_step = self._jaw_step_from_tail(run_time_sec)
            hold_active, jaw_events = self.jaw_interpreter.update(run_time_sec, self.latest_jaw_step)
            detections.extend(jaw_events)
            for event in jaw_events:
                if event.action == ACTION_CLICK:
                    messages.append("Jaw click")
                elif event.action == ACTION_HOLD_START:
                    messages.append("Jaw hold started")
                elif event.action == ACTION_HOLD_END:
                    messages.append("Jaw hold released")
        else:
            hold_active = False
            self.latest_jaw_step = {
                "jaw_probability": 0.0,
                "jaw_onset_probability": 0.0,
                "jaw_active_probability": 0.0,
                "jaw_offset_probability": 0.0,
                "jaw_event_label": "DISABLED",
                "emitted_click": False,
                "trigger_reason": "disabled_for_replay_contract",
            }

        hand_state = self.hand_interpreter.state_at(run_time_sec) if self.hand_enabled else HAND_NEUTRAL
        ground_truth_state = self.ground_truth.state_at_time(run_time_sec)
        click_events = [event for event in detections if event.accepted and event.action == ACTION_CLICK]
        decoded_state = self._decoded_state(hand_state, hold_active, click_events)

        if stale_stream:
            status_text = "stale stream"
        elif self.hand_enabled and not self.hand_ready:
            remaining = max(0.0, self.config.calibration_sec - run_time_sec)
            status_text = f"hand calibrating ({remaining:.1f}s)"
        elif hold_active:
            status_text = STATE_CLICK_HOLD
        elif click_events:
            status_text = STATE_CLICK
        elif hand_state in {LEFT, RIGHT}:
            status_text = hand_state
        else:
            status_text = "tracking"

        return TrackingSnapshot(
            mode=self.mode,
            source_name=self.source_name,
            run_time_sec=run_time_sec,
            sample_count=int(self.sample_store.total_samples),
            decoded_state=decoded_state,
            status_text=status_text,
            jaw_confidence=float(self.latest_jaw_step.get("jaw_probability", 0.0)),
            jaw_onset_confidence=float(self.latest_jaw_step.get("jaw_onset_probability", 0.0)),
            jaw_active_confidence=float(self.latest_jaw_step.get("jaw_active_probability", 0.0)),
            jaw_offset_confidence=float(self.latest_jaw_step.get("jaw_offset_probability", 0.0)),
            jaw_event_label=str(self.latest_jaw_step.get("jaw_event_label", GROUND_TRUTH_UNKNOWN)),
            jaw_hold_active=bool(hold_active),
            jaw_enabled=self.jaw_enabled,
            hand_left_confidence=float(self.last_direction_prediction.left_probability),
            hand_right_confidence=float(self.last_direction_prediction.right_probability),
            hand_prediction=str(self.last_direction_prediction.best_label),
            hand_margin=float(self.last_direction_prediction.margin),
            hand_agreed=bool(self.last_direction_prediction.agreed),
            hand_notes=str(self.last_direction_prediction.notes),
            hand_state=hand_state,
            hand_ready=bool(self.hand_ready),
            hand_enabled=self.hand_enabled,
            ground_truth_state=ground_truth_state,
            stale_stream=bool(stale_stream),
            calibration_remaining_sec=max(0.0, self.config.calibration_sec - run_time_sec) if self.hand_enabled and not self.hand_ready else 0.0,
            detections=[event.as_row(self.mode, self.source_name) for event in detections],
            messages=messages,
        )

    def channel_quality_rows(self) -> list[dict[str, Any]]:
        if self.sample_store.total_samples <= 0:
            return []
        frame = self.sample_store.frame_slice(0, self.sample_store.total_samples)
        return compute_channel_quality(frame).to_dict(orient="records")


class ReplayGameController:
    def __init__(
        self,
        config: BCITrackingGameConfig,
        replay_csv: Path,
        speed: float,
        *,
        run_tag: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
        status_callback: Callable[[TrackingSnapshot], None] | None = None,
    ) -> None:
        self.config = config
        self.replay_csv = replay_csv.resolve()
        self.contract = _contract_for_csv(self.replay_csv)
        self.status_callback = status_callback
        self.raw_df = load_openbci_csv(self.replay_csv)
        self.sampling = estimate_sampling(self.raw_df)
        self.fs_hz = float(self.sampling["fs_used_hz"])
        hand_enabled, jaw_enabled = self._enabled_branches()
        self.decoder = BCITrackingDecoder(
            config=config,
            fs_hz=self.fs_hz,
            mode=REPLAY,
            source_name=self.replay_csv.name,
            csv_path=self.replay_csv,
            contract=self.contract,
            hand_enabled=hand_enabled,
            jaw_enabled=jaw_enabled,
        )
        self.decoder.attach_ground_truth(self.raw_df)
        self.playing = False
        self.speed = float(speed)
        self.current_sample = 0
        self.sample_accumulator = 0.0
        self.last_logged_sample_count = -1
        self.logger = TrackingGameLogger(
            output_dir=_create_run_dir(config.output_root, REPLAY, run_tag or self.replay_csv.stem),
            metadata={
                "mode": REPLAY,
                "created_at_utc": _utc_now(),
                "csv_path": str(self.replay_csv),
                "source_name": self.replay_csv.name,
                "protocol_family": self.contract.protocol_family if self.contract is not None else "unknown",
                "marker_semantics": self.contract.marker_semantics if self.contract is not None else "",
                "hand_enabled": hand_enabled,
                "jaw_enabled": jaw_enabled,
                "sampling_note": self.sampling,
                "config": _config_metadata(config),
                **(extra_metadata or {}),
            },
        )
        self.logger.log_ground_truth_segments(self.decoder.ground_truth.segments)
        self.logger.set_classifier_provenance(self.decoder.classifier_provenance())

    def _enabled_branches(self) -> tuple[bool, bool]:
        if self.contract is None:
            return True, True
        if self.contract.protocol_family == "HR":
            return False, True
        if self.contract.protocol_family == "LR":
            return True, False
        if self.contract.protocol_family == "LRJ":
            return True, True
        return True, True

    def set_playing(self, playing: bool) -> None:
        self.playing = bool(playing)

    def set_speed(self, speed: float) -> None:
        self.speed = max(0.1, float(speed))

    def finished(self) -> bool:
        return self.current_sample >= len(self.raw_df)

    def tick(self) -> TrackingSnapshot:
        if self.playing and not self.finished():
            samples_per_tick = self.speed * (self.config.timer_interval_ms / 1000.0) * self.fs_hz
            self.sample_accumulator += samples_per_tick
            advance = int(self.sample_accumulator)
            if advance > 0:
                self.sample_accumulator -= advance
                end_sample = min(len(self.raw_df), self.current_sample + advance)
                batch = {
                    channel: self.raw_df.iloc[self.current_sample:end_sample][channel].to_numpy(dtype=float)
                    for channel in self.decoder.union_channels
                }
                self.decoder.append_batch(batch)
                self.current_sample = end_sample

        snapshot = self.decoder.tick(stale_stream=False)
        self._log_snapshot(snapshot)
        if self.status_callback is not None:
            self.status_callback(snapshot)
        return snapshot

    def _log_snapshot(self, snapshot: TrackingSnapshot) -> None:
        if snapshot.sample_count != self.last_logged_sample_count or snapshot.detections:
            self.last_logged_sample_count = snapshot.sample_count
            self.logger.log_control_trace(
                {
                    "mode": snapshot.mode,
                    "source_name": snapshot.source_name,
                    "run_time_sec": float(snapshot.run_time_sec),
                    "wall_time_utc": _utc_now(),
                    "sample_count": int(snapshot.sample_count),
                    "decoded_state": snapshot.decoded_state,
                    "status_text": snapshot.status_text,
                    "ground_truth_state": snapshot.ground_truth_state,
                    "jaw_confidence": float(snapshot.jaw_confidence),
                    "jaw_onset_confidence": float(snapshot.jaw_onset_confidence),
                    "jaw_active_confidence": float(snapshot.jaw_active_confidence),
                    "jaw_offset_confidence": float(snapshot.jaw_offset_confidence),
                    "jaw_event_label": snapshot.jaw_event_label,
                    "jaw_hold_active": bool(snapshot.jaw_hold_active),
                    "hand_left_confidence": float(snapshot.hand_left_confidence),
                    "hand_right_confidence": float(snapshot.hand_right_confidence),
                    "hand_prediction": snapshot.hand_prediction,
                    "hand_margin": float(snapshot.hand_margin),
                    "hand_agreed": bool(snapshot.hand_agreed),
                    "hand_notes": snapshot.hand_notes,
                    "hand_state": snapshot.hand_state,
                    "hand_ready": bool(snapshot.hand_ready),
                    "hand_enabled": bool(snapshot.hand_enabled),
                    "jaw_enabled": bool(snapshot.jaw_enabled),
                    "stale_stream": bool(snapshot.stale_stream),
                }
            )
            for detection in snapshot.detections:
                self.logger.log_control_event(detection)

    def insert_marker(
        self,
        marker_code: int,
        *,
        marker_role: str,
        phase_name: str,
        protocol_label: str,
        expected_count: int | None = None,
        notes: str = "",
        run_time_sec: float | None = None,
    ) -> None:
        timestamp = self.decoder._current_run_time_sec() if run_time_sec is None else float(run_time_sec)
        self.logger.log_marker(
            {
                "mode": REPLAY,
                "wall_time_utc": _utc_now(),
                "run_time_sec": float(timestamp),
                "sample_count": int(self.decoder.sample_store.total_samples),
                "marker_code": int(marker_code),
                "marker_role": marker_role,
                "phase_name": phase_name,
                "protocol_label": protocol_label,
                "expected_count": expected_count,
                "notes": notes,
            }
        )

    def save(self) -> dict[str, Path]:
        self.logger.metadata["channel_quality"] = self.decoder.channel_quality_rows()
        self.logger.metadata["playback_finished"] = bool(self.finished())
        self.logger.set_signal_trace(self.decoder.export_signal_frame())
        self.logger.set_classifier_provenance(self.decoder.classifier_provenance())
        return self.logger.save()


class LiveGameController:
    def __init__(
        self,
        config: BCITrackingGameConfig,
        *,
        board: str,
        serial_port: str = "",
        playback_file: Path | None = None,
        max_live_sec: float | None = None,
        run_tag: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
        status_callback: Callable[[TrackingSnapshot], None] | None = None,
    ) -> None:
        try:
            from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
        except ModuleNotFoundError as exc:
            raise RuntimeError("brainflow is required for live mode.") from exc

        self.BoardIds = BoardIds
        self.BoardShim = BoardShim
        self.BrainFlowInputParams = BrainFlowInputParams
        self.config = config
        self.board_name = board
        self.serial_port = serial_port
        self.playback_file = playback_file
        self.max_live_sec = max_live_sec
        self.status_callback = status_callback
        self.last_logged_sample_count = -1
        self.last_data_wall_time = time.monotonic()
        self.live_started = False
        self.stopped = False

        self.direction_model = DirectionRuntimeModel(config.direction_artifact_path)
        self.jaw_model = JawRuntimeModel(config.jaw_artifact_path)
        self.union_channels = sorted(set(self.direction_model.selected_channels) | set(self.jaw_model.selected_channels))

        self._prepare_board()
        self.decoder = BCITrackingDecoder(
            config=config,
            fs_hz=self.fs_hz,
            mode=LIVE,
            source_name=f"{board}_live",
            hand_enabled=True,
            jaw_enabled=True,
        )
        self.logger = TrackingGameLogger(
            output_dir=_create_run_dir(config.output_root, LIVE, run_tag or board),
            metadata={
                "mode": LIVE,
                "created_at_utc": _utc_now(),
                "board": board,
                "serial_port": serial_port,
                "playback_file": str(playback_file.resolve()) if playback_file else "",
                "config": _config_metadata(config),
                **(extra_metadata or {}),
            },
        )
        self.logger.set_classifier_provenance(self.decoder.classifier_provenance())

    def _prepare_board(self) -> None:
        _validate_live_board_request(self.board_name, self.serial_port, self.playback_file)

        params = self.BrainFlowInputParams()
        if self.serial_port:
            params.serial_port = self.serial_port
        if self.board_name == "playback":
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
        try:
            self.board.prepare_session()
            self.board.start_stream()
            resolved_board_id = int(self.board.get_board_id())
            self.fs_hz = float(self.BoardShim.get_sampling_rate(resolved_board_id))
            eeg_rows = self.BoardShim.get_eeg_channels(resolved_board_id)
            self.board_rows = _selected_board_rows(self.union_channels, eeg_rows)
            self.live_started = True
        except Exception as exc:
            try:
                self.board.release_session()
            except Exception:
                pass
            raise RuntimeError(_board_error_message(self.board_name, exc)) from exc

    def _append_new_samples(self) -> int:
        data = self.board.get_board_data()
        if data.shape[1] == 0:
            return 0
        batch = {name: np.asarray(data[row_index], dtype=float) for name, row_index in self.board_rows}
        batch_len = len(next(iter(batch.values()))) if batch else 0
        self.decoder.append_batch(batch)
        if batch_len > 0:
            self.last_data_wall_time = time.monotonic()
        return batch_len

    def tick(self) -> TrackingSnapshot:
        if self.stopped:
            return self.decoder.tick(stale_stream=True)

        self._append_new_samples()
        stale_stream = (time.monotonic() - self.last_data_wall_time) >= self.config.stale_stream_warning_sec
        snapshot = self.decoder.tick(stale_stream=stale_stream)
        self._log_snapshot(snapshot)

        if self.max_live_sec is not None and snapshot.run_time_sec >= float(self.max_live_sec):
            self.stopped = True

        if self.status_callback is not None:
            self.status_callback(snapshot)
        return snapshot

    def _log_snapshot(self, snapshot: TrackingSnapshot) -> None:
        if snapshot.sample_count != self.last_logged_sample_count or snapshot.detections:
            self.last_logged_sample_count = snapshot.sample_count
            self.logger.log_control_trace(
                {
                    "mode": snapshot.mode,
                    "source_name": snapshot.source_name,
                    "run_time_sec": float(snapshot.run_time_sec),
                    "wall_time_utc": _utc_now(),
                    "sample_count": int(snapshot.sample_count),
                    "decoded_state": snapshot.decoded_state,
                    "status_text": snapshot.status_text,
                    "ground_truth_state": snapshot.ground_truth_state,
                    "jaw_confidence": float(snapshot.jaw_confidence),
                    "jaw_onset_confidence": float(snapshot.jaw_onset_confidence),
                    "jaw_active_confidence": float(snapshot.jaw_active_confidence),
                    "jaw_offset_confidence": float(snapshot.jaw_offset_confidence),
                    "jaw_event_label": snapshot.jaw_event_label,
                    "jaw_hold_active": bool(snapshot.jaw_hold_active),
                    "hand_left_confidence": float(snapshot.hand_left_confidence),
                    "hand_right_confidence": float(snapshot.hand_right_confidence),
                    "hand_prediction": snapshot.hand_prediction,
                    "hand_margin": float(snapshot.hand_margin),
                    "hand_agreed": bool(snapshot.hand_agreed),
                    "hand_notes": snapshot.hand_notes,
                    "hand_state": snapshot.hand_state,
                    "hand_ready": bool(snapshot.hand_ready),
                    "hand_enabled": bool(snapshot.hand_enabled),
                    "jaw_enabled": bool(snapshot.jaw_enabled),
                    "stale_stream": bool(snapshot.stale_stream),
                }
            )
            for detection in snapshot.detections:
                self.logger.log_control_event(detection)

    def insert_marker(
        self,
        marker_code: int,
        *,
        marker_role: str,
        phase_name: str,
        protocol_label: str,
        expected_count: int | None = None,
        notes: str = "",
        run_time_sec: float | None = None,
    ) -> None:
        timestamp = self.decoder._current_run_time_sec() if run_time_sec is None else float(run_time_sec)
        marker_inserted = False
        if self.live_started:
            try:
                self.board.insert_marker(float(marker_code))
                marker_inserted = True
            except Exception:
                marker_inserted = False
        self.logger.log_marker(
            {
                "mode": LIVE,
                "wall_time_utc": _utc_now(),
                "run_time_sec": float(timestamp),
                "sample_count": int(self.decoder.sample_store.total_samples),
                "marker_code": int(marker_code),
                "marker_role": marker_role,
                "phase_name": phase_name,
                "protocol_label": protocol_label,
                "expected_count": expected_count,
                "notes": notes,
                "board_inserted": marker_inserted,
            }
        )

    def save(self) -> dict[str, Path]:
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
        self.logger.metadata["channel_quality"] = self.decoder.channel_quality_rows()
        self.logger.set_signal_trace(self.decoder.export_signal_frame())
        self.logger.set_classifier_provenance(self.decoder.classifier_provenance())
        return self.logger.save()
