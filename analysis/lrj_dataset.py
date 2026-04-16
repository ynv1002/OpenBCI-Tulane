from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .utils import (
    PROJECT_ROOT,
    collapse_marker_events,
    compute_channel_quality,
    estimate_sampling,
    load_openbci_csv,
    preprocess_session_signals,
)


LABEL_BY_MARKER = {1: "LEFT", 2: "RIGHT", 3: "JAW"}
EXPECTED_BLOCK_ORDER = [1, 2, 3]
EXPECTED_PULSES_PER_CODE = 12
EXPECTED_INTERVALS_PER_CODE = 6
COUNT_SMOOTHING_SAMPLES = 75
COUNT_MIN_PEAK_DISTANCE_SEC = 0.75
COUNT_PROMINENCE_STD_SCALE = 0.60
COUNT_MIN_PROMINENCE_UV = 0.02
COUNT_EARLY_BOUNDARY_EXCLUSION_SEC = 0.10
DEFAULT_LRJ_FS = 250.0
DEFAULT_LRJ_WINDOW_SEC = 1.0
DEFAULT_LRJ_OVERLAP = 0.5
DEFAULT_LRJ_FEATURE_MODE = "combined"
DEFAULT_LRJ_WITH_ASYMMETRY = True


@dataclass(frozen=True)
class LRJSessionSpec:
    subject: str
    display_name: str
    filename: str
    csv_path: Path


OPENBCI_RUNS_ROOT = PROJECT_ROOT.parent / "OPENBCI_runs"
LRJ_SESSION_SPECS: tuple[LRJSessionSpec, ...] = (
    LRJSessionSpec(
        subject="Ben",
        display_name="Ben LRJ",
        filename="Ben-LRJ(1-6)-4:9.csv",
        csv_path=OPENBCI_RUNS_ROOT / "Ben" / "Ben-LRJ(1-6)-4:9.csv",
    ),
    LRJSessionSpec(
        subject="Yaniv",
        display_name="Yaniv LRJ",
        filename="Yaniv-LRJ(6)-4:7.csv",
        csv_path=OPENBCI_RUNS_ROOT / "Yaniv" / "Yaniv-LRJ(6)-4:7.csv",
    ),
)


def lrj_session_specs() -> list[LRJSessionSpec]:
    return list(LRJ_SESSION_SPECS)


def _smooth_values(values: np.ndarray, window_samples: int) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(window=window_samples, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def _channel_index(channel_name: str) -> int:
    return int(str(channel_name).split("_")[-1])


def select_lrj_count_channels(channel_quality_df: pd.DataFrame) -> list[str]:
    usable = channel_quality_df[channel_quality_df["status"] != "unsafe"].copy()
    usable = usable.sort_values("channel", key=lambda series: series.map(_channel_index))
    channels = usable["channel"].astype(str).tolist()
    if not channels:
        raise ValueError("No safe/questionable channels survived the unsafe-channel screen.")
    return channels


def count_family_for_label(label: str) -> str:
    return "jaw" if str(label).upper() == "JAW" else "left_right"


def build_lrj_count_signals(
    raw_df: pd.DataFrame,
    count_channels: Sequence[str],
    fs_hz: float,
) -> dict[str, dict[str, np.ndarray]]:
    signals: dict[str, dict[str, np.ndarray]] = {}
    for family_key in ("left_right", "jaw"):
        filtered = preprocess_session_signals(raw_df, family_key, count_channels, fs_hz)
        aggregate_rms = np.sqrt(np.mean(filtered.to_numpy(dtype=float) ** 2, axis=1))
        aggregate_rms_smooth = _smooth_values(aggregate_rms, COUNT_SMOOTHING_SAMPLES)
        signals[family_key] = {
            "aggregate_rms": aggregate_rms,
            "aggregate_rms_smooth": aggregate_rms_smooth,
        }
    return signals


def _pair_same_code_events(
    code_events: list[dict[str, Any]],
    marker_code: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    intervals: list[dict[str, Any]] = []
    issues: list[str] = []
    if len(code_events) % 2 != 0:
        issues.append(
            f"marker code {marker_code} had an odd pulse count ({len(code_events)}), leaving an unpaired marker"
        )

    for pair_index, start_idx in enumerate(range(0, len(code_events) - 1, 2), start=1):
        start_event = code_events[start_idx]
        end_event = code_events[start_idx + 1]
        intervals.append(
            {
                "marker_code": marker_code,
                "pair_index": pair_index,
                "start_sample": int(start_event["start_sample"]),
                "end_sample": int(end_event["start_sample"]),
                "start_event_end_sample": int(start_event["end_sample"]),
                "end_event_end_sample": int(end_event["end_sample"]),
            }
        )
    return intervals, issues


def _build_marker_rows(
    marker_events: list[dict[str, Any]],
    fs_hz: float,
) -> tuple[pd.DataFrame, dict[int, list[dict[str, Any]]], list[str], list[str]]:
    hard_issues: list[str] = []
    soft_warnings: list[str] = []
    marker_rows: list[dict[str, Any]] = []
    events_by_code: dict[int, list[dict[str, Any]]] = {code: [] for code in LABEL_BY_MARKER}

    unexpected_codes = sorted(
        {
            int(event["code"])
            for event in marker_events
            if int(event["code"]) != 0 and int(event["code"]) not in LABEL_BY_MARKER
        }
    )
    if unexpected_codes:
        hard_issues.append(f"Unexpected nonzero marker values observed: {unexpected_codes}")

    if any(int(event["run_length_samples"]) > 1 for event in marker_events):
        soft_warnings.append("One or more marker values persisted for multiple samples before collapsing.")

    previous_sample: int | None = None
    for event_id, event in enumerate(marker_events, start=1):
        marker_code = int(event["code"])
        if marker_code not in LABEL_BY_MARKER:
            continue

        events_by_code[marker_code].append(event)
        occurrence_index = len(events_by_code[marker_code])
        pair_index = (occurrence_index + 1) // 2
        pair_role = "start" if occurrence_index % 2 == 1 else "end"
        sample_index = int(event["start_sample"])
        gap_from_previous_sec = (
            float((sample_index - previous_sample) / fs_hz) if previous_sample is not None else np.nan
        )
        previous_sample = sample_index

        marker_rows.append(
            {
                "event_id": event_id,
                "marker_code": marker_code,
                "label": LABEL_BY_MARKER[marker_code],
                "sample_index": sample_index,
                "time_sec": float(sample_index / fs_hz),
                "run_length_samples": int(event["run_length_samples"]),
                "gap_from_previous_sec": gap_from_previous_sec,
                "occurrence_index_within_label": occurrence_index,
                "pair_index": pair_index,
                "pair_role": pair_role,
            }
        )

    block_sequence: list[int] = []
    previous_code: int | None = None
    for event in marker_events:
        marker_code = int(event["code"])
        if marker_code not in LABEL_BY_MARKER:
            continue
        if marker_code != previous_code:
            block_sequence.append(marker_code)
            previous_code = marker_code
    if block_sequence != EXPECTED_BLOCK_ORDER:
        hard_issues.append(
            "Marker blocks were not in the expected LEFT -> RIGHT -> JAW order. "
            f"Observed block sequence: {[LABEL_BY_MARKER.get(code, code) for code in block_sequence]}"
        )

    for marker_code, label in LABEL_BY_MARKER.items():
        pulse_count = len(events_by_code[marker_code])
        if pulse_count != EXPECTED_PULSES_PER_CODE:
            hard_issues.append(
                f"{label} marker code {marker_code} appeared {pulse_count} times; expected {EXPECTED_PULSES_PER_CODE}"
            )

    marker_audit_df = pd.DataFrame(marker_rows)
    if marker_audit_df.empty:
        hard_issues.append("No usable nonzero markers were found in the file.")

    return marker_audit_df, events_by_code, hard_issues, soft_warnings


def reconstruct_lrj_trials(
    raw_df: pd.DataFrame,
    fs_hz: float,
    dataset_display_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    if "Marker" not in raw_df.columns:
        raise ValueError(f"The target {dataset_display_name} dataset does not contain a Marker column.")

    marker_events = [event for event in collapse_marker_events(raw_df["Marker"]) if int(event["code"]) != 0]
    marker_audit_df, events_by_code, hard_issues, soft_warnings = _build_marker_rows(marker_events, fs_hz)

    trial_rows: list[dict[str, Any]] = []
    for marker_code, label in LABEL_BY_MARKER.items():
        intervals, pairing_issues = _pair_same_code_events(events_by_code[marker_code], marker_code)
        hard_issues.extend(pairing_issues)
        if len(intervals) != EXPECTED_INTERVALS_PER_CODE:
            hard_issues.append(
                f"{label} produced {len(intervals)} interval(s); expected {EXPECTED_INTERVALS_PER_CODE}"
            )

        for within_label_index, interval in enumerate(intervals, start=1):
            start_sample = int(interval["start_sample"])
            end_sample = int(interval["end_sample"])
            trial_rows.append(
                {
                    "label": label,
                    "marker_code": marker_code,
                    "marker_pair_type": f"{marker_code}->{marker_code}",
                    "trial_index_within_label": within_label_index,
                    "expected_count": within_label_index,
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "start_time_sec": float(start_sample / fs_hz),
                    "end_time_sec": float(end_sample / fs_hz),
                    "duration_sec": float((end_sample - start_sample) / fs_hz),
                    "structural_notes": "",
                }
            )

    trials_df = pd.DataFrame(trial_rows)
    if trials_df.empty:
        return marker_audit_df, trials_df, hard_issues, soft_warnings

    trials_df = trials_df.sort_values(["start_sample", "end_sample"]).reset_index(drop=True)
    trials_df["trial_index_overall"] = np.arange(1, len(trials_df) + 1, dtype=int)
    trials_df["gap_from_previous_trial_sec"] = np.nan

    for index in range(len(trials_df)):
        if index == 0:
            continue
        previous_end = int(trials_df.loc[index - 1, "end_sample"])
        current_start = int(trials_df.loc[index, "start_sample"])
        if current_start <= previous_end:
            hard_issues.append(
                "Reconstructed trial intervals overlap: "
                f"trial {int(trials_df.loc[index - 1, 'trial_index_overall'])} ends at sample {previous_end}, "
                f"trial {int(trials_df.loc[index, 'trial_index_overall'])} starts at sample {current_start}"
            )
        trials_df.loc[index, "gap_from_previous_trial_sec"] = float((current_start - previous_end) / fs_hz)

    label_block_sequence = trials_df["label"].loc[trials_df["label"].ne(trials_df["label"].shift())].tolist()
    if label_block_sequence != [LABEL_BY_MARKER[code] for code in EXPECTED_BLOCK_ORDER]:
        hard_issues.append(
            "Reconstructed trial labels were not ordered LEFT -> RIGHT -> JAW. "
            f"Observed order: {label_block_sequence}"
        )

    short_trials = trials_df[trials_df["duration_sec"] < 1.0]
    if not short_trials.empty:
        soft_warnings.append(
            "One or more reconstructed trials were shorter than 1.0 s: "
            + ", ".join(
                f"{row.label}#{int(row.trial_index_within_label)} ({row.duration_sec:.2f}s)"
                for row in short_trials.itertuples()
            )
        )

    long_trials = trials_df[trials_df["duration_sec"] > 15.0]
    if not long_trials.empty:
        soft_warnings.append(
            "One or more reconstructed trials were longer than 15.0 s: "
            + ", ".join(
                f"{row.label}#{int(row.trial_index_within_label)} ({row.duration_sec:.2f}s)"
                for row in long_trials.itertuples()
            )
        )

    unusual_gaps = trials_df[
        trials_df["gap_from_previous_trial_sec"].notna()
        & ((trials_df["gap_from_previous_trial_sec"] < 0.25) | (trials_df["gap_from_previous_trial_sec"] > 20.0))
    ]
    if not unusual_gaps.empty:
        soft_warnings.append(
            "One or more inter-trial gaps looked unusual: "
            + ", ".join(
                f"trial {int(row.trial_index_overall)} gap {row.gap_from_previous_trial_sec:.2f}s"
                for row in unusual_gaps.itertuples()
            )
        )

    return marker_audit_df, trials_df, hard_issues, soft_warnings


def detect_lrj_trial_counts(
    trials_df: pd.DataFrame,
    count_signals: dict[str, dict[str, np.ndarray]],
    fs_hz: float,
    count_channels: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    min_distance_samples = max(1, int(round(COUNT_MIN_PEAK_DISTANCE_SEC * fs_hz)))

    for trial in trials_df.to_dict(orient="records"):
        family_key = count_family_for_label(str(trial["label"]))
        aggregate_rms = count_signals[family_key]["aggregate_rms"]
        aggregate_rms_smooth = count_signals[family_key]["aggregate_rms_smooth"]

        start_sample = int(trial["start_sample"])
        end_sample = int(trial["end_sample"])
        trial_values = aggregate_rms[start_sample : end_sample + 1]
        trial_smooth = _smooth_values(trial_values, COUNT_SMOOTHING_SAMPLES)
        prominence_threshold = max(float(np.std(trial_smooth)) * COUNT_PROMINENCE_STD_SCALE, COUNT_MIN_PROMINENCE_UV)

        peak_indices, properties = find_peaks(
            trial_smooth,
            prominence=prominence_threshold,
            distance=min_distance_samples,
        )

        detected_times: list[float] = []
        kept_count = 0
        issue_parts: list[str] = []
        for event_index, peak_idx in enumerate(peak_indices, start=1):
            peak_sample = start_sample + int(peak_idx)
            peak_time_sec = float(peak_sample / fs_hz)
            peak_time_relative_sec = float(peak_time_sec - float(trial["start_time_sec"]))
            kept_for_count = peak_time_relative_sec >= COUNT_EARLY_BOUNDARY_EXCLUSION_SEC
            if kept_for_count:
                kept_count += 1
                detected_times.append(round(peak_time_sec, 3))

            event_rows.append(
                {
                    "trial_index_overall": int(trial["trial_index_overall"]),
                    "label": str(trial["label"]),
                    "trial_index_within_label": int(trial["trial_index_within_label"]),
                    "peak_index": event_index,
                    "peak_sample": peak_sample,
                    "peak_time_sec": peak_time_sec,
                    "peak_time_relative_sec": peak_time_relative_sec,
                    "peak_value": float(aggregate_rms_smooth[peak_sample]),
                    "peak_prominence": float(properties["prominences"][event_index - 1]),
                    "kept_for_count": bool(kept_for_count),
                }
            )

        observed_count = int(kept_count)
        expected_count = int(trial["expected_count"])
        count_error = int(observed_count - expected_count)
        count_match = bool(observed_count == expected_count)
        if not count_match:
            issue_parts.append(f"observed_count {observed_count} vs expected_count {expected_count}")

        trial_rows.append(
            {
                **trial,
                "count_filter_family": family_key,
                "count_channels": ", ".join(str(channel) for channel in count_channels),
                "observed_count": observed_count,
                "count_match": count_match,
                "count_error": count_error,
                "detected_event_times_sec": str(detected_times),
                "count_basis": (
                    f"family={family_key}; smooth={COUNT_SMOOTHING_SAMPLES} samples; "
                    f"min_distance={COUNT_MIN_PEAK_DISTANCE_SEC:.2f}s; "
                    f"prominence>={prominence_threshold:.3f}uV; "
                    f"ignore_peaks_before={COUNT_EARLY_BOUNDARY_EXCLUSION_SEC:.2f}s"
                ),
                "issues": "; ".join(part for part in [trial["structural_notes"], *issue_parts] if part),
            }
        )

    ordered_columns = [
        "trial_index_overall",
        "label",
        "marker_code",
        "marker_pair_type",
        "trial_index_within_label",
        "expected_count",
        "observed_count",
        "count_match",
        "count_error",
        "count_filter_family",
        "count_channels",
        "start_sample",
        "end_sample",
        "start_time_sec",
        "end_time_sec",
        "duration_sec",
        "gap_from_previous_trial_sec",
        "detected_event_times_sec",
        "count_basis",
        "issues",
    ]
    return pd.DataFrame(trial_rows)[ordered_columns], pd.DataFrame(event_rows)


def build_lrj_session_record(
    session_spec: LRJSessionSpec,
    fs_hz: float = DEFAULT_LRJ_FS,
) -> dict[str, Any]:
    if not session_spec.csv_path.exists():
        raise ValueError(f"CSV not found: {session_spec.csv_path}")

    raw_df = load_openbci_csv(session_spec.csv_path.resolve())
    sampling_note = estimate_sampling(raw_df)
    marker_audit_df, trials_df, hard_issues, soft_warnings = reconstruct_lrj_trials(
        raw_df=raw_df,
        fs_hz=fs_hz,
        dataset_display_name=session_spec.display_name,
    )
    if trials_df.empty:
        raise ValueError(f"No trial intervals could be reconstructed from {session_spec.display_name}.")

    channel_quality_df = compute_channel_quality(raw_df)
    count_channels = select_lrj_count_channels(channel_quality_df)
    count_signals = build_lrj_count_signals(raw_df, count_channels, fs_hz)
    trial_summary_df, event_candidates_df = detect_lrj_trial_counts(
        trials_df=trials_df,
        count_signals=count_signals,
        fs_hz=fs_hz,
        count_channels=count_channels,
    )

    count_mismatches = trial_summary_df[trial_summary_df["count_match"] == False].copy()
    if not count_mismatches.empty:
        soft_warnings.append(
            f"Diagnostic count overlay mismatched {len(count_mismatches)}/{len(trial_summary_df)} trials."
        )

    return {
        "subject": session_spec.subject,
        "display_name": session_spec.display_name,
        "filename": session_spec.filename,
        "file_path": session_spec.csv_path,
        "raw_df": raw_df,
        "fs_hz": float(fs_hz),
        "sampling": sampling_note,
        "marker_audit_df": marker_audit_df,
        "trials_df": trials_df,
        "trial_summary_df": trial_summary_df,
        "event_candidates_df": event_candidates_df,
        "channel_quality_df": channel_quality_df,
        "channel_quality": channel_quality_df.to_dict(orient="records"),
        "count_channels": list(count_channels),
        "count_signals": count_signals,
        "hard_issues": list(hard_issues),
        "soft_warnings": list(soft_warnings),
    }
