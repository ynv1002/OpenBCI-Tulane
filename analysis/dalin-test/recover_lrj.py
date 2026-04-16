from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.event_utils import EventWindowConfig, build_event_windows
from analysis.jaw_trigger_rules import JawClickTrigger, JawClickTriggerConfig
from analysis.realtime_clench_detector import _load_artifact
from analysis.utils import (
    compute_channel_quality,
    ensure_output_dir,
    estimate_sampling,
    load_openbci_csv,
    preprocess_session_signals,
)


DEFAULT_ARTIFACT_PATH = REPO_ROOT / "analysis" / "outputs" / "realtime_clench_model.pkl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
SMALL_GAP_CLUSTER_SEC = 3.0
LARGE_GAP_BLOCK_SEC = 20.0
CLUSTER_SUPPORT_PADDING_SEC = 1.5
TRIGGER_SUPPORT_RADIUS_SEC = 0.5
NON_JAW_PROBABILITY_CEILING = 0.45
PEAK_PROMINENCE_FLOOR = 0.05
PHASE_LABELS = ("L1", "R1", "J1", "L2", "R2", "J2")
PART1_SUPPORT_PADDING_SEC = 0.75
PART1_PEAK_PROBABILITY_THRESHOLD = 0.70
PART1_PEAK_PROMINENCE_THRESHOLD = 0.05
PART1_PEAK_MIN_SEPARATION_SEC = 0.35
PART1_LIKELY_REAL_PROBABILITY_THRESHOLD = 0.70
PART1_LIKELY_JAW_TRACE_THRESHOLD = 0.80
MIN_TRIAL_DURATION_SEC = 1.5
MAX_TRIAL_DURATION_SEC = 3.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-off LRJ recovery and debugging workflow for the corrupted Dalin recording."
    )
    parser.add_argument("--csv", type=Path, required=True, help="Path to the corrupted LRJ CSV.")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=DEFAULT_ARTIFACT_PATH,
        help="Saved jaw replay artifact to reuse.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for derived recovery outputs.",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=250.0,
        help="Authoritative sample rate for reconstruction timing.",
    )
    parser.add_argument(
        "--cluster-gap-sec",
        type=float,
        default=SMALL_GAP_CLUSTER_SEC,
        help="Gap threshold for small marker clusters.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the debug probability plot.",
    )
    parser.add_argument(
        "--part1",
        action="store_true",
        help="Add Part 1-only cluster interpretation outputs through the last marker.",
    )
    parser.add_argument(
        "--pairs",
        action="store_true",
        help="Add marker-pair trial analysis outputs built from consecutive collapsed marker pairs.",
    )
    return parser.parse_args()


def _fail(message: str) -> None:
    raise SystemExit(message)


def _detect_marker_column(df: pd.DataFrame) -> tuple[str, pd.Series, str]:
    if "Marker" in df.columns:
        return "Marker", pd.to_numeric(df["Marker"], errors="coerce").fillna(0.0), "used Marker column"

    trailing_columns = list(df.columns[-8:])
    best_column = None
    best_series = None
    best_score = None

    for column in reversed(trailing_columns):
        series = pd.to_numeric(df[column], errors="coerce")
        valid = series.dropna()
        if valid.empty:
            continue
        integer_like_fraction = float(np.mean(np.isclose(valid, np.round(valid), atol=1e-6)))
        nonzero_fraction = float(np.mean(~np.isclose(valid, 0.0, atol=1e-12)))
        trailing_rank = trailing_columns.index(column)
        score = (integer_like_fraction, -nonzero_fraction, -trailing_rank)
        if best_score is None or score > best_score:
            best_score = score
            best_column = str(column)
            best_series = series.fillna(0.0)

    if best_column is None or best_series is None:
        fallback = str(df.columns[-1])
        return fallback, pd.to_numeric(df.iloc[:, -1], errors="coerce").fillna(0.0), f"fallback to {fallback}"

    return (
        best_column,
        best_series,
        f"fallback to sparsest integer-like trailing column {best_column}",
    )


def _collapse_marker_events(marker_series: pd.Series) -> list[dict[str, Any]]:
    rounded = pd.to_numeric(marker_series, errors="coerce").fillna(0.0).round().astype(int).to_numpy()
    events: list[dict[str, Any]] = []
    active_code = 0
    start_idx = 0

    for idx, code in enumerate(rounded):
        if code != 0 and active_code == 0:
            active_code = int(code)
            start_idx = idx
        elif code == 0 and active_code != 0:
            events.append(
                {
                    "code": active_code,
                    "start_sample": start_idx,
                    "end_sample": idx - 1,
                    "run_length_samples": idx - start_idx,
                }
            )
            active_code = 0
        elif code != 0 and active_code != 0 and code != active_code:
            events.append(
                {
                    "code": active_code,
                    "start_sample": start_idx,
                    "end_sample": idx - 1,
                    "run_length_samples": idx - start_idx,
                }
            )
            active_code = int(code)
            start_idx = idx

    if active_code != 0:
        events.append(
            {
                "code": active_code,
                "start_sample": start_idx,
                "end_sample": len(rounded) - 1,
                "run_length_samples": len(rounded) - start_idx,
            }
        )

    return events


def _build_marker_audit(
    marker_events: list[dict[str, Any]],
    fs_hz: float,
    cluster_gap_sec: float,
    block_gap_sec: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cluster_id = 1
    block_id = 1
    previous_start = None

    for event_index, event in enumerate(marker_events, start=1):
        sample_index = int(event["start_sample"])
        time_sec = float(sample_index / fs_hz)
        gap_from_previous_sec = None
        if previous_start is not None:
            gap_from_previous_sec = float((sample_index - previous_start) / fs_hz)
            if gap_from_previous_sec > cluster_gap_sec:
                cluster_id += 1
            if gap_from_previous_sec > block_gap_sec:
                block_id += 1

        rows.append(
            {
                "event_id": event_index,
                "sample_index": sample_index,
                "time_sec": time_sec,
                "original_marker_value": int(event["code"]),
                "run_length_samples": int(event["run_length_samples"]),
                "gap_from_previous_sec": gap_from_previous_sec,
                "cluster_id": cluster_id,
                "block_id": block_id,
            }
        )
        previous_start = sample_index

    marker_audit_df = pd.DataFrame(rows)
    if not marker_audit_df.empty:
        marker_audit_df["cluster_id"] = marker_audit_df["cluster_id"].astype("Int64")
        marker_audit_df["block_id"] = marker_audit_df["block_id"].astype("Int64")
    return marker_audit_df


def _build_recovery_sample_frame(
    raw_df: pd.DataFrame,
    csv_path: Path,
    selected_channels: Iterable[str],
    fs_hz: float,
    smoothing_sec: float,
) -> tuple[pd.DataFrame, list[str]]:
    selected_channels = list(selected_channels)
    filtered = preprocess_session_signals(raw_df, "jaw", selected_channels, fs_hz)
    signal_columns = [f"signal_{column}" for column in selected_channels]
    filtered = filtered.rename(columns={column: f"signal_{column}" for column in selected_channels})

    sample_frame = pd.DataFrame(
        {
            "filename": csv_path.name,
            "file_path": str(csv_path.resolve()),
            "parsed_date": None,
            "session_rank": 0,
            "split_role": "recovery",
            "sample_row": np.arange(len(raw_df), dtype=int),
            "sample_index_raw": raw_df["Sample_Index"].to_numpy(dtype=float),
            "time_sec": np.arange(len(raw_df), dtype=float) / fs_hz,
            "event_label": "UNLABELED",
            "coarse_label": "UNLABELED",
        }
    )
    for column in signal_columns:
        sample_frame[column] = filtered[column].to_numpy(dtype=float)

    aggregate_rms = np.sqrt(np.mean(filtered.to_numpy(dtype=float) ** 2, axis=1))
    smoothing_samples = max(1, int(round(smoothing_sec * fs_hz)))
    aggregate_rms_smooth = (
        pd.Series(aggregate_rms).rolling(smoothing_samples, center=True, min_periods=1).mean().to_numpy(dtype=float)
    )
    sample_frame["aggregate_rms"] = aggregate_rms
    sample_frame["aggregate_rms_smooth"] = aggregate_rms_smooth
    return sample_frame, signal_columns


def _score_recovery_windows(
    sample_frame: pd.DataFrame,
    signal_columns: list[str],
    model_bundle: dict[str, Any],
    artifact: dict[str, Any],
    fs_hz: float,
) -> pd.DataFrame:
    window_config = EventWindowConfig(**artifact["window_config"])
    window_bundle = build_event_windows(sample_frame, signal_columns, window_config)
    score_frame = window_bundle["window_frame"].copy()

    feature_columns = list(model_bundle["feature_columns"])
    missing = [column for column in feature_columns if column not in score_frame.columns]
    if missing:
        _fail(f"Recovery score frame is missing required feature columns: {missing}")

    X = score_frame[feature_columns].to_numpy(dtype=float)

    jaw_model = model_bundle["jaw_4state_model"]
    jaw_labels = list(model_bundle["jaw_4state_labels"])
    jaw_probs = jaw_model.predict_proba(X)
    for label_index, label_name in enumerate(jaw_labels):
        score_frame[f"prob_4state_{label_name}"] = jaw_probs[:, label_index]
    score_frame["pred_4state"] = jaw_model.predict(X)

    binary_model = model_bundle["binary_model"]
    binary_labels = list(model_bundle["binary_labels"])
    binary_probs = binary_model.predict_proba(X)
    for label_index, label_name in enumerate(binary_labels):
        score_frame[f"prob_binary_{label_name}"] = binary_probs[:, label_index]
    score_frame["pred_binary"] = binary_model.predict(X)

    trigger_config = JawClickTriggerConfig(**artifact["trigger_config"])
    detector = JawClickTrigger(trigger_config)
    click_flags: list[bool] = []
    click_reasons: list[str] = []
    armed_states: list[bool] = []
    smoothed_clench: list[float] = []
    smoothed_onset: list[float] = []
    smoothed_active: list[float] = []

    for _, row in score_frame.sort_values("center_time_sec").iterrows():
        scores = {
            "clench_probability": float(row.get("prob_binary_CLENCH", 0.0)),
            "non_clench_probability": float(row.get("prob_binary_NON_CLENCH", 0.0)),
            "inactive_probability": float(row.get("prob_4state_INACTIVE", 0.0)),
            "onset_probability": float(row.get("prob_4state_ONSET", 0.0)),
            "active_probability": float(row.get("prob_4state_ACTIVE", 0.0)),
            "offset_probability": float(row.get("prob_4state_OFFSET", 0.0)),
            "envelope_uv": float(row.get("aggregate_rms_smooth_mean", 0.0)),
        }
        decision = detector.step(
            timestamp_sec=float(row["center_time_sec"]),
            scores=scores,
            event_label=str(row["pred_4state"]),
        )
        click_flags.append(bool(decision.emitted_click))
        click_reasons.append(str(decision.reason))
        armed_states.append(bool(decision.armed))
        smoothed_clench.append(float(decision.smoothed_scores.get("clench_probability", 0.0)))
        smoothed_onset.append(float(decision.smoothed_scores.get("onset_probability", 0.0)))
        smoothed_active.append(float(decision.smoothed_scores.get("active_probability", 0.0)))

    window_samples = max(1, int(round(float(window_config.window_sec) * fs_hz)))
    score_frame["window_start_sample"] = score_frame["start_sample"].astype(int)
    score_frame["window_end_sample"] = score_frame["end_sample"].astype(int)
    score_frame["window_center_sample"] = score_frame["window_start_sample"] + (window_samples // 2)
    score_frame["center_time_sec"] = score_frame["window_center_sample"] / fs_hz
    score_frame["trigger_click"] = click_flags
    score_frame["trigger_reason"] = click_reasons
    score_frame["trigger_armed_after_step"] = armed_states
    score_frame["trigger_smoothed_clench_probability"] = smoothed_clench
    score_frame["trigger_smoothed_onset_probability"] = smoothed_onset
    score_frame["trigger_smoothed_active_probability"] = smoothed_active
    return score_frame.sort_values("window_center_sample").reset_index(drop=True)


def _distance_to_span(value_sec: float, start_sec: float, end_sec: float) -> float:
    if start_sec <= value_sec <= end_sec:
        return 0.0
    if value_sec < start_sec:
        return float(start_sec - value_sec)
    return float(value_sec - end_sec)


def _part1_last_marker_time_sec(marker_audit_df: pd.DataFrame) -> float:
    if marker_audit_df.empty:
        return 0.0
    return float(marker_audit_df["time_sec"].max())


def _build_cluster_index(marker_audit_df: pd.DataFrame) -> pd.DataFrame:
    if marker_audit_df.empty:
        return pd.DataFrame()

    cluster_df = (
        marker_audit_df.groupby(["cluster_id", "block_id"], dropna=False)
        .agg(
            cluster_start_sample=("sample_index", "min"),
            cluster_end_sample=("sample_index", "max"),
            cluster_start_time_sec=("time_sec", "min"),
            cluster_end_time_sec=("time_sec", "max"),
            marker_count=("event_id", "count"),
            original_marker_values=("original_marker_value", lambda values: json.dumps(list(map(int, values)))),
        )
        .reset_index()
        .sort_values(["cluster_start_sample", "cluster_id"])
        .reset_index(drop=True)
    )
    return cluster_df


def _build_jaw_events(
    score_trace_df: pd.DataFrame,
    marker_audit_df: pd.DataFrame,
    cluster_index_df: pd.DataFrame,
    trigger_config: JawClickTriggerConfig,
    fs_hz: float,
) -> pd.DataFrame:
    if score_trace_df.empty:
        return pd.DataFrame()

    trace = score_trace_df.sort_values("window_center_sample").reset_index(drop=True)
    probability = trace["prob_binary_CLENCH"].to_numpy(dtype=float)
    center_times = trace["center_time_sec"].to_numpy(dtype=float)
    center_samples = trace["window_center_sample"].to_numpy(dtype=int)
    step_interval_sec = float(np.median(np.diff(center_times))) if len(trace) > 1 else float(
        trace["window_sec"].iloc[0]
    )
    peak_threshold = max(0.70, float(trigger_config.clench_probability_threshold) - 0.10)
    distance_steps = max(
        1,
        int(round(max(trigger_config.minimum_separation_ms / 1000.0, 0.35) / max(step_interval_sec, 1e-6))),
    )

    peak_indices, peak_properties = find_peaks(
        probability,
        height=peak_threshold,
        prominence=PEAK_PROMINENCE_FLOOR,
        distance=distance_steps,
    )

    marker_times = marker_audit_df["time_sec"].to_numpy(dtype=float) if not marker_audit_df.empty else np.array([])
    last_marker_time = float(marker_times.max()) if len(marker_times) else float("-inf")
    click_times = trace.loc[trace["trigger_click"], "center_time_sec"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []

    for peak_order, peak_index in enumerate(peak_indices, start=1):
        peak_time = float(center_times[peak_index])
        peak_sample = int(center_samples[peak_index])
        prominence = float(peak_properties["prominences"][peak_order - 1])
        nearby_triggers = click_times[np.abs(click_times - peak_time) <= TRIGGER_SUPPORT_RADIUS_SEC]

        nearest_marker_id = pd.NA
        nearest_marker_time = np.nan
        nearest_marker_delta_sec = np.nan
        if len(marker_times):
            nearest_marker_position = int(np.argmin(np.abs(marker_times - peak_time)))
            nearest_marker_row = marker_audit_df.iloc[nearest_marker_position]
            nearest_marker_id = int(nearest_marker_row["event_id"])
            nearest_marker_time = float(nearest_marker_row["time_sec"])
            nearest_marker_delta_sec = float(peak_time - nearest_marker_time)

        assigned_cluster_id: Any = pd.NA
        assigned_block_id: Any = pd.NA
        cluster_anchor_status = "unanchored"
        cluster_distance_sec = np.nan
        if not cluster_index_df.empty:
            distances = cluster_index_df.apply(
                lambda row: _distance_to_span(
                    peak_time,
                    float(row["cluster_start_time_sec"]) - CLUSTER_SUPPORT_PADDING_SEC,
                    float(row["cluster_end_time_sec"]) + CLUSTER_SUPPORT_PADDING_SEC,
                ),
                axis=1,
            )
            nearest_cluster_position = int(distances.idxmin())
            nearest_cluster = cluster_index_df.loc[nearest_cluster_position]
            cluster_distance_sec = float(distances.loc[nearest_cluster_position])
            if np.isclose(cluster_distance_sec, 0.0, atol=1e-9):
                assigned_cluster_id = int(nearest_cluster["cluster_id"])
                assigned_block_id = int(nearest_cluster["block_id"])
                cluster_anchor_status = "cluster_aligned"
            elif peak_time > last_marker_time + CLUSTER_SUPPORT_PADDING_SEC:
                cluster_anchor_status = "post_marker_tail"

        rows.append(
            {
                "jaw_event_id": peak_order,
                "peak_sample_index": peak_sample,
                "peak_time_sec": peak_time,
                "peak_probability": float(probability[peak_index]),
                "peak_prominence": prominence,
                "pred_4state": str(trace.iloc[peak_index]["pred_4state"]),
                "trigger_click_nearby": bool(len(nearby_triggers) > 0),
                "trigger_click_count_nearby": int(len(nearby_triggers)),
                "trigger_click_times_sec": json.dumps([round(float(value), 6) for value in nearby_triggers.tolist()]),
                "nearest_marker_id": nearest_marker_id,
                "nearest_marker_time_sec": nearest_marker_time,
                "nearest_marker_delta_sec": nearest_marker_delta_sec,
                "assigned_cluster_id": assigned_cluster_id,
                "assigned_block_id": assigned_block_id,
                "cluster_anchor_status": cluster_anchor_status,
                "cluster_distance_sec": cluster_distance_sec,
                "suggested_sequence_position": pd.NA,
            }
        )

    jaw_events_df = pd.DataFrame(rows)
    for column in ("nearest_marker_id", "assigned_cluster_id", "assigned_block_id"):
        if column in jaw_events_df.columns:
            jaw_events_df[column] = jaw_events_df[column].astype("Int64")
    return jaw_events_df


def _score_phase_assignment(phase_label: str, jaw_support_level: str) -> float:
    is_jaw_phase = phase_label in {"J1", "J2"}
    if is_jaw_phase and jaw_support_level == "strong":
        return 3.0
    if is_jaw_phase and jaw_support_level == "moderate":
        return 1.5
    if is_jaw_phase and jaw_support_level == "low":
        return -2.0
    if (not is_jaw_phase) and jaw_support_level == "strong":
        return -2.5
    if (not is_jaw_phase) and jaw_support_level == "moderate":
        return -1.0
    if (not is_jaw_phase) and jaw_support_level == "low":
        return 0.75
    return 0.0


def _apply_block_phase_fit(cluster_summary_df: pd.DataFrame) -> pd.DataFrame:
    if cluster_summary_df.empty:
        return cluster_summary_df

    out = cluster_summary_df.copy()
    out["phase_suggestion"] = pd.NA
    out["phase_fit_accepted"] = False

    for block_id, block_df in out.groupby("block_id", dropna=False, sort=False):
        ordered = block_df.sort_values(["cluster_start_sample", "cluster_id"]).reset_index()
        if ordered.empty:
            continue

        scored_offsets: list[tuple[int, float, list[str]]] = []
        for offset in range(len(PHASE_LABELS)):
            assigned = [PHASE_LABELS[(idx + offset) % len(PHASE_LABELS)] for idx in range(len(ordered))]
            score = 0.0
            for phase_label, jaw_support_level in zip(assigned, ordered["jaw_support_level"], strict=True):
                score += _score_phase_assignment(phase_label, str(jaw_support_level))
            scored_offsets.append((offset, score, assigned))

        scored_offsets.sort(key=lambda item: item[1], reverse=True)
        best_offset, best_score, best_assignment = scored_offsets[0]
        second_best_score = scored_offsets[1][1] if len(scored_offsets) > 1 else float("-inf")
        best_is_unique = best_score >= second_best_score + 3.0
        jaw_slots_with_strong_support = sum(
            1
            for phase_label, jaw_support_level in zip(best_assignment, ordered["jaw_support_level"], strict=True)
            if phase_label in {"J1", "J2"} and str(jaw_support_level) == "strong"
        )

        accepted = bool(best_score > 0.0 and best_is_unique and jaw_slots_with_strong_support >= 1)
        out.loc[ordered["index"], "phase_fit_accepted"] = accepted
        if accepted:
            out.loc[ordered["index"], "phase_suggestion"] = best_assignment
    return out


def _build_cluster_summary(
    marker_audit_df: pd.DataFrame,
    score_trace_df: pd.DataFrame,
    jaw_events_df: pd.DataFrame,
    trigger_config: JawClickTriggerConfig,
) -> pd.DataFrame:
    cluster_index_df = _build_cluster_index(marker_audit_df)
    if cluster_index_df.empty:
        return cluster_index_df

    rows: list[dict[str, Any]] = []
    for _, cluster in cluster_index_df.iterrows():
        cluster_id = int(cluster["cluster_id"])
        block_id = int(cluster["block_id"])
        support_start = float(cluster["cluster_start_time_sec"]) - CLUSTER_SUPPORT_PADDING_SEC
        support_end = float(cluster["cluster_end_time_sec"]) + CLUSTER_SUPPORT_PADDING_SEC

        support_trace = score_trace_df[
            (score_trace_df["center_time_sec"] >= support_start) & (score_trace_df["center_time_sec"] <= support_end)
        ].copy()
        cluster_events = jaw_events_df[jaw_events_df["assigned_cluster_id"] == cluster_id].copy()
        best_event = (
            cluster_events.sort_values(["peak_probability", "peak_prominence"], ascending=[False, False]).head(1)
        )
        trigger_click_count = int(support_trace["trigger_click"].sum()) if not support_trace.empty else 0
        max_clench_probability = float(support_trace["prob_binary_CLENCH"].max()) if not support_trace.empty else np.nan

        best_jaw_event_time = float(best_event.iloc[0]["peak_time_sec"]) if not best_event.empty else np.nan
        best_jaw_event_probability = (
            float(best_event.iloc[0]["peak_probability"]) if not best_event.empty else np.nan
        )

        if (
            not best_event.empty
            and (
                best_jaw_event_probability >= float(trigger_config.clench_probability_threshold)
                or bool(best_event.iloc[0]["trigger_click_nearby"])
            )
        ) or max_clench_probability >= float(trigger_config.clench_probability_threshold):
            jaw_support_level = "strong"
        elif not best_event.empty:
            jaw_support_level = "moderate"
        elif np.isfinite(max_clench_probability) and max_clench_probability < NON_JAW_PROBABILITY_CEILING:
            jaw_support_level = "low"
        else:
            jaw_support_level = "ambiguous"

        rows.append(
            {
                "cluster_id": cluster_id,
                "block_id": block_id,
                "cluster_start_sample": int(cluster["cluster_start_sample"]),
                "cluster_end_sample": int(cluster["cluster_end_sample"]),
                "cluster_start_time_sec": float(cluster["cluster_start_time_sec"]),
                "cluster_end_time_sec": float(cluster["cluster_end_time_sec"]),
                "marker_count": int(cluster["marker_count"]),
                "original_marker_values": cluster["original_marker_values"],
                "cluster_support_start_sec": support_start,
                "cluster_support_end_sec": support_end,
                "max_clench_probability": max_clench_probability,
                "trigger_click_count": trigger_click_count,
                "jaw_event_count": int(len(cluster_events)),
                "best_jaw_event_time_sec": best_jaw_event_time,
                "best_jaw_event_probability": best_jaw_event_probability,
                "jaw_support_level": jaw_support_level,
                "phase_suggestion": pd.NA,
                "phase_fit_accepted": False,
                "suggested_label": "unknown",
                "suggestion_basis": "no conservative reconstruction suggestion was strong enough",
            }
        )

    cluster_summary_df = pd.DataFrame(rows)
    cluster_summary_df = _apply_block_phase_fit(cluster_summary_df)

    for idx, row in cluster_summary_df.iterrows():
        phase_suggestion = row["phase_suggestion"]
        jaw_support_level = str(row["jaw_support_level"])
        best_prob = row["best_jaw_event_probability"]
        max_prob = row["max_clench_probability"]

        if jaw_support_level in {"strong", "moderate"}:
            if row["phase_fit_accepted"] and phase_suggestion in {"J1", "J2"} and jaw_support_level == "strong":
                cluster_summary_df.loc[idx, "suggested_label"] = f"likely {phase_suggestion}"
                cluster_summary_df.loc[idx, "suggestion_basis"] = (
                    f"strong jaw model evidence (best peak P={best_prob:.3f}) plus an accepted block-level "
                    f"{phase_suggestion} phase fit"
                )
            else:
                cluster_summary_df.loc[idx, "suggested_label"] = "likely jaw"
                cluster_summary_df.loc[idx, "suggestion_basis"] = (
                    f"jaw model evidence was present (best peak P={best_prob:.3f}, max trace P={max_prob:.3f}), "
                    "but sequence placement was kept conservative"
                )
        elif jaw_support_level == "low" and row["phase_fit_accepted"] and phase_suggestion in {"L1", "R1", "L2", "R2"}:
            cluster_summary_df.loc[idx, "suggested_label"] = "likely non-jaw"
            cluster_summary_df.loc[idx, "suggestion_basis"] = (
                f"low jaw probability support (max trace P={max_prob:.3f}) plus an accepted block-level "
                f"{phase_suggestion} phase fit"
            )
        elif jaw_support_level == "low":
            cluster_summary_df.loc[idx, "suggestion_basis"] = (
                f"max trace P={max_prob:.3f} stayed low, but non-jaw labeling was not forced without a clear phase fit"
            )
        else:
            cluster_summary_df.loc[idx, "suggestion_basis"] = (
                "marker timing and model evidence disagreed or stayed ambiguous, so the cluster remains unknown"
            )

    return cluster_summary_df.sort_values(["cluster_start_sample", "cluster_id"]).reset_index(drop=True)


def _attach_sequence_suggestions(
    jaw_events_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    if jaw_events_df.empty or cluster_summary_df.empty:
        return jaw_events_df

    cluster_phase_map = (
        cluster_summary_df.set_index("cluster_id")[["phase_suggestion", "phase_fit_accepted"]].to_dict(orient="index")
    )
    out = jaw_events_df.copy()
    sequence_positions: list[Any] = []

    for _, row in out.iterrows():
        cluster_id = row["assigned_cluster_id"]
        if pd.isna(cluster_id) or int(cluster_id) not in cluster_phase_map:
            sequence_positions.append(pd.NA)
            continue
        phase_info = cluster_phase_map[int(cluster_id)]
        phase_suggestion = phase_info["phase_suggestion"]
        if phase_info["phase_fit_accepted"] and phase_suggestion in {"J1", "J2"}:
            sequence_positions.append(phase_suggestion)
        else:
            sequence_positions.append(pd.NA)

    out["suggested_sequence_position"] = sequence_positions
    return out


def _part1_support_window(
    start_time_sec: float,
    end_time_sec: float,
    last_marker_time_sec: float,
) -> tuple[float, float]:
    return (
        max(0.0, float(start_time_sec) - PART1_SUPPORT_PADDING_SEC),
        min(float(end_time_sec) + PART1_SUPPORT_PADDING_SEC, float(last_marker_time_sec)),
    )


def _part1_sequence_score(phase_label: str, jaw_vs_nonjaw_label: str) -> float:
    is_jaw_phase = phase_label in {"J1", "J2"}
    if jaw_vs_nonjaw_label == "likely jaw":
        return 3.0 if is_jaw_phase else -3.0
    if jaw_vs_nonjaw_label == "likely non-jaw":
        return -3.0 if is_jaw_phase else 2.0
    return 0.0


def _load_collapsed_markers_from_csv(marker_audit_path: Path) -> pd.DataFrame:
    marker_audit_df = pd.read_csv(marker_audit_path)
    required_columns = {
        "event_id",
        "sample_index",
        "time_sec",
        "original_marker_value",
        "run_length_samples",
        "cluster_id",
        "block_id",
    }
    missing = sorted(required_columns - set(marker_audit_df.columns))
    if missing:
        _fail(f"marker_audit.csv is missing required columns for pair analysis: {missing}")
    return marker_audit_df.sort_values(["event_id", "time_sec"]).reset_index(drop=True)


def _format_marker_pair_type(start_value: Any, end_value: Any) -> str:
    return f"{int(start_value)}→{int(end_value)}"


def _build_valid_trials_from_marker_audit(marker_audit_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, Any]] = []
    stats = {
        "candidate_pairs": 0,
        "valid_trials": 0,
        "duration_rejected_pairs": 0,
    }

    if len(marker_audit_df) < 2:
        return pd.DataFrame(rows), stats

    ordered = marker_audit_df.sort_values(["event_id", "time_sec"]).reset_index(drop=True)
    for position in range(len(ordered) - 1):
        current = ordered.iloc[position]
        following = ordered.iloc[position + 1]
        stats["candidate_pairs"] += 1

        start_time_sec = float(current["time_sec"])
        end_time_sec = float(following["time_sec"])
        duration_sec = float(end_time_sec - start_time_sec)
        if duration_sec < MIN_TRIAL_DURATION_SEC or duration_sec > MAX_TRIAL_DURATION_SEC:
            stats["duration_rejected_pairs"] += 1
            continue

        rows.append(
            {
                "trial_id": len(rows) + 1,
                "start_time_sec": start_time_sec,
                "end_time_sec": end_time_sec,
                "duration_sec": duration_sec,
                "marker_pair_type": _format_marker_pair_type(
                    current["original_marker_value"],
                    following["original_marker_value"],
                ),
                "start_event_id": int(current["event_id"]),
                "end_event_id": int(following["event_id"]),
            }
        )
        stats["valid_trials"] += 1

    return pd.DataFrame(rows), stats


def _extract_trial_peaks(
    score_trace_df: pd.DataFrame,
    start_time_sec: float,
    end_time_sec: float,
) -> pd.DataFrame:
    trial_trace = score_trace_df[
        (score_trace_df["center_time_sec"] >= start_time_sec)
        & (score_trace_df["center_time_sec"] <= end_time_sec)
    ].copy().sort_values("center_time_sec")
    if trial_trace.empty:
        return pd.DataFrame(columns=["peak_time_sec", "peak_probability", "peak_prominence"])

    peak_times = trial_trace["center_time_sec"].to_numpy(dtype=float)
    peak_values = trial_trace["prob_binary_CLENCH"].to_numpy(dtype=float)
    if len(trial_trace) < 2:
        return pd.DataFrame(columns=["peak_time_sec", "peak_probability", "peak_prominence"])

    step_interval_sec = float(np.median(np.diff(peak_times)))
    distance_steps = max(
        1,
        int(round(PART1_PEAK_MIN_SEPARATION_SEC / max(step_interval_sec, 1e-6))),
    )
    peak_indices, properties = find_peaks(
        peak_values,
        height=PART1_PEAK_PROBABILITY_THRESHOLD,
        prominence=PART1_PEAK_PROMINENCE_THRESHOLD,
        distance=distance_steps,
    )
    if len(peak_indices) == 0:
        return pd.DataFrame(columns=["peak_time_sec", "peak_probability", "peak_prominence"])

    return pd.DataFrame(
        {
            "peak_time_sec": peak_times[peak_indices],
            "peak_probability": peak_values[peak_indices],
            "peak_prominence": properties["prominences"],
        }
    ).sort_values("peak_time_sec").reset_index(drop=True)


def _build_trial_summary(
    marker_audit_path: Path,
    score_trace_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    marker_audit_df = _load_collapsed_markers_from_csv(marker_audit_path)
    trials_df, trial_stats = _build_valid_trials_from_marker_audit(marker_audit_df)
    if trials_df.empty:
        return pd.DataFrame(), pd.DataFrame(), trial_stats

    trial_rows: list[dict[str, Any]] = []
    peak_rows: list[dict[str, Any]] = []
    for _, trial in trials_df.iterrows():
        start_time_sec = float(trial["start_time_sec"])
        end_time_sec = float(trial["end_time_sec"])
        trial_trace = score_trace_df[
            (score_trace_df["center_time_sec"] >= start_time_sec)
            & (score_trace_df["center_time_sec"] <= end_time_sec)
        ].copy()
        max_clench_probability = (
            float(trial_trace["prob_binary_CLENCH"].max()) if not trial_trace.empty else float("nan")
        )
        trial_peaks = _extract_trial_peaks(score_trace_df, start_time_sec, end_time_sec)
        peak_count = int(len(trial_peaks))

        if peak_count >= 1 or (np.isfinite(max_clench_probability) and max_clench_probability >= PART1_LIKELY_JAW_TRACE_THRESHOLD):
            jaw_label = "likely jaw"
        elif np.isfinite(max_clench_probability) and max_clench_probability < NON_JAW_PROBABILITY_CEILING:
            jaw_label = "non-jaw"
        else:
            jaw_label = "unknown"

        trial_rows.append(
            {
                "trial_id": int(trial["trial_id"]),
                "start_time_sec": start_time_sec,
                "end_time_sec": end_time_sec,
                "duration_sec": float(trial["duration_sec"]),
                "marker_pair_type": str(trial["marker_pair_type"]),
                "max_clench_probability": max_clench_probability,
                "peak_count": peak_count,
                "jaw_label": jaw_label,
            }
        )

        for _, peak in trial_peaks.iterrows():
            peak_rows.append(
                {
                    "trial_id": int(trial["trial_id"]),
                    "peak_time_sec": float(peak["peak_time_sec"]),
                    "peak_probability": float(peak["peak_probability"]),
                    "peak_prominence": float(peak["peak_prominence"]),
                }
            )

    return pd.DataFrame(trial_rows), pd.DataFrame(peak_rows), trial_stats


def _build_part1_cluster_interpretation(
    marker_audit_df: pd.DataFrame,
    score_trace_df: pd.DataFrame,
    jaw_events_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    last_marker_time_sec = _part1_last_marker_time_sec(marker_audit_df)
    part1_clusters_df = (
        cluster_summary_df[cluster_summary_df["cluster_end_time_sec"] <= last_marker_time_sec + 1e-9]
        .copy()
        .sort_values(["cluster_start_sample", "cluster_id"])
        .reset_index(drop=True)
    )
    part1_score_trace_df = score_trace_df[score_trace_df["center_time_sec"] <= last_marker_time_sec + 1e-9].copy()
    part1_jaw_events_df = jaw_events_df[jaw_events_df["peak_time_sec"] <= last_marker_time_sec + 1e-9].copy()

    rows: list[dict[str, Any]] = []
    for _, cluster in part1_clusters_df.iterrows():
        cluster_id = int(cluster["cluster_id"])
        start_time_sec = float(cluster["cluster_start_time_sec"])
        end_time_sec = float(cluster["cluster_end_time_sec"])
        support_start_sec, support_end_sec = _part1_support_window(
            start_time_sec,
            end_time_sec,
            last_marker_time_sec,
        )

        support_trace = part1_score_trace_df[
            (part1_score_trace_df["center_time_sec"] >= support_start_sec)
            & (part1_score_trace_df["center_time_sec"] <= support_end_sec)
        ].copy()
        support_peaks = part1_jaw_events_df[
            (part1_jaw_events_df["peak_time_sec"] >= support_start_sec)
            & (part1_jaw_events_df["peak_time_sec"] <= support_end_sec)
        ].copy().sort_values("peak_time_sec")

        max_clench_probability = (
            float(support_trace["prob_binary_CLENCH"].max()) if not support_trace.empty else float("nan")
        )
        jaw_event_count = int(len(support_peaks))
        best_peak_time_sec = float("nan")
        best_peak_probability = float("nan")
        if not support_peaks.empty:
            best_peak = support_peaks.sort_values(
                ["peak_probability", "peak_prominence", "peak_time_sec"],
                ascending=[False, False, True],
            ).iloc[0]
            best_peak_time_sec = float(best_peak["peak_time_sec"])
            best_peak_probability = float(best_peak["peak_probability"])

        real_reasons: list[str] = []
        if int(cluster["marker_count"]) >= 2:
            real_reasons.append("marker_count>=2")
        if jaw_event_count >= 1:
            real_reasons.append("jaw_peak_in_support_window")
        if np.isfinite(max_clench_probability) and max_clench_probability >= PART1_LIKELY_REAL_PROBABILITY_THRESHOLD:
            real_reasons.append(f"maxP={max_clench_probability:.3f}>=0.70")
        likely_real_cluster = "yes" if real_reasons else "no"

        jaw_like_peaks = support_peaks[
            support_peaks["peak_probability"] >= PART1_PEAK_PROBABILITY_THRESHOLD
        ].copy()
        if not jaw_like_peaks.empty or (
            np.isfinite(max_clench_probability) and max_clench_probability >= PART1_LIKELY_JAW_TRACE_THRESHOLD
        ):
            jaw_vs_nonjaw_label = "likely jaw"
        elif likely_real_cluster == "yes" and support_peaks.empty and (
            np.isfinite(max_clench_probability) and max_clench_probability < NON_JAW_PROBABILITY_CEILING
        ):
            jaw_vs_nonjaw_label = "likely non-jaw"
        else:
            jaw_vs_nonjaw_label = "unknown"

        qualifying_peaks = support_peaks[
            (support_peaks["peak_probability"] >= PART1_PEAK_PROBABILITY_THRESHOLD)
            & (support_peaks["peak_prominence"] >= PART1_PEAK_PROMINENCE_THRESHOLD)
        ].copy().sort_values("peak_time_sec")
        clench_count_guess = "unknown"
        clench_count_basis = ""
        if qualifying_peaks.empty:
            clench_count_basis = "0 qualifying peaks"
        else:
            peak_times = qualifying_peaks["peak_time_sec"].to_numpy(dtype=float)
            peak_diffs = np.diff(peak_times)
            if np.any(peak_diffs < PART1_PEAK_MIN_SEPARATION_SEC):
                clench_count_basis = (
                    f"ambiguous: qualifying peaks closer than {PART1_PEAK_MIN_SEPARATION_SEC:.2f} s"
                )
            elif len(qualifying_peaks) == 1:
                peak = qualifying_peaks.iloc[0]
                clench_count_guess = "1"
                clench_count_basis = f"1 peak >=0.70 (P={float(peak['peak_probability']):.3f})"
            elif len(qualifying_peaks) == 2:
                clench_count_guess = "2"
                clench_count_basis = (
                    f"2 peaks >=0.70 separated by {float(peak_diffs[0]):.2f} s"
                )
            else:
                clench_count_basis = f"{len(qualifying_peaks)} qualifying peaks"

        notes = (
            "; ".join(real_reasons)
            if real_reasons
            else (
                "single-marker cluster with no Part 1 jaw peak and low clench probability"
                if int(cluster["marker_count"]) == 1
                else "no conservative real-cluster signal cleared threshold"
            )
        )

        rows.append(
            {
                "cluster_id": cluster_id,
                "block_id": int(cluster["block_id"]),
                "start_time_sec": start_time_sec,
                "end_time_sec": end_time_sec,
                "duration_sec": end_time_sec - start_time_sec,
                "marker_count": int(cluster["marker_count"]),
                "marker_values_list": cluster["original_marker_values"],
                "max_clench_probability": max_clench_probability,
                "jaw_event_count": jaw_event_count,
                "best_jaw_peak_time_sec": best_peak_time_sec,
                "best_jaw_peak_probability": best_peak_probability,
                "likely_real_cluster": likely_real_cluster,
                "jaw_vs_nonjaw_label": jaw_vs_nonjaw_label,
                "clench_count_guess": clench_count_guess,
                "clench_count_basis": clench_count_basis,
                "sequence_position_suggestion": "unknown",
                "sequence_suggestion_basis": "no confident block-level fit was accepted",
                "notes": notes,
            }
        )

    interpretation_df = pd.DataFrame(rows)
    if interpretation_df.empty:
        return interpretation_df, last_marker_time_sec

    for block_id, block_df in interpretation_df.groupby("block_id", sort=False):
        real_block_df = (
            block_df[block_df["likely_real_cluster"] == "yes"]
            .sort_values(["start_time_sec", "cluster_id"])
            .reset_index()
        )
        block_indices = block_df.index.tolist()
        non_real_indices = block_df[block_df["likely_real_cluster"] != "yes"].index.tolist()
        if non_real_indices:
            interpretation_df.loc[non_real_indices, "sequence_suggestion_basis"] = (
                "cluster was not considered real enough for Part 1 sequence fitting"
            )

        if len(real_block_df) < 4:
            reason = f"block has only {len(real_block_df)} real clusters (<4)"
            interpretation_df.loc[block_indices, "sequence_suggestion_basis"] = reason
            if non_real_indices:
                interpretation_df.loc[non_real_indices, "sequence_suggestion_basis"] = (
                    "cluster was not considered real enough for Part 1 sequence fitting"
                )
            continue

        scored_offsets: list[tuple[int, float, list[str]]] = []
        for offset in range(len(PHASE_LABELS)):
            assignments = [PHASE_LABELS[(position + offset) % len(PHASE_LABELS)] for position in range(len(real_block_df))]
            score = 0.0
            for phase_label, label in zip(assignments, real_block_df["jaw_vs_nonjaw_label"], strict=True):
                score += _part1_sequence_score(phase_label, str(label))
            scored_offsets.append((offset, score, assignments))

        scored_offsets.sort(key=lambda item: item[1], reverse=True)
        best_offset, best_score, best_assignments = scored_offsets[0]
        runner_up_score = scored_offsets[1][1] if len(scored_offsets) > 1 else float("-inf")
        score_gap = float(best_score - runner_up_score) if np.isfinite(runner_up_score) else float("inf")

        consistency_ok = True
        for phase_label, label in zip(best_assignments, real_block_df["jaw_vs_nonjaw_label"], strict=True):
            if phase_label in {"J1", "J2"} and label != "likely jaw":
                consistency_ok = False
                break
            if phase_label in {"L1", "R1", "L2", "R2"} and label != "likely non-jaw":
                consistency_ok = False
                break

        if score_gap < 3.0:
            reason = f"best LRJ offset gap was only {score_gap:.1f} (<3.0)"
            interpretation_df.loc[block_indices, "sequence_suggestion_basis"] = reason
            if non_real_indices:
                interpretation_df.loc[non_real_indices, "sequence_suggestion_basis"] = (
                    "cluster was not considered real enough for Part 1 sequence fitting"
                )
            continue
        if not consistency_ok:
            reason = "best LRJ offset conflicted with jaw/non-jaw labels"
            interpretation_df.loc[block_indices, "sequence_suggestion_basis"] = reason
            if non_real_indices:
                interpretation_df.loc[non_real_indices, "sequence_suggestion_basis"] = (
                    "cluster was not considered real enough for Part 1 sequence fitting"
                )
            continue

        accepted_reason = f"accepted LRJ offset {best_offset} (score {best_score:.1f}, gap {score_gap:.1f})"
        interpretation_df.loc[block_indices, "sequence_suggestion_basis"] = accepted_reason
        if non_real_indices:
            interpretation_df.loc[non_real_indices, "sequence_suggestion_basis"] = (
                "cluster was not considered real enough for Part 1 sequence fitting"
            )
        for (_, real_row), assignment in zip(real_block_df.iterrows(), best_assignments, strict=True):
            interpretation_df.loc[int(real_row["index"]), "sequence_position_suggestion"] = assignment
            interpretation_df.loc[int(real_row["index"]), "sequence_suggestion_basis"] = (
                f"{accepted_reason}; cluster order in block mapped to {assignment}"
            )

    ordered_columns = [
        "cluster_id",
        "start_time_sec",
        "end_time_sec",
        "duration_sec",
        "marker_count",
        "marker_values_list",
        "max_clench_probability",
        "jaw_event_count",
        "best_jaw_peak_time_sec",
        "best_jaw_peak_probability",
        "likely_real_cluster",
        "jaw_vs_nonjaw_label",
        "clench_count_guess",
        "clench_count_basis",
        "sequence_position_suggestion",
        "sequence_suggestion_basis",
        "notes",
    ]
    return interpretation_df[ordered_columns].copy(), last_marker_time_sec


def _write_debug_plot(
    output_path: Path,
    score_trace_df: pd.DataFrame,
    marker_audit_df: pd.DataFrame,
    jaw_events_df: pd.DataFrame,
    cluster_spans_df: pd.DataFrame | None = None,
    x_min_sec: float | None = None,
    x_max_sec: float | None = None,
    title: str = "LRJ debug timeline: markers and jaw probability",
    show_tail: bool = True,
) -> str | None:
    mpl_config_dir = output_path.parent / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir.resolve()))
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return "matplotlib is unavailable; skipped lrj_debug_plot.png"

    fig, ax = plt.subplots(figsize=(14, 5))
    if cluster_spans_df is not None and not cluster_spans_df.empty:
        for _, row in cluster_spans_df.iterrows():
            ax.axvspan(
                float(row["start_time_sec"]),
                float(row["end_time_sec"]),
                color="tab:green",
                alpha=0.08,
                zorder=0,
            )
    ax.plot(
        score_trace_df["center_time_sec"],
        score_trace_df["prob_binary_CLENCH"],
        color="tab:blue",
        linewidth=1.25,
        label="P(clench)",
    )

    for event_time in marker_audit_df["time_sec"].tolist():
        ax.axvline(event_time, color="0.75", linewidth=0.8, alpha=0.9)

    if not jaw_events_df.empty:
        aligned = jaw_events_df if not show_tail else jaw_events_df[jaw_events_df["cluster_anchor_status"] == "cluster_aligned"]
        tail = jaw_events_df[jaw_events_df["cluster_anchor_status"] == "post_marker_tail"] if show_tail else jaw_events_df.iloc[0:0]
        ax.scatter(
            aligned["peak_time_sec"],
            aligned["peak_probability"],
            color="tab:red",
            s=28,
            label="Jaw peaks",
            zorder=3,
        )
        if not tail.empty:
            ax.scatter(
                tail["peak_time_sec"],
                tail["peak_probability"],
                color="tab:orange",
                s=28,
                label="Post-marker tail peaks",
                zorder=3,
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    if x_min_sec is not None or x_max_sec is not None:
        ax.set_xlim(left=x_min_sec, right=x_max_sec)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return None


def _write_trial_review_plot(
    output_path: Path,
    score_trace_df: pd.DataFrame,
    marker_audit_df: pd.DataFrame,
    trial_summary_df: pd.DataFrame,
    trial_peaks_df: pd.DataFrame,
) -> str | None:
    mpl_config_dir = output_path.parent / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir.resolve()))
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return "matplotlib is unavailable; skipped trial_review.png"

    if marker_audit_df.empty:
        return "marker_audit.csv had no collapsed markers; skipped trial_review.png"

    last_marker_time_sec = _part1_last_marker_time_sec(marker_audit_df)
    plot_trace = score_trace_df[score_trace_df["center_time_sec"] <= last_marker_time_sec + 1e-9].copy()

    fig, ax = plt.subplots(figsize=(14, 5))
    for _, trial in trial_summary_df.iterrows():
        ax.axvspan(
            float(trial["start_time_sec"]),
            float(trial["end_time_sec"]),
            color="tab:green",
            alpha=0.10,
            zorder=0,
        )
    ax.plot(
        plot_trace["center_time_sec"],
        plot_trace["prob_binary_CLENCH"],
        color="tab:blue",
        linewidth=1.25,
        label="P(clench)",
    )
    for event_time in marker_audit_df["time_sec"].tolist():
        ax.axvline(event_time, color="0.75", linewidth=0.8, alpha=0.9)
    if not trial_peaks_df.empty:
        ax.scatter(
            trial_peaks_df["peak_time_sec"],
            trial_peaks_df["peak_probability"],
            color="tab:red",
            s=28,
            label="Trial peaks",
            zorder=3,
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Probability")
    ax.set_title("LRJ trial review: marker-pair windows and clench probability")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(left=0.0, right=last_marker_time_sec)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return None


def _format_summary_lines(
    csv_path: Path,
    output_dir: Path,
    marker_column_note: str,
    marker_audit_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
    jaw_events_df: pd.DataFrame,
    score_trace_df: pd.DataFrame,
    estimated_sampling: dict[str, Any],
    selected_channels: list[str],
    channel_quality_df: pd.DataFrame,
) -> list[str]:
    lines = [
        f"LRJ recovery summary for {csv_path.name}",
        f"  Output directory: {output_dir}",
        f"  Fixed reconstruction rate: 250.0 Hz",
        f"  Timestamp-derived rate note: {estimated_sampling['fs_used_hz']:.3f} Hz ({' | '.join(estimated_sampling['notes'])})",
        f"  Marker column: {marker_column_note}",
        f"  Marker events: {len(marker_audit_df)}",
        f"  Marker clusters: {len(cluster_summary_df)}",
        f"  Jaw peak candidates: {len(jaw_events_df)}",
        f"  Trigger-positive windows: {int(score_trace_df['trigger_click'].sum())}",
        f"  Artifact channels: {', '.join(selected_channels)}",
    ]

    selected_quality = channel_quality_df[channel_quality_df["channel"].isin(selected_channels)]
    questionable = selected_quality[selected_quality["status"] != "safe"]
    if not questionable.empty:
        lines.append("  Channel warnings:")
        for _, row in questionable.iterrows():
            lines.append(f"    - {row['channel']}: {row['status']} ({row['reasons']})")

    lines.append("  Cluster suggestions:")
    for _, row in cluster_summary_df.iterrows():
        lines.append(
            "    - "
            f"cluster {int(row['cluster_id'])} [{row['cluster_start_time_sec']:.3f}s, {row['cluster_end_time_sec']:.3f}s] "
            f"{row['suggested_label']} | markers={int(row['marker_count'])} "
            f"| maxP={row['max_clench_probability']:.3f}"
    )
    return lines


def _format_part1_summary_lines(
    interpretation_df: pd.DataFrame,
    last_marker_time_sec: float,
) -> list[str]:
    if interpretation_df.empty:
        return [
            f"Part 1 summary (0.0 s to {last_marker_time_sec:.3f} s)",
            "  No Part 1 clusters were available for interpretation.",
        ]

    real_count = int((interpretation_df["likely_real_cluster"] == "yes").sum())
    likely_jaw_count = int((interpretation_df["jaw_vs_nonjaw_label"] == "likely jaw").sum())
    likely_non_jaw_count = int((interpretation_df["jaw_vs_nonjaw_label"] == "likely non-jaw").sum())
    one_clench_count = int((interpretation_df["clench_count_guess"] == "1").sum())
    two_clench_count = int((interpretation_df["clench_count_guess"] == "2").sum())
    unknown_clench_count = int((interpretation_df["clench_count_guess"] == "unknown").sum())
    sequence_hints = interpretation_df[
        interpretation_df["sequence_position_suggestion"].astype(str) != "unknown"
    ].copy()

    lines = [
        f"Part 1 summary (0.0 s to {last_marker_time_sec:.3f} s)",
        f"  Real clusters: {real_count}",
        f"  Likely jaw clusters: {likely_jaw_count}",
        f"  Likely non-jaw clusters: {likely_non_jaw_count}",
        f"  Clench-count guesses: 1={one_clench_count}, 2={two_clench_count}, unknown={unknown_clench_count}",
    ]
    if sequence_hints.empty:
        lines.append("  Confident LRJ sequence hints: none")
    else:
        lines.append("  Confident LRJ sequence hints:")
        for _, row in sequence_hints.iterrows():
            lines.append(
                f"    - cluster {int(row['cluster_id'])}: {row['sequence_position_suggestion']} "
                f"({row['sequence_suggestion_basis']})"
            )
    return lines


def _format_pairs_summary_lines(
    trial_summary_df: pd.DataFrame,
    trial_stats: dict[str, int],
) -> list[str]:
    if trial_summary_df.empty:
        return [
            "Marker-pair trial summary",
            f"  Trial windows: 0 (candidate consecutive pairs={trial_stats['candidate_pairs']}, duration rejected={trial_stats['duration_rejected_pairs']})",
        ]

    likely_jaw_count = int((trial_summary_df["jaw_label"] == "likely jaw").sum())
    non_jaw_count = int((trial_summary_df["jaw_label"] == "non-jaw").sum())
    unknown_count = int((trial_summary_df["jaw_label"] == "unknown").sum())
    one_peak_count = int((trial_summary_df["peak_count"] == 1).sum())
    two_peak_count = int((trial_summary_df["peak_count"] == 2).sum())
    more_than_two_peak_count = int((trial_summary_df["peak_count"] > 2).sum())

    return [
        "Marker-pair trial summary",
        f"  Trial windows: {len(trial_summary_df)} (candidate consecutive pairs={trial_stats['candidate_pairs']}, duration rejected={trial_stats['duration_rejected_pairs']})",
        f"  Likely jaw trials: {likely_jaw_count}",
        f"  Non-jaw trials: {non_jaw_count}",
        f"  Unknown trials: {unknown_count}",
        f"  Peak-count hints: 1={one_peak_count}, 2={two_peak_count}, >2={more_than_two_peak_count}",
    ]


def run_recovery(
    csv_path: Path,
    artifact_path: Path,
    output_dir: Path,
    fs_hz: float,
    cluster_gap_sec: float,
    make_plot: bool,
    part1: bool = False,
    pairs: bool = False,
) -> dict[str, Any]:
    if not csv_path.exists():
        _fail(f"CSV not found: {csv_path}")
    if not artifact_path.exists():
        _fail(f"Saved jaw artifact not found: {artifact_path}")

    output_dir = ensure_output_dir(output_dir)
    artifact = _load_artifact(artifact_path.resolve())
    model_bundle = artifact["model_bundle"]
    selected_channels = list(model_bundle["selected_channels"])

    raw_df = load_openbci_csv(csv_path.resolve())
    estimated_sampling = estimate_sampling(raw_df)
    marker_column_name, marker_series, marker_column_note = _detect_marker_column(raw_df)
    if marker_column_name != "Marker":
        raw_df = raw_df.copy()
        raw_df["Marker"] = marker_series
    marker_events = _collapse_marker_events(marker_series)
    marker_audit_df = _build_marker_audit(
        marker_events,
        fs_hz=fs_hz,
        cluster_gap_sec=cluster_gap_sec,
        block_gap_sec=LARGE_GAP_BLOCK_SEC,
    )

    channel_quality_df = compute_channel_quality(raw_df)
    sample_frame, signal_columns = _build_recovery_sample_frame(
        raw_df,
        csv_path=csv_path,
        selected_channels=selected_channels,
        fs_hz=fs_hz,
        smoothing_sec=float(artifact["event_config"]["smoothing_sec"]),
    )
    score_trace_df = _score_recovery_windows(
        sample_frame=sample_frame,
        signal_columns=signal_columns,
        model_bundle=model_bundle,
        artifact=artifact,
        fs_hz=fs_hz,
    )
    trigger_config = JawClickTriggerConfig(**artifact["trigger_config"])
    cluster_index_df = _build_cluster_index(marker_audit_df)
    jaw_events_df = _build_jaw_events(
        score_trace_df=score_trace_df,
        marker_audit_df=marker_audit_df,
        cluster_index_df=cluster_index_df,
        trigger_config=trigger_config,
        fs_hz=fs_hz,
    )
    cluster_summary_df = _build_cluster_summary(
        marker_audit_df=marker_audit_df,
        score_trace_df=score_trace_df,
        jaw_events_df=jaw_events_df,
        trigger_config=trigger_config,
    )
    jaw_events_df = _attach_sequence_suggestions(jaw_events_df, cluster_summary_df)

    marker_audit_path = output_dir / "marker_audit.csv"
    score_trace_path = output_dir / "score_trace.csv"
    jaw_events_path = output_dir / "jaw_events.csv"
    cluster_summary_path = output_dir / "cluster_summary.csv"

    marker_audit_df.to_csv(marker_audit_path, index=False)
    score_trace_df.to_csv(score_trace_path, index=False)
    jaw_events_df.to_csv(jaw_events_path, index=False)
    cluster_summary_df.to_csv(cluster_summary_path, index=False)

    plot_warning = None
    if make_plot:
        plot_warning = _write_debug_plot(
            output_dir / "lrj_debug_plot.png",
            score_trace_df,
            marker_audit_df,
            jaw_events_df,
        )

    part1_artifacts: dict[str, Any] = {}
    if part1:
        part1_interpretation_df, last_marker_time_sec = _build_part1_cluster_interpretation(
            marker_audit_df=marker_audit_df,
            score_trace_df=score_trace_df,
            jaw_events_df=jaw_events_df,
            cluster_summary_df=cluster_summary_df,
        )
        part1_interpretation_path = output_dir / "part1_cluster_interpretation.csv"
        part1_interpretation_df.to_csv(part1_interpretation_path, index=False)

        part1_plot_warning = None
        if make_plot:
            part1_score_trace_df = score_trace_df[
                score_trace_df["center_time_sec"] <= last_marker_time_sec + 1e-9
            ].copy()
            part1_marker_audit_df = marker_audit_df[
                marker_audit_df["time_sec"] <= last_marker_time_sec + 1e-9
            ].copy()
            part1_jaw_events_df = jaw_events_df[
                jaw_events_df["peak_time_sec"] <= last_marker_time_sec + 1e-9
            ].copy()
            part1_plot_warning = _write_debug_plot(
                output_dir / "part1_cluster_review.png",
                part1_score_trace_df,
                part1_marker_audit_df,
                part1_jaw_events_df,
                cluster_spans_df=part1_interpretation_df,
                x_min_sec=0.0,
                x_max_sec=last_marker_time_sec,
                title="Part 1 cluster review: markers, clusters, and jaw probability",
                show_tail=False,
            )
        part1_artifacts = {
            "part1_cluster_interpretation_path": part1_interpretation_path,
            "part1_plot_warning": part1_plot_warning,
            "part1_last_marker_time_sec": last_marker_time_sec,
            "part1_interpretation_df": part1_interpretation_df,
        }

    pair_artifacts: dict[str, Any] = {}
    if pairs:
        trial_summary_df, trial_peaks_df, trial_stats = _build_trial_summary(
            marker_audit_path=marker_audit_path,
            score_trace_df=score_trace_df,
        )
        trial_summary_path = output_dir / "trial_summary.csv"
        trial_summary_df.to_csv(trial_summary_path, index=False)
        trial_plot_warning = None
        if make_plot and not trial_summary_df.empty:
            trial_plot_warning = _write_trial_review_plot(
                output_dir / "trial_review.png",
                score_trace_df=score_trace_df,
                marker_audit_df=marker_audit_df,
                trial_summary_df=trial_summary_df,
                trial_peaks_df=trial_peaks_df,
            )
        pair_artifacts = {
            "trial_summary_path": trial_summary_path,
            "trial_plot_warning": trial_plot_warning,
            "trial_summary_df": trial_summary_df,
            "trial_stats": trial_stats,
        }

    summary_lines = _format_summary_lines(
        csv_path=csv_path,
        output_dir=output_dir,
        marker_column_note=marker_column_note,
        marker_audit_df=marker_audit_df,
        cluster_summary_df=cluster_summary_df,
        jaw_events_df=jaw_events_df,
        score_trace_df=score_trace_df,
        estimated_sampling=estimated_sampling,
        selected_channels=selected_channels,
        channel_quality_df=channel_quality_df,
    )
    if plot_warning:
        summary_lines.append(f"  Plot warning: {plot_warning}")
    if part1:
        summary_lines.append("")
        summary_lines.extend(
            _format_part1_summary_lines(
                interpretation_df=part1_artifacts["part1_interpretation_df"],
                last_marker_time_sec=float(part1_artifacts["part1_last_marker_time_sec"]),
            )
        )
        if part1_artifacts.get("part1_plot_warning"):
            summary_lines.append(f"  Part 1 plot warning: {part1_artifacts['part1_plot_warning']}")
    if pairs:
        summary_lines.append("")
        summary_lines.extend(
            _format_pairs_summary_lines(
                trial_summary_df=pair_artifacts["trial_summary_df"],
                trial_stats=pair_artifacts["trial_stats"],
            )
        )
        if pair_artifacts.get("trial_plot_warning"):
            summary_lines.append(f"  Trial plot warning: {pair_artifacts['trial_plot_warning']}")

    print("\n".join(summary_lines))

    return {
        "marker_audit_path": marker_audit_path,
        "score_trace_path": score_trace_path,
        "jaw_events_path": jaw_events_path,
        "cluster_summary_path": cluster_summary_path,
        "plot_warning": plot_warning,
        **part1_artifacts,
        **pair_artifacts,
    }


def main() -> None:
    args = parse_args()
    run_recovery(
        csv_path=args.csv,
        artifact_path=args.artifact,
        output_dir=args.output_dir,
        fs_hz=float(args.fs),
        cluster_gap_sec=float(args.cluster_gap_sec),
        make_plot=not args.no_plot,
        part1=bool(args.part1),
        pairs=bool(args.pairs),
    )


if __name__ == "__main__":
    main()
