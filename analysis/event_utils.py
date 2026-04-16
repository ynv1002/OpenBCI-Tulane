from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .utils import LABEL_BASELINE, LABEL_REST, load_openbci_csv, preprocess_session_signals, select_model_channels, split_train_test_audits


EVENT_INACTIVE = "INACTIVE"
EVENT_ONSET = "ONSET"
EVENT_ACTIVE = "ACTIVE"
EVENT_OFFSET = "OFFSET"
EVENT_LABELS_4STATE = [EVENT_INACTIVE, EVENT_ONSET, EVENT_ACTIVE, EVENT_OFFSET]


@dataclass(frozen=True)
class JawEventConfig:
    smoothing_sec: float = 0.25
    onset_duration_sec: float = 0.20
    offset_duration_sec: float = 0.20
    minimum_active_duration_sec: float = 0.15
    minimum_peak_distance_sec: float = 0.35
    inactive_quantile: float = 0.90
    active_quantile: float = 0.75
    threshold_mix: float = 0.35
    release_threshold_mix: float = 0.15
    minimum_reference_gap_uv: float = 0.05
    peak_prominence_scale: float = 0.25
    minimum_peak_prominence_uv: float = 0.05
    manual_active_threshold_uv: float | None = None
    manual_release_threshold_uv: float | None = None
    repeated_fallback_mode: str = "coarse_single_event"


@dataclass(frozen=True)
class EventWindowConfig:
    window_sec: float = 0.20
    overlap: float = 0.50


def _center_label(values: Sequence[str]) -> str:
    return str(values[len(values) // 2])


def _linear_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    x = x - x.mean()
    y = values - float(np.mean(values))
    denom = float(np.sum(x * x))
    if denom == 0.0:
        return 0.0
    return float(np.sum(x * y) / denom)


def _event_feature_names(channel_columns: Sequence[str]) -> List[str]:
    names: List[str] = []
    base = ["mean", "std", "rms", "ptp", "var", "energy", "abs_mean", "slope"]
    for channel in channel_columns:
        names.extend(f"{channel}_{name}" for name in base)
    names.extend(
        [
            "aggregate_rms_mean",
            "aggregate_rms_std",
            "aggregate_rms_max",
            "aggregate_rms_delta",
            "aggregate_rms_slope",
            "aggregate_rms_smooth_mean",
            "aggregate_rms_smooth_std",
            "aggregate_rms_smooth_max",
            "aggregate_rms_smooth_delta",
            "aggregate_rms_smooth_slope",
        ]
    )
    return names


def _split_event_interval(
    start_sample: int,
    end_sample: int,
    onset_samples: int,
    offset_samples: int,
    minimum_active_samples: int,
) -> List[Tuple[str, int, int]]:
    total = end_sample - start_sample + 1
    if total <= 0:
        return []

    onset_len = min(max(1, onset_samples), max(1, total // 3))
    offset_len = min(max(1, offset_samples), max(1, total // 3))

    while onset_len + offset_len + minimum_active_samples > total and (onset_len > 1 or offset_len > 1):
        if onset_len >= offset_len and onset_len > 1:
            onset_len -= 1
        elif offset_len > 1:
            offset_len -= 1
        else:
            break

    active_len = total - onset_len - offset_len
    if active_len < 0:
        active_len = 0
    if active_len == 0 and total >= 2:
        onset_len = max(1, total // 2)
        offset_len = total - onset_len

    onset_end = start_sample + onset_len - 1
    offset_start = end_sample - offset_len + 1
    intervals: List[Tuple[str, int, int]] = [(EVENT_ONSET, start_sample, onset_end)]
    if offset_start > onset_end + 1:
        intervals.append((EVENT_ACTIVE, onset_end + 1, offset_start - 1))
    if offset_start <= end_sample:
        intervals.append((EVENT_OFFSET, offset_start, end_sample))
    return intervals


def _detect_repeated_peak_intervals(
    envelope_segment: np.ndarray,
    fs_hz: float,
    active_threshold: float,
    release_threshold: float,
    prominence_threshold: float,
    min_peak_distance_sec: float,
    minimum_active_duration_sec: float,
) -> List[Dict[str, Any]]:
    min_peak_distance_samples = max(1, int(round(min_peak_distance_sec * fs_hz)))
    minimum_active_samples = max(1, int(round(minimum_active_duration_sec * fs_hz)))

    peaks, properties = find_peaks(
        envelope_segment,
        height=active_threshold,
        prominence=prominence_threshold,
        distance=min_peak_distance_samples,
    )
    if len(peaks) == 0:
        return []

    intervals: List[Dict[str, Any]] = []
    for peak_index, peak_sample in enumerate(peaks):
        left_limit = 0 if peak_index == 0 else int((peaks[peak_index - 1] + peak_sample) // 2) + 1
        right_limit = (
            len(envelope_segment) - 1
            if peak_index == len(peaks) - 1
            else int((peak_sample + peaks[peak_index + 1]) // 2) - 1
        )

        start = int(peak_sample)
        while start > left_limit and envelope_segment[start - 1] >= release_threshold:
            start -= 1
        end = int(peak_sample)
        while end < right_limit and envelope_segment[end + 1] >= release_threshold:
            end += 1

        if end - start + 1 < minimum_active_samples:
            pad = (minimum_active_samples - (end - start + 1)) // 2 + 1
            start = max(left_limit, start - pad)
            end = min(right_limit, end + pad)

        if end - start + 1 >= minimum_active_samples:
            intervals.append(
                {
                    "peak_sample_local": int(peak_sample),
                    "start_sample_local": int(start),
                    "end_sample_local": int(end),
                    "peak_height": float(properties["peak_heights"][peak_index]),
                    "peak_prominence": float(properties["prominences"][peak_index]),
                }
            )

    return intervals


def _coarse_segment_arrays(audit: Dict[str, Any], total_samples: int) -> Dict[str, np.ndarray]:
    coarse_label = np.full(total_samples, EVENT_INACTIVE, dtype=object)
    coarse_segment_kind = np.full(total_samples, "", dtype=object)
    coarse_segment_index = np.full(total_samples, -1, dtype=int)
    coarse_segment_context = np.full(total_samples, "", dtype=object)
    for segment in audit["segments"]:
        start = int(segment["start_sample"])
        end = int(segment["end_sample"]) + 1
        coarse_label[start:end] = segment["label"]
        coarse_segment_kind[start:end] = segment["segment_kind"]
        coarse_segment_index[start:end] = int(segment["segment_index"])
        coarse_segment_context[start:end] = str(segment["segment_context"] or "")
    return {
        "coarse_label": coarse_label,
        "coarse_segment_kind": coarse_segment_kind,
        "coarse_segment_index": coarse_segment_index,
        "coarse_segment_context": coarse_segment_context,
    }


def build_jaw_event_labels_for_audit(
    audit: Dict[str, Any],
    channel_columns: Sequence[str],
    config: JawEventConfig,
) -> Dict[str, Any]:
    df = load_openbci_csv(Path(audit["file_path"]))
    fs_hz = float(audit["sampling"]["fs_used_hz"])
    processed = preprocess_session_signals(df, "jaw", channel_columns, fs_hz)
    signal_columns = [f"signal_{column}" for column in channel_columns]
    processed = processed.rename(columns={column: f"signal_{column}" for column in channel_columns})

    sample_frame = pd.DataFrame(
        {
            "family": audit["family"],
            "filename": audit["filename"],
            "file_path": str(audit["file_path"]),
            "parsed_date": audit["parsed_date"],
            "session_rank": audit["session_rank"],
            "sample_row": np.arange(len(df), dtype=int),
            "sample_index_raw": df["Sample_Index"].to_numpy(dtype=float),
            "time_sec": np.arange(len(df), dtype=float) / fs_hz,
        }
    )
    for column in signal_columns:
        sample_frame[column] = processed[column].to_numpy(dtype=float)

    aggregate_rms = np.sqrt(np.mean(processed.to_numpy(dtype=float) ** 2, axis=1))
    smoothing_samples = max(1, int(round(config.smoothing_sec * fs_hz)))
    aggregate_rms_smooth = (
        pd.Series(aggregate_rms).rolling(smoothing_samples, center=True, min_periods=1).mean().to_numpy()
    )
    sample_frame["aggregate_rms"] = aggregate_rms
    sample_frame["aggregate_rms_smooth"] = aggregate_rms_smooth

    coarse_arrays = _coarse_segment_arrays(audit, len(df))
    for name, values in coarse_arrays.items():
        sample_frame[name] = values

    inactive_mask = sample_frame["coarse_label"].isin([LABEL_BASELINE, LABEL_REST]).to_numpy()
    active_mask = ~inactive_mask
    inactive_reference = float(np.quantile(aggregate_rms_smooth[inactive_mask], config.inactive_quantile))
    active_reference = float(np.quantile(aggregate_rms_smooth[active_mask], config.active_quantile))
    reference_gap = max(active_reference - inactive_reference, config.minimum_reference_gap_uv)

    active_threshold = (
        float(config.manual_active_threshold_uv)
        if config.manual_active_threshold_uv is not None
        else inactive_reference + config.threshold_mix * reference_gap
    )
    release_threshold = (
        float(config.manual_release_threshold_uv)
        if config.manual_release_threshold_uv is not None
        else inactive_reference + config.release_threshold_mix * reference_gap
    )
    if release_threshold >= active_threshold:
        release_threshold = inactive_reference + 0.5 * (active_threshold - inactive_reference)
    peak_prominence = max(config.minimum_peak_prominence_uv, config.peak_prominence_scale * reference_gap)

    sample_frame["active_threshold"] = active_threshold
    sample_frame["release_threshold"] = release_threshold
    sample_frame["event_label"] = EVENT_INACTIVE
    sample_frame["event_id"] = -1
    sample_frame["event_source"] = "inactive_default"
    sample_frame["event_detection_mode"] = "inactive_default"

    onset_samples = max(1, int(round(config.onset_duration_sec * fs_hz)))
    offset_samples = max(1, int(round(config.offset_duration_sec * fs_hz)))
    minimum_active_samples = max(1, int(round(config.minimum_active_duration_sec * fs_hz)))

    event_interval_rows: List[Dict[str, Any]] = []
    coarse_segment_rows: List[Dict[str, Any]] = []
    event_counter = 0
    split_role = "train" if audit["session_rank"] <= 2 else "test"

    for segment in audit["segments"]:
        segment_row = {
            "family": audit["family"],
            "filename": audit["filename"],
            "file_path": str(audit["file_path"]),
            "parsed_date": audit["parsed_date"],
            "session_rank": audit["session_rank"],
            "split_role": split_role,
            "coarse_label": segment["label"],
            "coarse_segment_index": int(segment["segment_index"]),
            "coarse_segment_kind": segment["segment_kind"],
            "start_sample": int(segment["start_sample"]),
            "end_sample": int(segment["end_sample"]),
            "start_time_sec": float(segment["start_time_sec"]),
            "end_time_sec": float(segment["end_time_sec"]),
            "duration_sec": float(segment["duration_sec"]),
            "active_threshold": active_threshold,
            "release_threshold": release_threshold,
            "peak_prominence_threshold": peak_prominence,
            "detection_mode": "inactive_coarse_segment",
            "detected_peak_count": 0,
            "detected_event_count": 0,
            "fallback_used": False,
            "notes": "",
        }
        if segment["label"] in (LABEL_BASELINE, LABEL_REST):
            coarse_segment_rows.append(segment_row)
            continue

        local_start = int(segment["start_sample"])
        local_end = int(segment["end_sample"])
        event_source = "hold_marker_edges"
        event_intervals_local: List[Dict[str, Any]] = []

        if segment["label"] == "HOLD":
            state_intervals = _split_event_interval(
                local_start,
                local_end,
                onset_samples=onset_samples,
                offset_samples=offset_samples,
                minimum_active_samples=minimum_active_samples,
            )
            event_intervals_local.append(
                {
                    "event_id": event_counter,
                    "event_source": event_source,
                    "peak_sample": None,
                    "state_intervals": state_intervals,
                }
            )
            segment_row["detection_mode"] = "hold_marker_edges"
            segment_row["detected_event_count"] = 1
        else:
            envelope_segment = aggregate_rms_smooth[local_start : local_end + 1]
            detected_bouts = _detect_repeated_peak_intervals(
                envelope_segment=envelope_segment,
                fs_hz=fs_hz,
                active_threshold=active_threshold,
                release_threshold=release_threshold,
                prominence_threshold=peak_prominence,
                min_peak_distance_sec=config.minimum_peak_distance_sec,
                minimum_active_duration_sec=config.minimum_active_duration_sec,
            )
            if detected_bouts:
                segment_row["detection_mode"] = "repeated_peaks"
                segment_row["detected_peak_count"] = len(detected_bouts)
                segment_row["detected_event_count"] = len(detected_bouts)
                for bout_index, bout in enumerate(detected_bouts):
                    absolute_start = local_start + int(bout["start_sample_local"])
                    absolute_end = local_start + int(bout["end_sample_local"])
                    state_intervals = _split_event_interval(
                        absolute_start,
                        absolute_end,
                        onset_samples=onset_samples,
                        offset_samples=offset_samples,
                        minimum_active_samples=minimum_active_samples,
                    )
                    event_intervals_local.append(
                        {
                            "event_id": event_counter + bout_index,
                            "event_source": "repeated_peaks",
                            "peak_sample": local_start + int(bout["peak_sample_local"]),
                            "state_intervals": state_intervals,
                            "peak_height": bout["peak_height"],
                            "peak_prominence": bout["peak_prominence"],
                        }
                    )
            else:
                state_intervals = _split_event_interval(
                    local_start,
                    local_end,
                    onset_samples=onset_samples,
                    offset_samples=offset_samples,
                    minimum_active_samples=minimum_active_samples,
                )
                event_intervals_local.append(
                    {
                        "event_id": event_counter,
                        "event_source": "repeated_coarse_fallback",
                        "peak_sample": None,
                        "state_intervals": state_intervals,
                    }
                )
                segment_row["detection_mode"] = "repeated_coarse_fallback"
                segment_row["detected_event_count"] = 1
                segment_row["fallback_used"] = True
                segment_row["notes"] = "No repeated internal peaks passed the configured threshold; used one coarse fallback event."

        for local_event_index, event_interval in enumerate(event_intervals_local):
            event_id = int(event_interval["event_id"])
            for state_label, state_start, state_end in event_interval["state_intervals"]:
                if state_end < state_start:
                    continue
                sample_frame.loc[state_start : state_end, "event_label"] = state_label
                sample_frame.loc[state_start : state_end, "event_id"] = event_id
                sample_frame.loc[state_start : state_end, "event_source"] = event_interval["event_source"]
                sample_frame.loc[state_start : state_end, "event_detection_mode"] = segment_row["detection_mode"]
                event_interval_rows.append(
                    {
                        "family": audit["family"],
                        "filename": audit["filename"],
                        "file_path": str(audit["file_path"]),
                        "parsed_date": audit["parsed_date"],
                        "session_rank": audit["session_rank"],
                        "split_role": split_role,
                        "coarse_label": segment["label"],
                        "coarse_segment_index": int(segment["segment_index"]),
                        "event_id": event_id,
                        "event_index_within_segment": local_event_index,
                        "event_source": event_interval["event_source"],
                        "state_label": state_label,
                        "start_sample": int(state_start),
                        "end_sample": int(state_end),
                        "start_time_sec": float(state_start / fs_hz),
                        "end_time_sec": float(state_end / fs_hz),
                        "duration_sec": float((state_end - state_start + 1) / fs_hz),
                        "peak_sample": event_interval.get("peak_sample"),
                        "peak_height": event_interval.get("peak_height"),
                        "peak_prominence": event_interval.get("peak_prominence"),
                    }
                )
            if event_interval["state_intervals"]:
                event_counter = max(event_counter, event_id + 1)

        coarse_segment_rows.append(segment_row)

    sample_frame["split_role"] = split_role
    interval_df = pd.DataFrame(event_interval_rows)
    coarse_segment_df = pd.DataFrame(coarse_segment_rows)
    threshold_summary = {
        "inactive_reference_uv": inactive_reference,
        "active_reference_uv": active_reference,
        "reference_gap_uv": reference_gap,
        "active_threshold_uv": active_threshold,
        "release_threshold_uv": release_threshold,
        "peak_prominence_uv": peak_prominence,
        "config": asdict(config),
        "split_role": split_role,
    }

    return {
        "sample_frame": sample_frame,
        "interval_frame": interval_df,
        "coarse_segment_frame": coarse_segment_df,
        "threshold_summary": threshold_summary,
        "selected_channels": list(channel_columns),
    }


def build_jaw_event_label_bundle(
    audits: Sequence[Dict[str, Any]],
    config: JawEventConfig,
) -> Dict[str, Any]:
    train_audits, test_audit = split_train_test_audits(audits, "jaw")
    selected_channels, excluded_channels = select_model_channels(train_audits)

    session_bundles = []
    for audit in [*train_audits, test_audit]:
        session_bundle = build_jaw_event_labels_for_audit(audit, selected_channels, config)
        session_bundles.append({"audit": audit, **session_bundle})

    sample_frame = pd.concat([bundle["sample_frame"] for bundle in session_bundles], ignore_index=True)
    interval_frame = pd.concat([bundle["interval_frame"] for bundle in session_bundles], ignore_index=True)
    coarse_segment_frame = pd.concat(
        [bundle["coarse_segment_frame"] for bundle in session_bundles], ignore_index=True
    )
    threshold_frame = pd.DataFrame(
        [
            {
                "family": bundle["audit"]["family"],
                "filename": bundle["audit"]["filename"],
                "session_rank": bundle["audit"]["session_rank"],
                "parsed_date": bundle["audit"]["parsed_date"],
                **bundle["threshold_summary"],
            }
            for bundle in session_bundles
        ]
    )

    return {
        "session_bundles": session_bundles,
        "sample_frame": sample_frame,
        "interval_frame": interval_frame,
        "coarse_segment_frame": coarse_segment_frame,
        "threshold_frame": threshold_frame,
        "train_audits": train_audits,
        "test_audit": test_audit,
        "selected_channels": selected_channels,
        "excluded_channels": excluded_channels,
        "signal_columns": [f"signal_{column}" for column in selected_channels],
    }


def summarize_jaw_event_labels(bundle: Dict[str, Any]) -> Dict[str, Any]:
    sample_frame = bundle["sample_frame"]
    coarse_segment_frame = bundle["coarse_segment_frame"]
    repeated = coarse_segment_frame[coarse_segment_frame["coarse_label"] == "REPEATED"].copy()
    hold = coarse_segment_frame[coarse_segment_frame["coarse_label"] == "HOLD"].copy()

    event_counts = (
        sample_frame.groupby(["filename", "event_label"]).size().reset_index(name="sample_count")
    )
    summary = {
        "event_state_sample_counts": sample_frame["event_label"].value_counts().sort_index().to_dict(),
        "per_file_event_state_counts": event_counts.to_dict(orient="records"),
        "repeated_segment_count": int(len(repeated)),
        "repeated_fallback_count": int(repeated["fallback_used"].sum()) if len(repeated) else 0,
        "repeated_detected_event_count_total": int(repeated["detected_event_count"].sum()) if len(repeated) else 0,
        "repeated_detected_event_count_median": float(repeated["detected_event_count"].median())
        if len(repeated)
        else 0.0,
        "hold_segment_count": int(len(hold)),
    }
    return summary


def build_event_windows(
    sample_frame: pd.DataFrame,
    signal_columns: Sequence[str],
    window_config: EventWindowConfig,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    feature_names = _event_feature_names(signal_columns)

    for filename, session_df in sample_frame.groupby("filename", sort=False):
        session_df = session_df.reset_index(drop=True)
        if len(session_df) < 2:
            continue
        time_values = session_df["time_sec"].to_numpy(dtype=float)
        fs_hz = 1.0 / float(np.median(np.diff(time_values)))
        window_samples = max(1, int(round(window_config.window_sec * fs_hz)))
        hop_samples = max(1, int(round(window_samples * (1.0 - window_config.overlap))))
        if len(session_df) < window_samples:
            continue

        signal_matrix = session_df.loc[:, list(signal_columns)].to_numpy(dtype=float)
        env_matrix = session_df.loc[:, ["aggregate_rms", "aggregate_rms_smooth"]].to_numpy(dtype=float)
        label_values = session_df["event_label"].to_numpy()

        for start in range(0, len(session_df) - window_samples + 1, hop_samples):
            end = start + window_samples
            center = start + window_samples // 2
            row = {
                "filename": filename,
                "file_path": session_df.iloc[center]["file_path"],
                "parsed_date": session_df.iloc[center]["parsed_date"],
                "session_rank": int(session_df.iloc[center]["session_rank"]),
                "split_role": session_df.iloc[center]["split_role"],
                "event_label": str(label_values[center]),
                "coarse_label_center": str(session_df.iloc[center]["coarse_label"]),
                "center_time_sec": float(session_df.iloc[center]["time_sec"]),
                "start_sample": int(session_df.iloc[start]["sample_row"]),
                "end_sample": int(session_df.iloc[end - 1]["sample_row"]),
                "window_sec": window_config.window_sec,
                "overlap": window_config.overlap,
            }

            window = signal_matrix[start:end, :]
            for channel_index, channel_name in enumerate(signal_columns):
                values = window[:, channel_index]
                row[f"{channel_name}_mean"] = float(np.mean(values))
                row[f"{channel_name}_std"] = float(np.std(values))
                row[f"{channel_name}_rms"] = float(np.sqrt(np.mean(values ** 2)))
                row[f"{channel_name}_ptp"] = float(np.ptp(values))
                row[f"{channel_name}_var"] = float(np.var(values))
                row[f"{channel_name}_energy"] = float(np.mean(values ** 2))
                row[f"{channel_name}_abs_mean"] = float(np.mean(np.abs(values)))
                row[f"{channel_name}_slope"] = _linear_slope(values)

            env_window = env_matrix[start:end, :]
            aggregate_rms = env_window[:, 0]
            aggregate_rms_smooth = env_window[:, 1]
            row["aggregate_rms_mean"] = float(np.mean(aggregate_rms))
            row["aggregate_rms_std"] = float(np.std(aggregate_rms))
            row["aggregate_rms_max"] = float(np.max(aggregate_rms))
            row["aggregate_rms_delta"] = float(aggregate_rms[-1] - aggregate_rms[0])
            row["aggregate_rms_slope"] = _linear_slope(aggregate_rms)
            row["aggregate_rms_smooth_mean"] = float(np.mean(aggregate_rms_smooth))
            row["aggregate_rms_smooth_std"] = float(np.std(aggregate_rms_smooth))
            row["aggregate_rms_smooth_max"] = float(np.max(aggregate_rms_smooth))
            row["aggregate_rms_smooth_delta"] = float(aggregate_rms_smooth[-1] - aggregate_rms_smooth[0])
            row["aggregate_rms_smooth_slope"] = _linear_slope(aggregate_rms_smooth)
            rows.append(row)

    window_df = pd.DataFrame(rows)
    metadata_columns = [
        "filename",
        "file_path",
        "parsed_date",
        "session_rank",
        "split_role",
        "event_label",
        "coarse_label_center",
        "center_time_sec",
        "start_sample",
        "end_sample",
        "window_sec",
        "overlap",
    ]
    return {
        "window_frame": window_df,
        "feature_columns": [name for name in feature_names if name in window_df.columns],
        "metadata_columns": metadata_columns,
    }


def remap_event_labels(frame: pd.DataFrame, task_name: str) -> pd.DataFrame:
    out = frame.copy()
    if task_name == "jaw_4state":
        out["label"] = out["event_label"]
    elif task_name == "clench_vs_nonclench":
        out["label"] = out["event_label"].map(
            {
                EVENT_INACTIVE: "NON_CLENCH",
                EVENT_ONSET: "CLENCH",
                EVENT_ACTIVE: "CLENCH",
                EVENT_OFFSET: "CLENCH",
            }
        )
    elif task_name == "active_vs_other":
        out["label"] = out["event_label"].map(
            {
                EVENT_ACTIVE: "ACTIVE",
                EVENT_INACTIVE: "OTHER",
                EVENT_ONSET: "OTHER",
                EVENT_OFFSET: "OTHER",
            }
        )
    else:
        raise ValueError(f"Unknown task name: {task_name}")
    return out


def task_label_order(task_name: str) -> List[str]:
    if task_name == "jaw_4state":
        return EVENT_LABELS_4STATE
    if task_name == "clench_vs_nonclench":
        return ["NON_CLENCH", "CLENCH"]
    if task_name == "active_vs_other":
        return ["OTHER", "ACTIVE"]
    raise ValueError(f"Unknown task name: {task_name}")
