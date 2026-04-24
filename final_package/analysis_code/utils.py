from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from itertools import combinations
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, welch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_ROOT = PROJECT_ROOT / "analysis"
DEFAULT_OUTPUT_ROOT = ANALYSIS_ROOT / "outputs"
OPENBCI_RUNS_ROOT = PROJECT_ROOT.parent / "OPENBCI_runs"
YANIV_RUNS_ROOT = OPENBCI_RUNS_ROOT / "Yaniv"

DEFAULT_COUNT_TO_UV = 0.02235
DEFAULT_FS_FALLBACK = 250.0
DEFAULT_WINDOW_SEC = 2.0
DEFAULT_OVERLAP = 0.5
FEATURE_MODES = ("time_only", "spectral_only", "combined")

NOMINAL_RAIL = 187500.022352
RAIL_TOLERANCE = 5.0
QUESTIONABLE_RAIL_FRACTION = 0.005
UNSAFE_RAIL_FRACTION = 0.05

LABEL_BASELINE = "BASELINE"
LABEL_REST = "REST"

SESSION_NAME_RE = re.compile(r"^[A-Za-z]+-(\d{1,2})-(\d{1,2})-(\d{2})-\((\d+)\)\.csv$")


def _resolve_run_folder(preferred_subdir: str) -> Path:
    preferred = YANIV_RUNS_ROOT / preferred_subdir
    if preferred.exists():
        return preferred
    return YANIV_RUNS_ROOT


@dataclass(frozen=True)
class FamilySpec:
    key: str
    display_name: str
    folder: Path
    filename_prefix: str
    movement1_label: str
    movement2_label: str
    train_session_count: int


FAMILY_SPECS: Dict[str, FamilySpec] = {
    "left_right": FamilySpec(
        key="left_right",
        display_name="Left/Right hand squeeze",
        folder=_resolve_run_folder("EEG_LR"),
        filename_prefix="LR",
        movement1_label="LEFT",
        movement2_label="RIGHT",
        train_session_count=3,
    ),
    "jaw": FamilySpec(
        key="jaw",
        display_name="Jaw hold vs repeated clench",
        folder=_resolve_run_folder("EMG_JvsN"),
        filename_prefix="HR",
        movement1_label="HOLD",
        movement2_label="REPEATED",
        train_session_count=2,
    ),
}

CANONICAL_CHANNELS = [f"Channel_{idx}" for idx in range(1, 9)]
LABEL_COLUMNS = {
    "movement1": 1,
    "movement1_end": 2,
    "movement2": 3,
    "movement2_end": 4,
}
WINDOW_METADATA_COLUMNS = [
    "family",
    "filename",
    "file_path",
    "session_date",
    "session_rank",
    "split_role",
    "label",
    "segment_kind",
    "segment_source",
    "segment_context",
    "segment_index",
    "start_sample",
    "end_sample",
    "start_time_sec",
    "end_time_sec",
    "window_sec",
    "overlap",
]


def ensure_output_dir(path: Path | str = DEFAULT_OUTPUT_ROOT) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_output_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2, sort_keys=True)


def dataframe_to_markdown(df: pd.DataFrame, include_index: bool = False) -> str:
    render = df.copy()
    if include_index:
        headers = [render.index.name or "index"] + render.columns.tolist()
        rows = [[idx] + row.tolist() for idx, row in render.iterrows()]
    else:
        headers = render.columns.tolist()
        rows = render.values.tolist()
    table = [
        "| " + " | ".join(str(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        table.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(table)


def write_markdown(path: Path, text: str) -> None:
    ensure_output_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def family_spec(family_key: str) -> FamilySpec:
    return FAMILY_SPECS[family_key]


def parse_session_identity(path: Path, family_key: str) -> Dict[str, Any]:
    spec = family_spec(family_key)
    match = SESSION_NAME_RE.match(path.name)
    parsed_date = None
    run_index = None
    order_note = (
        "Ordered by embedded filename date (MM-DD-YY); ties are broken by the run index in parentheses."
    )
    if match:
        month = int(match.group(1))
        day = int(match.group(2))
        year = 2000 + int(match.group(3))
        run_index = int(match.group(4))
        parsed_date = date(year, month, day)
    else:
        order_note = "Embedded filename date could not be parsed; fell back to lexical filename ordering."

    return {
        "family": family_key,
        "family_display_name": spec.display_name,
        "filename": path.name,
        "file_path": path,
        "parsed_date": parsed_date.isoformat() if parsed_date else None,
        "run_index": run_index,
        "order_note": order_note,
        "order_key": (
            parsed_date or date.min,
            run_index if run_index is not None else math.inf,
            path.name,
        ),
    }


def ordered_session_records(family_key: str) -> List[Dict[str, Any]]:
    spec = family_spec(family_key)
    rows = [
        parse_session_identity(path, family_key)
        for path in spec.folder.glob(f"{spec.filename_prefix}-*.csv")
        if path.is_file()
    ]
    rows.sort(key=lambda row: row["order_key"])
    for rank, row in enumerate(rows, start=1):
        row["session_rank"] = rank
    return rows


def ordered_session_paths(family_key: str) -> List[Path]:
    return [row["file_path"] for row in ordered_session_records(family_key)]


def build_openbci_column_names(ncols: int) -> List[str]:
    if ncols < 10:
        raise ValueError(f"Expected at least 10 columns, found {ncols}.")
    columns = ["Sample_Index"] + CANONICAL_CHANNELS.copy()
    remaining = ncols - len(columns)
    if remaining == 1:
        return columns + ["Marker"]

    accel_count = min(3, max(0, remaining - 2))
    columns.extend(f"Accel_{idx}" for idx in range(accel_count))
    aux_count = ncols - len(columns) - 2
    columns.extend(f"Aux_{idx + 1}" for idx in range(max(0, aux_count)))

    trailing = ncols - len(columns)
    if trailing == 2:
        columns.extend(["Timestamp", "Marker"])
    elif trailing == 1:
        columns.append("Marker")
    return columns


def load_openbci_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None)
    df.columns = build_openbci_column_names(df.shape[1])
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def estimate_sampling(df: pd.DataFrame) -> Dict[str, Any]:
    if "Timestamp" not in df.columns:
        return {
            "timestamp_present": False,
            "fs_fallback_hz": DEFAULT_FS_FALLBACK,
            "fs_used_hz": DEFAULT_FS_FALLBACK,
            "notes": ["No timestamp column was available; used the 250 Hz fallback."],
        }

    timestamp = pd.to_numeric(df["Timestamp"], errors="coerce")
    timestamp_missing_fraction = float(timestamp.isna().mean())
    diffs = timestamp.diff()
    positive_diffs = diffs[(diffs > 0) & (diffs < 1)]

    fs_median = float("nan")
    median_dt = float("nan")
    jump_count = 0
    if not positive_diffs.empty:
        median_dt = float(positive_diffs.median())
        fs_median = float(1.0 / median_dt)
        jump_count = int((diffs > median_dt * 5.0).sum())

    fs_overall = float("nan")
    if len(timestamp) > 1 and pd.notna(timestamp.iloc[0]) and pd.notna(timestamp.iloc[-1]):
        total_dt = float(timestamp.iloc[-1] - timestamp.iloc[0])
        if total_dt > 0:
            fs_overall = float((len(timestamp) - 1) / total_dt)

    fs_used = fs_median if math.isfinite(fs_median) else DEFAULT_FS_FALLBACK
    notes: List[str] = []
    if jump_count:
        notes.append(
            f"Timestamp has {jump_count} jump(s) larger than 5x the median step; windowing uses the median step instead of total-span duration."
        )
    if math.isfinite(fs_overall) and math.isfinite(fs_median):
        relative_gap = abs(fs_overall - fs_median) / max(fs_median, 1e-9)
        if relative_gap > 0.10:
            notes.append(
                f"Overall timestamp-derived rate ({fs_overall:.2f} Hz) disagrees with median-step rate ({fs_median:.2f} Hz)."
            )
    if not notes and math.isfinite(fs_median):
        notes.append(f"Median timestamp delta is stable enough to use ({fs_median:.2f} Hz).")

    return {
        "timestamp_present": True,
        "timestamp_missing_fraction": timestamp_missing_fraction,
        "median_timestamp_step_sec": median_dt if math.isfinite(median_dt) else None,
        "fs_median_hz": fs_median if math.isfinite(fs_median) else None,
        "fs_overall_hz": fs_overall if math.isfinite(fs_overall) else None,
        "timestamp_jump_count": jump_count,
        "timestamp_negative_count": int((diffs < 0).sum()),
        "timestamp_zero_count": int((diffs == 0).sum()),
        "duration_from_samples_sec": float(len(df) / fs_used),
        "duration_from_timestamp_sec": float(timestamp.iloc[-1] - timestamp.iloc[0])
        if len(timestamp) > 1 and pd.notna(timestamp.iloc[0]) and pd.notna(timestamp.iloc[-1])
        else None,
        "fs_used_hz": fs_used,
        "notes": notes,
    }


def collapse_marker_events(marker: pd.Series) -> List[Dict[str, Any]]:
    rounded = pd.to_numeric(marker, errors="coerce").fillna(0).round().astype(int).to_numpy()
    events: List[Dict[str, Any]] = []
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


def pair_marker_events(
    events: Sequence[Dict[str, Any]], start_code: int, end_code: int
) -> Tuple[List[Dict[str, Any]], List[str]]:
    intervals: List[Dict[str, Any]] = []
    issues: List[str] = []
    open_start: Dict[str, Any] | None = None

    for event in events:
        code = int(event["code"])
        if code == start_code:
            if open_start is not None:
                issues.append(
                    f"marker {start_code} reappeared before marker {end_code}; replaced the open interval that started at sample {open_start['start_sample']}"
                )
            open_start = event
        elif code == end_code:
            if open_start is None:
                issues.append(
                    f"marker {end_code} appeared without a matching prior marker {start_code} at sample {event['start_sample']}"
                )
            else:
                intervals.append(
                    {
                        "start_sample": int(open_start["start_sample"]),
                        "end_sample": int(event["start_sample"]),
                        "start_event_end_sample": int(open_start["end_sample"]),
                        "end_event_end_sample": int(event["end_sample"]),
                    }
                )
                open_start = None

    if open_start is not None:
        issues.append(
            f"marker {start_code} at sample {open_start['start_sample']} was never closed by marker {end_code}"
        )

    return intervals, issues


def marker_transition_issues(events: Sequence[Dict[str, Any]]) -> List[str]:
    codes = [int(event["code"]) for event in events]
    allowed_next = {
        1: {2},
        2: {1, 3},
        3: {4},
        4: {1, 3},
    }
    issues: List[str] = []
    for idx, (previous_code, current_code) in enumerate(zip(codes, codes[1:]), start=1):
        if current_code not in allowed_next.get(previous_code, {1, 2, 3, 4}):
            issues.append(
                f"unexpected marker transition {previous_code}->{current_code} between event {idx} and {idx + 1}"
            )
    return issues


def summarize_marker_counts(marker: pd.Series) -> Dict[int, int]:
    rounded = pd.to_numeric(marker, errors="coerce").fillna(0).round().astype(int)
    counts = rounded.value_counts().sort_index().to_dict()
    return {int(key): int(value) for key, value in counts.items()}


def build_segments(
    family_key: str,
    total_samples: int,
    events: Sequence[Dict[str, Any]],
    fs_hz: float,
) -> Dict[str, Any]:
    spec = family_spec(family_key)
    movement1_intervals, movement1_issues = pair_marker_events(events, 1, 2)
    movement2_intervals, movement2_issues = pair_marker_events(events, 3, 4)

    active_segments: List[Dict[str, Any]] = []
    for interval in movement1_intervals:
        active_segments.append(
            {
                "label": spec.movement1_label,
                "segment_kind": "movement1",
                "segment_source": "marker_1_to_2",
                "segment_context": None,
                "start_sample": interval["start_sample"],
                "end_sample": interval["end_sample"],
                "expected_duration_sec": 8.0,
            }
        )
    for interval in movement2_intervals:
        active_segments.append(
            {
                "label": spec.movement2_label,
                "segment_kind": "movement2",
                "segment_source": "marker_3_to_4",
                "segment_context": None,
                "start_sample": interval["start_sample"],
                "end_sample": interval["end_sample"],
                "expected_duration_sec": 8.0,
            }
        )
    active_segments.sort(key=lambda row: (row["start_sample"], row["end_sample"]))

    overlap_issues: List[str] = []
    for previous, current in zip(active_segments, active_segments[1:]):
        if current["start_sample"] <= previous["end_sample"]:
            overlap_issues.append(
                f"active segments overlap: {previous['segment_kind']} ending at {previous['end_sample']} and {current['segment_kind']} starting at {current['start_sample']}"
            )

    segments: List[Dict[str, Any]] = []
    if active_segments:
        first_start = active_segments[0]["start_sample"]
        if first_start > 0:
            segments.append(
                {
                    "label": LABEL_BASELINE,
                    "segment_kind": "baseline",
                    "segment_source": "inferred_pre_first_active_gap",
                    "segment_context": "start_to_first_active",
                    "start_sample": 0,
                    "end_sample": first_start - 1,
                    "expected_duration_sec": 45.0,
                }
            )
        for index, active in enumerate(active_segments):
            segments.append(active)
            rest_start = active["end_sample"] + 1
            if index + 1 < len(active_segments):
                next_active = active_segments[index + 1]
                rest_end = next_active["start_sample"] - 1
                next_kind = next_active["segment_kind"]
            else:
                rest_end = total_samples - 1
                next_kind = "end_of_recording"
            if rest_end >= rest_start:
                expected_duration = None
                if active["segment_kind"] == "movement1" and next_kind == "movement2":
                    expected_duration = 5.0
                elif active["segment_kind"] == "movement2" and next_kind == "movement1":
                    expected_duration = 15.0
                segments.append(
                    {
                        "label": LABEL_REST,
                        "segment_kind": "rest",
                        "segment_source": "inferred_between_active_gaps",
                        "segment_context": f"{active['segment_kind']}_to_{next_kind}",
                        "start_sample": rest_start,
                        "end_sample": rest_end,
                        "expected_duration_sec": expected_duration,
                    }
                )
    else:
        segments.append(
            {
                "label": LABEL_BASELINE,
                "segment_kind": "baseline",
                "segment_source": "inferred_no_active_markers",
                "segment_context": "full_recording",
                "start_sample": 0,
                "end_sample": total_samples - 1,
                "expected_duration_sec": None,
            }
        )

    for segment_index, segment in enumerate(segments):
        duration_sec = (segment["end_sample"] - segment["start_sample"] + 1) / fs_hz
        segment["segment_index"] = segment_index
        segment["start_time_sec"] = segment["start_sample"] / fs_hz
        segment["end_time_sec"] = segment["end_sample"] / fs_hz
        segment["duration_sec"] = duration_sec
        expected_duration = segment["expected_duration_sec"]
        segment["duration_error_sec"] = (
            duration_sec - expected_duration if expected_duration is not None else None
        )

    return {
        "segments": segments,
        "movement1_count": len(movement1_intervals),
        "movement2_count": len(movement2_intervals),
        "pairing_issues": movement1_issues + movement2_issues + overlap_issues,
    }


def segment_dataframe(segments: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    if not segments:
        return pd.DataFrame(
            columns=[
                "label",
                "segment_kind",
                "segment_source",
                "segment_context",
                "segment_index",
                "start_sample",
                "end_sample",
                "start_time_sec",
                "end_time_sec",
                "duration_sec",
                "expected_duration_sec",
                "duration_error_sec",
            ]
        )
    return pd.DataFrame(segments)


def summarize_segment_protocol(
    segments: Sequence[Dict[str, Any]], expected_movement_count: int = 10
) -> Dict[str, Any]:
    segment_df = segment_dataframe(segments)
    summary: Dict[str, Any] = {}

    for segment_kind in ("baseline", "rest", "movement1", "movement2"):
        subset = segment_df[segment_df["segment_kind"] == segment_kind]
        if subset.empty:
            summary[segment_kind] = {"count": 0}
            continue
        summary[segment_kind] = {
            "count": int(len(subset)),
            "median_duration_sec": float(subset["duration_sec"].median()),
            "min_duration_sec": float(subset["duration_sec"].min()),
            "max_duration_sec": float(subset["duration_sec"].max()),
        }

    movement1_count = summary["movement1"]["count"]
    movement2_count = summary["movement2"]["count"]
    protocol_issues: List[str] = []
    if movement1_count != expected_movement_count:
        protocol_issues.append(
            f"movement 1 produced {movement1_count} interval(s); protocol expected about {expected_movement_count}"
        )
    if movement2_count != expected_movement_count:
        protocol_issues.append(
            f"movement 2 produced {movement2_count} interval(s); protocol expected about {expected_movement_count}"
        )

    baseline_summary = summary.get("baseline", {})
    if baseline_summary.get("count", 0):
        baseline_median = baseline_summary["median_duration_sec"]
        if abs(baseline_median - 45.0) > 10.0:
            protocol_issues.append(
                f"baseline interval is {baseline_median:.2f} s, which is materially different from the expected 45 s"
            )

    for active_key in ("movement1", "movement2"):
        if summary[active_key].get("count", 0):
            active_median = summary[active_key]["median_duration_sec"]
            if abs(active_median - 8.0) > 2.0:
                protocol_issues.append(
                    f"{active_key} median duration is {active_median:.2f} s instead of the expected 8 s"
                )

    rest_subset = segment_df[segment_df["segment_kind"] == "rest"]
    if not rest_subset.empty:
        for context, expected_duration in (
            ("movement1_to_movement2", 5.0),
            ("movement2_to_movement1", 15.0),
        ):
            context_subset = rest_subset[rest_subset["segment_context"] == context]
            if context_subset.empty:
                continue
            context_median = float(context_subset["duration_sec"].median())
            if abs(context_median - expected_duration) > 3.0:
                protocol_issues.append(
                    f"{context} rest median is {context_median:.2f} s instead of the expected {expected_duration:.0f} s"
                )

    summary["protocol_issues"] = protocol_issues
    return summary


def compute_channel_quality(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for channel in CANONICAL_CHANNELS:
        if channel not in df.columns:
            continue
        series = pd.to_numeric(df[channel], errors="coerce")
        abs_series = series.abs()
        dominant_fraction = (
            float(series.value_counts(dropna=False, normalize=True).iloc[0]) if len(series) else 1.0
        )
        rail_fraction = float((np.abs(abs_series - NOMINAL_RAIL) <= RAIL_TOLERANCE).mean())
        max_abs = float(abs_series.max()) if len(series) else float("nan")
        centered_std = float((series - series.median()).std()) if len(series) else float("nan")
        missing_fraction = float(series.isna().mean()) if len(series) else 1.0
        status = "safe"
        reasons: List[str] = []

        if missing_fraction > 0.0:
            reasons.append(f"missing fraction {missing_fraction:.4f}")
        if dominant_fraction >= 0.999:
            status = "unsafe"
            reasons.append(f"dominant exact value fraction {dominant_fraction:.4f}")
        if centered_std == 0.0:
            status = "unsafe"
            reasons.append("centered standard deviation is 0")
        if rail_fraction >= UNSAFE_RAIL_FRACTION:
            status = "unsafe"
            reasons.append(f"rail fraction {rail_fraction:.4f}")
        elif rail_fraction >= QUESTIONABLE_RAIL_FRACTION and status != "unsafe":
            status = "questionable"
            reasons.append(f"rail fraction {rail_fraction:.4f}")
        elif max_abs >= NOMINAL_RAIL - RAIL_TOLERANCE and status == "safe":
            status = "questionable"
            reasons.append("at least one sample touched the nominal OpenBCI rail")
        if missing_fraction > 0.0 and status == "safe":
            status = "questionable"

        rows.append(
            {
                "channel": channel,
                "status": status,
                "missing_fraction": missing_fraction,
                "dominant_value_fraction": dominant_fraction,
                "rail_fraction": rail_fraction,
                "centered_std": centered_std,
                "min_value": float(series.min()),
                "max_value": float(series.max()),
                "max_abs_value": max_abs,
                "reasons": "; ".join(reasons),
            }
        )

    return pd.DataFrame(rows)


def summarize_channel_quality(channel_quality_df: pd.DataFrame) -> Dict[str, List[str]]:
    safe = sorted(channel_quality_df.loc[channel_quality_df["status"] == "safe", "channel"].tolist())
    questionable = sorted(
        channel_quality_df.loc[channel_quality_df["status"] == "questionable", "channel"].tolist()
    )
    unsafe = sorted(channel_quality_df.loc[channel_quality_df["status"] == "unsafe", "channel"].tolist())
    return {"safe": safe, "questionable": questionable, "unsafe": unsafe}


def audit_session(path: Path, family_key: str) -> Dict[str, Any]:
    meta = parse_session_identity(path, family_key)
    df = load_openbci_csv(path)
    sampling = estimate_sampling(df)
    marker_present = "Marker" in df.columns
    marker_events = collapse_marker_events(df["Marker"]) if marker_present else []
    marker_counts = summarize_marker_counts(df["Marker"]) if marker_present else {}
    transition_issues = marker_transition_issues(marker_events)
    segment_info = build_segments(
        family_key=family_key,
        total_samples=len(df),
        events=marker_events,
        fs_hz=float(sampling["fs_used_hz"]),
    )
    segments = segment_info["segments"]
    protocol_summary = summarize_segment_protocol(segments)
    channel_quality_df = compute_channel_quality(df)
    channel_summary = summarize_channel_quality(channel_quality_df)

    issues: List[str] = []
    issues.extend(transition_issues)
    issues.extend(segment_info["pairing_issues"])
    issues.extend(protocol_summary["protocol_issues"])
    issues.extend(sampling["notes"])

    expected_codes = {0, 1, 2, 3, 4}
    unexpected_codes = sorted(code for code in marker_counts if code not in expected_codes)
    if unexpected_codes:
        issues.append(f"unexpected marker values observed: {unexpected_codes}")
    if any(event["run_length_samples"] > 1 for event in marker_events):
        issues.append("one or more marker values persisted for multiple samples")

    usable_for_audit = marker_present and protocol_summary["movement1"]["count"] > 0 and protocol_summary["movement2"]["count"] > 0
    usable_channel_count = len(channel_summary["safe"]) + len(channel_summary["questionable"])
    usable_for_modeling = usable_for_audit and usable_channel_count >= 2

    return {
        **meta,
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
        "marker_column_exists": marker_present,
        "marker_counts": marker_counts,
        "marker_events": marker_events,
        "sampling": sampling,
        "segments": segments,
        "protocol_summary": protocol_summary,
        "channel_quality": channel_quality_df.to_dict(orient="records"),
        "channel_summary": channel_summary,
        "usable_for_audit": usable_for_audit,
        "usable_for_modeling": usable_for_modeling,
        "issues": issues,
    }


def audit_all_sessions() -> List[Dict[str, Any]]:
    audits: List[Dict[str, Any]] = []
    for family_key in FAMILY_SPECS:
        for row in ordered_session_records(family_key):
            audit = audit_session(row["file_path"], family_key)
            audit["session_rank"] = row["session_rank"]
            audit["run_index"] = row["run_index"]
            audit["parsed_date"] = row["parsed_date"]
            audit["order_note"] = row["order_note"]
            audits.append(audit)
    audits.sort(key=lambda row: (row["family"], row["session_rank"]))
    return audits


def audits_by_family(audits: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {family_key: [] for family_key in FAMILY_SPECS}
    for audit in audits:
        grouped[audit["family"]].append(audit)
    for family_key in grouped:
        grouped[family_key].sort(key=lambda row: row["session_rank"])
    return grouped


def split_train_test_audits(
    audits: Sequence[Dict[str, Any]], family_key: str
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    family_audits = audits_by_family(audits)[family_key]
    train_count = family_spec(family_key).train_session_count
    if len(family_audits) < train_count + 1:
        raise ValueError(f"Not enough sessions for family '{family_key}' to create the requested split.")
    train_audits = family_audits[:train_count]
    test_audit = family_audits[-1]
    return train_audits, test_audit


def select_model_channels(
    train_audits: Sequence[Dict[str, Any]],
) -> Tuple[List[str], Dict[str, str]]:
    statuses_by_channel: Dict[str, List[Tuple[str, str, str]]] = {channel: [] for channel in CANONICAL_CHANNELS}
    for audit in train_audits:
        channel_df = pd.DataFrame(audit["channel_quality"])
        if channel_df.empty:
            continue
        for _, row in channel_df.iterrows():
            statuses_by_channel[row["channel"]].append((audit["filename"], row["status"], row["reasons"]))

    selected: List[str] = []
    excluded: Dict[str, str] = {}
    for channel, statuses in statuses_by_channel.items():
        if not statuses:
            excluded[channel] = "channel metrics were unavailable"
            continue
        unsafe_hits = [f"{filename} ({reason or status})" for filename, status, reason in statuses if status == "unsafe"]
        if unsafe_hits:
            excluded[channel] = "unsafe in training sessions: " + ", ".join(unsafe_hits)
            continue
        selected.append(channel)
    return selected, excluded


def _butter_bandpass(low_hz: float, high_hz: float, fs_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    nyquist = 0.5 * fs_hz
    low = max(low_hz / nyquist, 1e-5)
    high = min(high_hz / nyquist, 0.99)
    if low >= high:
        raise ValueError(f"Invalid bandpass range ({low_hz}, {high_hz}) for fs={fs_hz}")
    return butter(2, [low, high], btype="band")


def preprocess_session_signals(
    df: pd.DataFrame,
    family_key: str,
    channel_columns: Sequence[str],
    fs_hz: float,
) -> pd.DataFrame:
    signals = df.loc[:, list(channel_columns)].astype(float).copy()
    signals = signals.sub(signals.median(axis=0), axis=1) * DEFAULT_COUNT_TO_UV

    b_notch, a_notch = iirnotch(60.0, 30.0, fs_hz)
    if family_key == "left_right":
        low_hz, high_hz = 1.0, min(40.0, fs_hz * 0.45)
    else:
        low_hz, high_hz = 20.0, min(100.0, fs_hz * 0.45)
    b_band, a_band = _butter_bandpass(low_hz, high_hz, fs_hz)

    for column in signals.columns:
        values = signals[column].to_numpy(dtype=float)
        if np.isnan(values).any():
            values = np.nan_to_num(values, nan=float(np.nanmedian(values)))
        filtered = filtfilt(b_notch, a_notch, values)
        filtered = filtfilt(b_band, a_band, filtered)
        signals[column] = filtered

    return signals


def _time_feature_dict(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "rms": float(np.sqrt(np.mean(values ** 2))),
        "ptp": float(np.ptp(values)),
        "var": float(np.var(values)),
        "energy": float(np.mean(values ** 2)),
    }


def _spectral_feature_dict(values: np.ndarray, fs_hz: float) -> Dict[str, float]:
    freqs, psd = welch(values, fs=fs_hz, nperseg=min(len(values), max(8, int(round(fs_hz)))))
    psd = np.nan_to_num(psd, nan=0.0, posinf=0.0, neginf=0.0)
    mu_mask = (freqs >= 8.0) & (freqs <= 12.0)
    beta_mask = (freqs >= 13.0) & (freqs <= 30.0)
    return {
        "mu_power": float(np.sum(psd[mu_mask])) if np.any(mu_mask) else 0.0,
        "beta_power": float(np.sum(psd[beta_mask])) if np.any(beta_mask) else 0.0,
    }


def extract_window_features(
    window: np.ndarray,
    channel_columns: Sequence[str],
    fs_hz: float,
    feature_mode: str = "time_only",
    with_asymmetry: bool = False,
) -> Dict[str, float]:
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"Unsupported feature_mode '{feature_mode}'. Expected one of {FEATURE_MODES}.")

    features: Dict[str, float] = {}
    spectral_values: Dict[str, Dict[str, float]] = {}

    for channel_index, channel_name in enumerate(channel_columns):
        values = window[:, channel_index]
        if feature_mode in ("time_only", "combined"):
            for suffix, value in _time_feature_dict(values).items():
                features[f"{channel_name}_{suffix}"] = value
        if feature_mode in ("spectral_only", "combined"):
            spectral_values[channel_name] = _spectral_feature_dict(values, fs_hz)
            for suffix, value in spectral_values[channel_name].items():
                features[f"{channel_name}_{suffix}"] = value

    if with_asymmetry and spectral_values:
        for left_channel, right_channel in combinations(channel_columns, 2):
            left_spectral = spectral_values[left_channel]
            right_spectral = spectral_values[right_channel]
            features[f"mu_asym_{left_channel}__{right_channel}"] = float(
                left_spectral["mu_power"] - right_spectral["mu_power"]
            )
            features[f"beta_asym_{left_channel}__{right_channel}"] = float(
                left_spectral["beta_power"] - right_spectral["beta_power"]
            )

    return features


def build_windows_for_audit(
    audit: Dict[str, Any],
    channel_columns: Sequence[str],
    split_role: str,
    window_sec: float = DEFAULT_WINDOW_SEC,
    overlap: float = DEFAULT_OVERLAP,
    feature_mode: str = "time_only",
    with_asymmetry: bool = False,
) -> pd.DataFrame:
    df = load_openbci_csv(Path(audit["file_path"]))
    fs_hz = float(audit["sampling"]["fs_used_hz"])
    signals = preprocess_session_signals(df, audit["family"], channel_columns, fs_hz)
    window_samples = max(1, int(round(window_sec * fs_hz)))
    hop_samples = max(1, int(round(window_samples * (1.0 - overlap))))

    rows: List[Dict[str, Any]] = []
    for segment in audit["segments"]:
        if segment["label"] not in {
            LABEL_BASELINE,
            LABEL_REST,
            family_spec(audit["family"]).movement1_label,
            family_spec(audit["family"]).movement2_label,
        }:
            continue

        start_sample = int(segment["start_sample"])
        end_sample = int(segment["end_sample"])
        if end_sample - start_sample + 1 < window_samples:
            continue

        for window_start in range(start_sample, end_sample - window_samples + 2, hop_samples):
            window_end = window_start + window_samples - 1
            window = signals.iloc[window_start : window_end + 1].to_numpy(dtype=float)
            row = {
                "family": audit["family"],
                "filename": audit["filename"],
                "file_path": str(audit["file_path"]),
                "session_date": audit["parsed_date"],
                "session_rank": audit["session_rank"],
                "split_role": split_role,
                "label": segment["label"],
                "segment_kind": segment["segment_kind"],
                "segment_source": segment["segment_source"],
                "segment_context": segment["segment_context"],
                "segment_index": segment["segment_index"],
                "start_sample": window_start,
                "end_sample": window_end,
                "start_time_sec": window_start / fs_hz,
                "end_time_sec": window_end / fs_hz,
                "window_sec": window_sec,
                "overlap": overlap,
            }
            row.update(
                extract_window_features(
                    window,
                    channel_columns,
                    fs_hz=fs_hz,
                    feature_mode=feature_mode,
                    with_asymmetry=with_asymmetry,
                )
            )
            rows.append(row)

    return pd.DataFrame(rows)


def build_family_windows(
    audits: Sequence[Dict[str, Any]],
    family_key: str,
    window_sec: float = DEFAULT_WINDOW_SEC,
    overlap: float = DEFAULT_OVERLAP,
    feature_mode: str = "time_only",
    with_asymmetry: bool = False,
) -> Dict[str, Any]:
    train_audits, test_audit = split_train_test_audits(audits, family_key)
    selected_channels, excluded_channels = select_model_channels(train_audits)
    if not selected_channels:
        raise ValueError(f"No modeling channels survived the training-session quality screen for {family_key}.")

    train_frames = [
        build_windows_for_audit(
            audit,
            selected_channels,
            split_role="train",
            window_sec=window_sec,
            overlap=overlap,
            feature_mode=feature_mode,
            with_asymmetry=with_asymmetry,
        )
        for audit in train_audits
    ]
    test_frame = build_windows_for_audit(
        test_audit,
        selected_channels,
        split_role="test",
        window_sec=window_sec,
        overlap=overlap,
        feature_mode=feature_mode,
        with_asymmetry=with_asymmetry,
    )
    train_frame = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()
    feature_columns = [column for column in train_frame.columns if column not in WINDOW_METADATA_COLUMNS]

    return {
        "family": family_key,
        "selected_channels": selected_channels,
        "excluded_channels": excluded_channels,
        "train_audits": train_audits,
        "test_audit": test_audit,
        "train_windows": train_frame,
        "test_windows": test_frame,
        "feature_columns": feature_columns,
        "window_sec": window_sec,
        "overlap": overlap,
        "feature_mode": feature_mode,
        "with_asymmetry": with_asymmetry,
    }


def inventory_rows(audits: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for audit in audits:
        issues = audit["issues"]
        rows.append(
            {
                "family": audit["family"],
                "session_rank": audit["session_rank"],
                "filename": audit["filename"],
                "file_path": str(audit["file_path"]),
                "parsed_date": audit["parsed_date"],
                "run_index": audit["run_index"],
                "shape": "x".join(str(value) for value in audit["shape"]),
                "columns": ", ".join(audit["columns"]),
                "marker_column_exists": audit["marker_column_exists"],
                "sample_rate_used_hz": round(float(audit["sampling"]["fs_used_hz"]), 3),
                "usable_for_audit": audit["usable_for_audit"],
                "usable_for_modeling": audit["usable_for_modeling"],
                "channel_safe_count": len(audit["channel_summary"]["safe"]),
                "channel_questionable_count": len(audit["channel_summary"]["questionable"]),
                "channel_unsafe_count": len(audit["channel_summary"]["unsafe"]),
                "top_issue": issues[0] if issues else "",
            }
        )
    return pd.DataFrame(rows).sort_values(["family", "session_rank"]).reset_index(drop=True)
