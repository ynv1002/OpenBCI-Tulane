from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.utils import (
    audit_session,
    collapse_marker_events,
    compute_channel_quality,
    dataframe_to_markdown,
    ensure_output_dir,
    estimate_sampling,
    load_openbci_csv,
    pair_marker_events,
    write_json,
)


DEFAULT_INPUT_DIR = Path("/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/EEG_LR")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
LEFT = "LEFT"
RIGHT = "RIGHT"
VALID_MARKER_PAIRS = {
    (1, 2): LEFT,
    (3, 4): RIGHT,
}


def _load_frozen_lr6_review() -> Any:
    module_path = REPO_ROOT / "analysis" / "dalin-lr-test" / "review_lr6.py"
    spec = importlib.util.spec_from_file_location("frozen_lr6_review", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load frozen LR(6) review module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FROZEN_LR6 = _load_frozen_lr6_review()
COUNT_SMOOTHING_SAMPLES = int(FROZEN_LR6.COUNT_SMOOTHING_SAMPLES)
COUNT_MIN_PEAK_DISTANCE_SEC = float(FROZEN_LR6.COUNT_MIN_PEAK_DISTANCE_SEC)
COUNT_PROMINENCE_STD_SCALE = float(FROZEN_LR6.COUNT_PROMINENCE_STD_SCALE)
COUNT_MIN_PROMINENCE_UV = float(FROZEN_LR6.COUNT_MIN_PROMINENCE_UV)
COUNT_EARLY_BOUNDARY_EXCLUSION_SEC = float(FROZEN_LR6.COUNT_EARLY_BOUNDARY_EXCLUSION_SEC)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the frozen LR(6) event counter to Yaniv EEG_LR runs for event alignment validation."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing Yaniv EEG_LR CSV runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for per-file and aggregate validation outputs.",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=250.0,
        help="Authoritative sample rate for reporting and event timing.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip per-file debug plots.",
    )
    return parser.parse_args()


def _fail(message: str) -> None:
    raise SystemExit(message)


def _iter_csvs(input_dir: Path) -> list[Path]:
    csv_paths = sorted(path for path in input_dir.glob("*.csv") if path.is_file())
    if not csv_paths:
        _fail(f"No CSV files found in {input_dir}")
    return csv_paths


def _target_count_channels(channel_quality_df: pd.DataFrame) -> list[str]:
    ordered = channel_quality_df.copy()
    ordered["channel_index"] = ordered["channel"].str.extract(r"(\d+)").astype(int)
    usable = ordered[ordered["status"] != "unsafe"].sort_values("channel_index")
    channels = usable["channel"].astype(str).tolist()
    if len(channels) < 2:
        _fail("Fewer than two target channels survived the unsafe-channel screen.")
    return channels


def _build_block_table(marker_events: list[dict[str, Any]], fs_hz: float) -> tuple[pd.DataFrame, list[str]]:
    left_intervals, left_issues = pair_marker_events(marker_events, 1, 2)
    right_intervals, right_issues = pair_marker_events(marker_events, 3, 4)

    rows: list[dict[str, Any]] = []
    for interval in left_intervals:
        start_sample = int(interval["start_sample"])
        end_sample = int(interval["end_sample"])
        rows.append(
            {
                "side": LEFT,
                "marker_pair_type": "1→2",
                "start_sample": start_sample,
                "end_sample": end_sample,
                "start_time_sec": float(start_sample / fs_hz),
                "end_time_sec": float(end_sample / fs_hz),
                "duration_sec": float((end_sample - start_sample) / fs_hz),
            }
        )
    for interval in right_intervals:
        start_sample = int(interval["start_sample"])
        end_sample = int(interval["end_sample"])
        rows.append(
            {
                "side": RIGHT,
                "marker_pair_type": "3→4",
                "start_sample": start_sample,
                "end_sample": end_sample,
                "start_time_sec": float(start_sample / fs_hz),
                "end_time_sec": float(end_sample / fs_hz),
                "duration_sec": float((end_sample - start_sample) / fs_hz),
            }
        )

    block_df = pd.DataFrame(rows).sort_values(["start_sample", "end_sample"]).reset_index(drop=True)
    if not block_df.empty:
        block_df.insert(0, "block_id", np.arange(1, len(block_df) + 1, dtype=int))
    return block_df, left_issues + right_issues


def _format_seconds_list(values: list[float]) -> str:
    return json.dumps([round(float(value), 3) for value in values])


def _detect_events_for_window(
    trace_values: np.ndarray,
    start_sample: int,
    end_sample: int,
    fs_hz: float,
    *,
    start_time_sec: float,
    relative_origin_time_sec: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    window_values = trace_values[start_sample : end_sample + 1]
    window_smooth = FROZEN_LR6._smooth_values(window_values, COUNT_SMOOTHING_SAMPLES)
    min_distance_samples = max(1, int(round(COUNT_MIN_PEAK_DISTANCE_SEC * fs_hz)))
    prominence_threshold = max(float(np.std(window_smooth)) * COUNT_PROMINENCE_STD_SCALE, COUNT_MIN_PROMINENCE_UV)
    peak_indices, properties = find_peaks(
        window_smooth,
        prominence=prominence_threshold,
        distance=min_distance_samples,
    )

    event_rows: list[dict[str, Any]] = []
    kept_count = 0
    dropped_early_count = 0
    kept_times: list[float] = []
    for event_index, peak_idx in enumerate(peak_indices, start=1):
        peak_sample = start_sample + int(peak_idx)
        peak_time_sec = float(peak_sample / fs_hz)
        peak_time_relative_sec = float(peak_time_sec - relative_origin_time_sec)
        peak_prominence = float(properties["prominences"][event_index - 1])
        peak_value = float(window_smooth[peak_idx])
        kept_for_count = peak_time_relative_sec >= COUNT_EARLY_BOUNDARY_EXCLUSION_SEC
        filter_reason = "" if kept_for_count else "early_boundary_exclusion"
        if kept_for_count:
            kept_count += 1
            kept_times.append(peak_time_sec)
        else:
            dropped_early_count += 1

        event_rows.append(
            {
                "peak_sample": peak_sample,
                "event_time_sec": peak_time_sec,
                "event_time_relative_sec": peak_time_relative_sec,
                "peak_value": peak_value,
                "peak_prominence": peak_prominence,
                "kept_for_count": bool(kept_for_count),
                "filter_reason": filter_reason,
                "window_start_time_sec": start_time_sec,
                "window_end_time_sec": float(end_sample / fs_hz),
            }
        )

    metrics = {
        "detected_event_count": int(kept_count),
        "kept_event_times_sec": kept_times,
        "max_trace_value": float(window_smooth.max()) if len(window_smooth) else float("nan"),
        "mean_trace_value": float(window_smooth.mean()) if len(window_smooth) else float("nan"),
        "prominence_threshold_uv": float(prominence_threshold),
        "dropped_early_count": int(dropped_early_count),
    }
    return event_rows, metrics


def _sample_to_block(sample: int, blocks_df: pd.DataFrame) -> dict[str, Any] | None:
    if blocks_df.empty:
        return None
    matches = blocks_df[
        (blocks_df["start_sample"] <= sample)
        & (blocks_df["end_sample"] >= sample)
    ]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def _build_file_outputs(
    csv_path: Path,
    raw_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    count_signal: np.ndarray,
    aggregate_rms_smooth: np.ndarray,
    fs_hz: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    block_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    next_event_id = 1

    for block in blocks_df.to_dict(orient="records"):
        block_events, block_metrics = _detect_events_for_window(
            count_signal,
            start_sample=int(block["start_sample"]),
            end_sample=int(block["end_sample"]),
            fs_hz=fs_hz,
            start_time_sec=float(block["start_time_sec"]),
            relative_origin_time_sec=float(block["start_time_sec"]),
        )

        block_notes: list[str] = []
        if block_metrics["detected_event_count"] == 0:
            block_notes.append("no kept events detected")
        if block_metrics["dropped_early_count"]:
            block_notes.append(
                f"{block_metrics['dropped_early_count']} early peak(s) excluded before {COUNT_EARLY_BOUNDARY_EXCLUSION_SEC:.2f}s"
            )

        block_rows.append(
            {
                "file": csv_path.name,
                "block_id": int(block["block_id"]),
                "side": str(block["side"]),
                "start_time_sec": float(block["start_time_sec"]),
                "end_time_sec": float(block["end_time_sec"]),
                "duration_sec": float(block["duration_sec"]),
                "detected_event_count": int(block_metrics["detected_event_count"]),
                "kept_event_times_sec": _format_seconds_list(block_metrics["kept_event_times_sec"]),
                "max_trace_value": float(block_metrics["max_trace_value"]),
                "mean_trace_value": float(block_metrics["mean_trace_value"]),
                "notes": "; ".join(block_notes),
            }
        )

        for block_event in block_events:
            event_rows.append(
                {
                    "file": csv_path.name,
                    "event_id": next_event_id,
                    "event_time_sec": float(block_event["event_time_sec"]),
                    "event_time_relative_sec": float(block_event["event_time_relative_sec"]),
                    "peak_value": float(block_event["peak_value"]),
                    "peak_prominence": float(block_event["peak_prominence"]),
                    "kept_for_count": bool(block_event["kept_for_count"]),
                    "filter_reason": str(block_event["filter_reason"]),
                    "assigned_block_id": int(block["block_id"]),
                    "assigned_side": str(block["side"]),
                    "inside_marker_block": "yes",
                    "notes": "",
                }
            )
            next_event_id += 1

    full_record_events, _ = _detect_events_for_window(
        count_signal,
        start_sample=0,
        end_sample=len(count_signal) - 1,
        fs_hz=fs_hz,
        start_time_sec=0.0,
        relative_origin_time_sec=0.0,
    )
    for event in full_record_events:
        match = _sample_to_block(int(event["peak_sample"]), blocks_df)
        if match is not None:
            continue
        event_rows.append(
            {
                "file": csv_path.name,
                "event_id": next_event_id,
                "event_time_sec": float(event["event_time_sec"]),
                "event_time_relative_sec": np.nan,
                "peak_value": float(event["peak_value"]),
                "peak_prominence": float(event["peak_prominence"]),
                "kept_for_count": bool(event["kept_for_count"]),
                "filter_reason": str(event["filter_reason"]),
                "assigned_block_id": "",
                "assigned_side": "",
                "inside_marker_block": "no",
                "notes": "detected outside all valid marker-defined blocks",
            }
        )
        next_event_id += 1

    block_summary_df = pd.DataFrame(block_rows)
    event_table_df = pd.DataFrame(event_rows).sort_values("event_time_sec").reset_index(drop=True)
    if not event_table_df.empty:
        event_table_df["event_id"] = np.arange(1, len(event_table_df) + 1, dtype=int)
    return block_summary_df, event_table_df


def _write_plot(
    output_path: Path,
    raw_df: pd.DataFrame,
    blocks_df: pd.DataFrame,
    aggregate_rms_smooth: np.ndarray,
    event_table_df: pd.DataFrame,
    fs_hz: float,
) -> str | None:
    mpl_config_dir = output_path.parent / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir.resolve()))
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return "matplotlib is unavailable; skipped validation plot"

    time_axis = np.arange(len(raw_df), dtype=float) / fs_hz
    fig, ax = plt.subplots(figsize=(15, 5))

    left_label_added = False
    right_label_added = False
    for block in blocks_df.to_dict(orient="records"):
        if block["side"] == LEFT:
            color = "#d9eaf7"
            label = "LEFT block" if not left_label_added else None
            left_label_added = True
        else:
            color = "#f8dfd0"
            label = "RIGHT block" if not right_label_added else None
            right_label_added = True
        ax.axvspan(
            float(block["start_time_sec"]),
            float(block["end_time_sec"]),
            color=color,
            alpha=0.35,
            label=label,
            zorder=0,
        )

    marker_samples = np.where(
        pd.to_numeric(raw_df["Marker"], errors="coerce").fillna(0.0).round().astype(int).to_numpy() != 0
    )[0]
    for marker_sample in marker_samples:
        ax.axvline(float(marker_sample / fs_hz), color="0.85", linewidth=0.7, alpha=0.7, zorder=1)

    ax.plot(
        time_axis,
        aggregate_rms_smooth,
        color="tab:blue",
        linewidth=1.2,
        label="Frozen LR(6) event-counter trace",
        zorder=2,
    )

    if not event_table_df.empty:
        kept = event_table_df[event_table_df["kept_for_count"] == True]
        dropped = event_table_df[event_table_df["kept_for_count"] == False]
        inside_left = kept[(kept["inside_marker_block"] == "yes") & (kept["assigned_side"] == LEFT)]
        inside_right = kept[(kept["inside_marker_block"] == "yes") & (kept["assigned_side"] == RIGHT)]
        outside = kept[kept["inside_marker_block"] == "no"]
        if not inside_left.empty:
            ax.scatter(
                inside_left["event_time_sec"],
                inside_left["peak_value"],
                color="tab:green",
                s=22,
                label="Kept LEFT events",
                zorder=3,
            )
        if not inside_right.empty:
            ax.scatter(
                inside_right["event_time_sec"],
                inside_right["peak_value"],
                color="tab:red",
                s=22,
                label="Kept RIGHT events",
                zorder=3,
            )
        if not outside.empty:
            ax.scatter(
                outside["event_time_sec"],
                outside["peak_value"],
                color="black",
                s=26,
                marker="x",
                label="Kept outside-block events",
                zorder=4,
            )
        if not dropped.empty:
            ax.scatter(
                dropped["event_time_sec"],
                dropped["peak_value"],
                color="orange",
                s=18,
                marker="v",
                label="Early excluded peaks",
                zorder=3,
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Aggregate RMS (uV)")
    ax.set_title(f"Yaniv EEG_LR validation: {output_path.stem.replace('_validation', '')}")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return None


def _file_interpretation(inside_events: int, outside_events: int, block_summary_df: pd.DataFrame) -> str:
    total_kept = inside_events + outside_events
    if total_kept == 0:
        return "No kept events were detected, so this file is not yet usable for event alignment."
    inside_fraction = inside_events / total_kept
    zero_blocks = int((block_summary_df["detected_event_count"] == 0).sum())
    if inside_fraction >= 0.90 and outside_events <= 2:
        return "Event alignment looks strong enough to support a later left/right event-classification pass."
    if inside_fraction >= 0.75:
        return "Event alignment is directionally usable, but outside-block events still need review before classification."
    if zero_blocks > 0:
        return "Many events fall outside valid blocks or too many blocks are empty; review this file before classifier work."
    return "Event alignment is weak enough that this file should stay diagnostic-only for now."


def _count_reasonableness_note(block_summary_df: pd.DataFrame) -> str:
    if block_summary_df.empty:
        return "No valid LEFT/RIGHT marker pairs were available."
    zero_blocks = int((block_summary_df["detected_event_count"] == 0).sum())
    median_count = float(block_summary_df["detected_event_count"].median())
    max_count = int(block_summary_df["detected_event_count"].max())
    if zero_blocks == 0 and max_count <= 8:
        return f"Per-block counts look broadly reasonable; median detected count is {median_count:.1f}."
    if zero_blocks <= max(1, len(block_summary_df) // 5):
        return f"Per-block counts are mixed but mostly usable; {zero_blocks} block(s) had no kept events."
    return f"Per-block counts look unstable; {zero_blocks} block(s) had no kept events."


def _summarize_file(
    csv_path: Path,
    output_dir: Path,
    block_summary_df: pd.DataFrame,
    event_table_df: pd.DataFrame,
    channel_quality_df: pd.DataFrame,
    count_channels: list[str],
    sampling_note: dict[str, Any],
    issues: list[str],
    plot_warning: str | None,
) -> tuple[dict[str, Any], list[str]]:
    kept_events = event_table_df[event_table_df["kept_for_count"] == True].copy()
    inside_events = int((kept_events["inside_marker_block"] == "yes").sum())
    outside_events = int((kept_events["inside_marker_block"] == "no").sum())
    total_kept = inside_events + outside_events
    inside_fraction = float(inside_events / total_kept) if total_kept else float("nan")
    left_blocks = int((block_summary_df["side"] == LEFT).sum())
    right_blocks = int((block_summary_df["side"] == RIGHT).sum())
    excluded_early = int((event_table_df["filter_reason"] == "early_boundary_exclusion").sum())
    interpretation = _file_interpretation(inside_events, outside_events, block_summary_df)
    count_note = _count_reasonableness_note(block_summary_df)

    summary_row = {
        "file": csv_path.name,
        "valid_block_count": int(len(block_summary_df)),
        "left_block_count": left_blocks,
        "right_block_count": right_blocks,
        "kept_events_inside_blocks": inside_events,
        "kept_events_outside_blocks": outside_events,
        "total_kept_events": total_kept,
        "inside_block_fraction": inside_fraction,
        "excluded_early_peak_count": excluded_early,
        "count_channels": ", ".join(count_channels),
        "audit_issue_count": int(len(issues)),
        "event_alignment_note": interpretation,
        "count_reasonableness_note": count_note,
    }

    preview = block_summary_df.loc[
        :,
        ["block_id", "side", "duration_sec", "detected_event_count", "notes"],
    ].copy()
    preview["duration_sec"] = preview["duration_sec"].map(lambda value: f"{float(value):.2f}")

    lines = [
        f"Yaniv EEG_LR validation for {csv_path.name}",
        f"  Output directory: {output_dir}",
        "  Fixed reporting rate: 250.0 Hz",
    ]
    lines.append(
        f"  Timestamp-derived rate note: {sampling_note['fs_used_hz']:.3f} Hz ({' | '.join(sampling_note['notes'])})"
    )
    lines.append(f"  Valid marker-defined blocks: {len(block_summary_df)} ({left_blocks} LEFT, {right_blocks} RIGHT)")
    lines.append(f"  Counting channels: {', '.join(count_channels)}")

    questionable = channel_quality_df[channel_quality_df["status"] != "safe"].copy()
    if not questionable.empty:
        lines.append("  Channel warnings:")
        for _, row in questionable.iterrows():
            lines.append(f"    - {row['channel']}: {row['status']} ({row['reasons']})")
    if issues:
        lines.append("  Audit warnings:")
        for issue in issues[:6]:
            lines.append(f"    - {issue}")
        if len(issues) > 6:
            lines.append(f"    - ... plus {len(issues) - 6} more")

    lines.extend(
        [
            "",
            "Per-block summary",
            dataframe_to_markdown(preview),
            "",
            "Event alignment summary",
            f"  Kept events inside valid blocks: {inside_events}",
            f"  Kept events outside all valid blocks: {outside_events}",
            f"  Inside-block fraction: {inside_fraction:.3f}" if total_kept else "  Inside-block fraction: n/a",
            f"  Early excluded peaks: {excluded_early}",
            "",
            "Interpretation",
            f"  - {interpretation}",
            f"  - {count_note}",
        ]
    )
    if plot_warning:
        lines.append(f"  - Plot warning: {plot_warning}")
    return summary_row, lines


def _write_readme(output_root: Path) -> None:
    readme_path = Path(__file__).resolve().parent / "README.md"
    readme_text = """# Yaniv EEG_LR Event Validation

This folder applies the frozen LR(6) event counter to Yaniv's cleaner EEG left/right recordings as a transfer and validation pass.

The goal is event alignment, not final side classification:
- marker pairs define the trusted LEFT and RIGHT block boundaries
- the frozen LR(6) counter is reused unchanged
- success is measured by whether detected events land inside valid marker-defined blocks

## Counter Settings

The event counter is intentionally frozen to match the tuned LR(6) review:
- smoothing window: `75` samples
- minimum peak distance: `0.75 s`
- prominence rule: `max(0.6 * trial_smoothed_std, 0.02 uV)`
- early-boundary exclusion: peaks before `0.10 s`

## Run

From the repo root:

```bash
python analysis/lr_event_validation/run_yaniv_lr_validation.py --input-dir "/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/EEG_LR"
```

## Outputs

Per file under `analysis/lr_event_validation/outputs/<file-stem>/`:
- `block_summary.csv`
- `event_table.csv`
- `<file-stem>_validation.png`
- `run_notes.json`

Aggregate outputs under `analysis/lr_event_validation/outputs/`:
- `all_block_summary.csv`
- `all_event_table.csv`
- `overall_summary.csv`

Marker anomalies are reported as warnings, but files are still processed whenever valid `1→2` and `3→4` pairs can be formed.
"""
    readme_path.write_text(readme_text, encoding="utf-8")


def run_validation(
    input_dir: Path,
    output_dir: Path,
    fs_hz: float,
    make_plot: bool,
) -> dict[str, Any]:
    csv_paths = _iter_csvs(input_dir)
    output_root = ensure_output_dir(output_dir)
    _write_readme(output_root)

    all_block_frames: list[pd.DataFrame] = []
    all_event_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for csv_path in csv_paths:
        raw_df = load_openbci_csv(csv_path.resolve())
        audit = audit_session(csv_path.resolve(), "left_right")
        sampling_note = estimate_sampling(raw_df)
        channel_quality_df = compute_channel_quality(raw_df)
        count_channels = _target_count_channels(channel_quality_df)
        marker_events = collapse_marker_events(raw_df["Marker"]) if "Marker" in raw_df.columns else []
        block_df, pairing_issues = _build_block_table(marker_events, fs_hz)
        file_dir = ensure_output_dir(output_root / csv_path.stem)

        if block_df.empty:
            run_notes = {
                "file": csv_path.name,
                "issues": ["No valid LEFT/RIGHT marker pairs were found.", *audit.get("issues", [])],
                "pairing_issues": pairing_issues,
                "sampling_note": sampling_note,
                "count_channels": count_channels,
            }
            write_json(file_dir / "run_notes.json", run_notes)
            summary_rows.append(
                {
                    "file": csv_path.name,
                    "valid_block_count": 0,
                    "left_block_count": 0,
                    "right_block_count": 0,
                    "kept_events_inside_blocks": 0,
                    "kept_events_outside_blocks": 0,
                    "total_kept_events": 0,
                    "inside_block_fraction": np.nan,
                    "excluded_early_peak_count": 0,
                    "count_channels": ", ".join(count_channels),
                    "audit_issue_count": int(len(audit.get("issues", []))),
                    "event_alignment_note": "No valid LEFT/RIGHT marker pairs were found.",
                    "count_reasonableness_note": "Counts were not evaluated because no valid blocks were available.",
                }
            )
            continue

        count_signal, aggregate_rms_smooth = FROZEN_LR6._build_count_signal(raw_df, count_channels, fs_hz)
        block_summary_df, event_table_df = _build_file_outputs(
            csv_path=csv_path,
            raw_df=raw_df,
            blocks_df=block_df,
            count_signal=count_signal,
            aggregate_rms_smooth=aggregate_rms_smooth,
            fs_hz=fs_hz,
        )
        block_summary_path = file_dir / "block_summary.csv"
        event_table_path = file_dir / "event_table.csv"
        block_summary_df.to_csv(block_summary_path, index=False)
        event_table_df.to_csv(event_table_path, index=False)

        plot_warning = None
        if make_plot:
            plot_warning = _write_plot(
                file_dir / f"{csv_path.stem}_validation.png",
                raw_df=raw_df,
                blocks_df=block_df,
                aggregate_rms_smooth=aggregate_rms_smooth,
                event_table_df=event_table_df,
                fs_hz=fs_hz,
            )

        file_issues = list(dict.fromkeys([*audit.get("issues", []), *pairing_issues]))
        run_notes = {
            "file": csv_path.name,
            "sampling_note": sampling_note,
            "count_channels": count_channels,
            "channel_quality": channel_quality_df.to_dict(orient="records"),
            "pairing_issues": pairing_issues,
            "audit_issues": audit.get("issues", []),
            "issues": file_issues,
            "plot_warning": plot_warning,
        }
        write_json(file_dir / "run_notes.json", run_notes)

        summary_row, summary_lines = _summarize_file(
            csv_path=csv_path,
            output_dir=file_dir,
            block_summary_df=block_summary_df,
            event_table_df=event_table_df,
            channel_quality_df=channel_quality_df,
            count_channels=count_channels,
            sampling_note=sampling_note,
            issues=file_issues,
            plot_warning=plot_warning,
        )
        print("\n".join(summary_lines))
        print("")

        all_block_frames.append(block_summary_df)
        all_event_frames.append(event_table_df)
        summary_rows.append(summary_row)

    all_block_summary_df = (
        pd.concat(all_block_frames, ignore_index=True).reset_index(drop=True)
        if all_block_frames
        else pd.DataFrame()
    )
    all_event_table_df = (
        pd.concat(all_event_frames, ignore_index=True).reset_index(drop=True)
        if all_event_frames
        else pd.DataFrame()
    )
    overall_summary_df = pd.DataFrame(summary_rows)
    if not overall_summary_df.empty:
        total_inside = int(overall_summary_df["kept_events_inside_blocks"].sum())
        total_outside = int(overall_summary_df["kept_events_outside_blocks"].sum())
        total_kept = total_inside + total_outside
        overall_row = {
            "file": "__overall__",
            "valid_block_count": int(overall_summary_df["valid_block_count"].sum()),
            "left_block_count": int(overall_summary_df["left_block_count"].sum()),
            "right_block_count": int(overall_summary_df["right_block_count"].sum()),
            "kept_events_inside_blocks": total_inside,
            "kept_events_outside_blocks": total_outside,
            "total_kept_events": total_kept,
            "inside_block_fraction": float(total_inside / total_kept) if total_kept else np.nan,
            "excluded_early_peak_count": int(overall_summary_df["excluded_early_peak_count"].sum()),
            "count_channels": "",
            "audit_issue_count": int(overall_summary_df["audit_issue_count"].sum()),
            "event_alignment_note": (
                "Most detected events stay inside trusted marker-defined blocks."
                if total_kept and (total_inside / total_kept) >= 0.80
                else "Outside-block events are still common enough that these runs need caution before classification."
            ),
            "count_reasonableness_note": "Use per-file summaries for block-level count sanity checks.",
        }
        overall_summary_df = pd.concat([overall_summary_df, pd.DataFrame([overall_row])], ignore_index=True)

    all_block_summary_df.to_csv(output_root / "all_block_summary.csv", index=False)
    all_event_table_df.to_csv(output_root / "all_event_table.csv", index=False)
    overall_summary_df.to_csv(output_root / "overall_summary.csv", index=False)

    kept_events = all_event_table_df[all_event_table_df["kept_for_count"] == True].copy()
    overall_inside = int((kept_events["inside_marker_block"] == "yes").sum()) if not kept_events.empty else 0
    overall_outside = int((kept_events["inside_marker_block"] == "no").sum()) if not kept_events.empty else 0
    overall_total = overall_inside + overall_outside
    overall_fraction = float(overall_inside / overall_total) if overall_total else float("nan")
    overall_note = (
        "usable for a later left/right event-classification pass"
        if overall_total and overall_fraction >= 0.80
        else "still mainly a diagnostic event-alignment pass"
    )
    print("Overall summary")
    print(f"  Files processed: {len(csv_paths)}")
    print(f"  Valid marker-defined blocks: {int(overall_summary_df.loc[overall_summary_df['file'] != '__overall__', 'valid_block_count'].sum())}")
    print(f"  Kept events inside valid blocks: {overall_inside}")
    print(f"  Kept events outside all valid blocks: {overall_outside}")
    print(f"  Inside-block fraction: {overall_fraction:.3f}" if overall_total else "  Inside-block fraction: n/a")
    print(f"  Interpretation: The event structure looks {overall_note}.")

    return {
        "all_block_summary_path": output_root / "all_block_summary.csv",
        "all_event_table_path": output_root / "all_event_table.csv",
        "overall_summary_path": output_root / "overall_summary.csv",
        "overall_summary_df": overall_summary_df,
    }


def main() -> None:
    args = parse_args()
    run_validation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fs_hz=float(args.fs),
        make_plot=not bool(args.no_plot),
    )


if __name__ == "__main__":
    main()
