from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent.parent / ".matplotlib-cache"),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.utils import (
        DEFAULT_OVERLAP,
        DEFAULT_WINDOW_SEC,
        LABEL_BASELINE,
        LABEL_REST,
        audit_all_sessions,
        ensure_output_dir,
        family_spec,
        load_openbci_csv,
        preprocess_session_signals,
        select_model_channels,
        split_train_test_audits,
        write_markdown,
    )
else:
    from .utils import (
        DEFAULT_OVERLAP,
        DEFAULT_WINDOW_SEC,
        LABEL_BASELINE,
        LABEL_REST,
        audit_all_sessions,
        ensure_output_dir,
        family_spec,
        load_openbci_csv,
        preprocess_session_signals,
        select_model_channels,
        split_train_test_audits,
        write_markdown,
    )


LABEL_ACTIVE = "ACTIVE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize aggregate RMS over time for one or more left/right sessions."
    )
    parser.add_argument(
        "--filename",
        action="append",
        default=None,
        help="Left/right session filename to inspect. Repeat to select multiple sessions. Defaults to all left/right sessions.",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=DEFAULT_WINDOW_SEC,
        help="Sliding window length in seconds.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=DEFAULT_OVERLAP,
        help="Window overlap fraction.",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.35,
        help="Fraction of highest-RMS windows to highlight (default: 0.35).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for plots and summary files.",
    )
    return parser.parse_args()


def map_segment_label(raw_label: str, spec) -> str:
    if raw_label in {spec.movement1_label, spec.movement2_label}:
        return LABEL_ACTIVE
    if raw_label == LABEL_REST:
        return LABEL_REST
    if raw_label == LABEL_BASELINE:
        return LABEL_BASELINE
    return "OTHER"


def build_sample_labels(total_samples: int, segments: Sequence[Dict[str, Any]], spec) -> np.ndarray:
    labels = np.full(total_samples, "OTHER", dtype=object)
    for segment in segments:
        mapped = map_segment_label(segment["label"], spec)
        labels[int(segment["start_sample"]) : int(segment["end_sample"]) + 1] = mapped
    return labels


def compute_rms_windows(
    signals: pd.DataFrame,
    sample_labels: np.ndarray,
    fs_hz: float,
    window_sec: float,
    overlap: float,
) -> pd.DataFrame:
    window_samples = max(1, int(round(window_sec * fs_hz)))
    hop_samples = max(1, int(round(window_samples * (1.0 - overlap))))
    rows: List[Dict[str, Any]] = []

    for start_sample in range(0, len(signals) - window_samples + 1, hop_samples):
        end_sample = start_sample + window_samples - 1
        center_sample = start_sample + (window_samples // 2)
        window = signals.iloc[start_sample : end_sample + 1].to_numpy(dtype=float)
        channel_rms = np.sqrt(np.mean(window ** 2, axis=0))
        row = {
            "start_sample": start_sample,
            "end_sample": end_sample,
            "center_sample": center_sample,
            "start_time_sec": start_sample / fs_hz,
            "end_time_sec": end_sample / fs_hz,
            "center_time_sec": center_sample / fs_hz,
            "mapped_label": str(sample_labels[center_sample]),
            "aggregate_rms": float(np.mean(channel_rms)),
        }
        for channel_name, rms_value in zip(signals.columns, channel_rms):
            row[f"{channel_name}_rms"] = float(rms_value)
        rows.append(row)

    return pd.DataFrame(rows)


def top_window_mask(window_df: pd.DataFrame, top_fraction: float) -> Tuple[pd.Series, float]:
    valid = window_df["mapped_label"].isin([LABEL_ACTIVE, LABEL_REST])
    threshold = float(window_df.loc[valid, "aggregate_rms"].quantile(1.0 - top_fraction))
    mask = valid & (window_df["aggregate_rms"] >= threshold)
    return mask, threshold


def add_segment_shading(ax: plt.Axes, segments: Sequence[Dict[str, Any]], spec) -> None:
    active_added = False
    rest_added = False
    baseline_added = False
    for segment in segments:
        mapped = map_segment_label(segment["label"], spec)
        if mapped == LABEL_ACTIVE:
            color = "#f6b26b"
            alpha = 0.18
            label = "ACTIVE" if not active_added else None
            active_added = True
        elif mapped == LABEL_REST:
            color = "#b6d7a8"
            alpha = 0.14
            label = "REST" if not rest_added else None
            rest_added = True
        elif mapped == LABEL_BASELINE:
            color = "#d9d9d9"
            alpha = 0.15
            label = "BASELINE" if not baseline_added else None
            baseline_added = True
        else:
            continue
        ax.axvspan(
            float(segment["start_time_sec"]),
            float(segment["end_time_sec"]),
            color=color,
            alpha=alpha,
            label=label,
        )


def safe_stem(filename: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", Path(filename).stem).strip("_")


def detect_rms_peaks(values: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return np.array([], dtype=int)
    smoothed = pd.Series(values).rolling(window=3, center=True, min_periods=1).mean().to_numpy()
    amplitude_range = float(np.max(smoothed) - np.min(smoothed))
    prominence = max(amplitude_range * 0.15, 1e-6)
    peak_indices, _ = find_peaks(smoothed, prominence=prominence, distance=1)
    return peak_indices


def active_segments(segments: Sequence[Dict[str, Any]], spec) -> List[Dict[str, Any]]:
    return [segment for segment in segments if map_segment_label(segment["label"], spec) == LABEL_ACTIVE]


def build_average_active_profile(
    signals: pd.DataFrame,
    segments: Sequence[Dict[str, Any]],
    spec,
    fs_hz: float,
) -> Dict[str, Any]:
    active = active_segments(segments, spec)
    if not active:
        raise ValueError("Could not find ACTIVE segments for averaging.")

    target_points = 300
    target_axis = np.linspace(0.0, 1.0, target_points)
    durations_sec: List[float] = []
    aggregate_traces: List[np.ndarray] = []
    per_channel_traces: Dict[str, List[np.ndarray]] = {channel: [] for channel in signals.columns}

    for segment in active:
        start_sample = int(segment["start_sample"])
        end_sample = int(segment["end_sample"])
        segment_values = signals.iloc[start_sample : end_sample + 1].to_numpy(dtype=float)
        if len(segment_values) < 4:
            continue
        source_axis = np.linspace(0.0, 1.0, len(segment_values))
        durations_sec.append(float(segment["duration_sec"]))

        aggregate_trace = np.sqrt(np.mean(segment_values ** 2, axis=1))
        aggregate_traces.append(np.interp(target_axis, source_axis, aggregate_trace))

        for channel_index, channel_name in enumerate(signals.columns):
            abs_trace = np.abs(segment_values[:, channel_index])
            per_channel_traces[channel_name].append(np.interp(target_axis, source_axis, abs_trace))

    if not aggregate_traces:
        raise ValueError("ACTIVE segments were too short to build an average profile.")

    median_duration_sec = float(np.median(durations_sec))
    time_axis_sec = target_axis * median_duration_sec

    mean_aggregate = np.mean(np.vstack(aggregate_traces), axis=0)
    std_aggregate = np.std(np.vstack(aggregate_traces), axis=0)
    peak_indices = detect_rms_peaks(mean_aggregate)

    channel_energy = {
        channel_name: float(np.mean([trace.mean() for trace in traces]))
        for channel_name, traces in per_channel_traces.items()
        if traces
    }
    top_channels = sorted(channel_energy, key=channel_energy.get, reverse=True)[:2]

    return {
        "segment_count": len(aggregate_traces),
        "median_duration_sec": median_duration_sec,
        "time_axis_sec": time_axis_sec,
        "mean_aggregate_rms": mean_aggregate,
        "std_aggregate_rms": std_aggregate,
        "peak_indices": peak_indices,
        "top_channels": top_channels,
        "per_channel_mean_abs": {
            channel_name: np.mean(np.vstack(per_channel_traces[channel_name]), axis=0)
            for channel_name in top_channels
        },
        "per_channel_std_abs": {
            channel_name: np.std(np.vstack(per_channel_traces[channel_name]), axis=0)
            for channel_name in top_channels
        },
    }


def build_summary(
    window_df: pd.DataFrame,
    top_mask: pd.Series,
    average_profile: Dict[str, Any],
    selected_channels: Sequence[str],
) -> List[str]:
    active_rest = window_df[window_df["mapped_label"].isin([LABEL_ACTIVE, LABEL_REST])].copy()
    overall_active_fraction = float((active_rest["mapped_label"] == LABEL_ACTIVE).mean())
    top_windows = active_rest.loc[top_mask.loc[active_rest.index]]
    top_active_fraction = float((top_windows["mapped_label"] == LABEL_ACTIVE).mean()) if len(top_windows) else float("nan")

    active_rms = active_rest.loc[active_rest["mapped_label"] == LABEL_ACTIVE, "aggregate_rms"]
    rest_rms = active_rest.loc[active_rest["mapped_label"] == LABEL_REST, "aggregate_rms"]

    lines = [
        f"Selected channels: {', '.join(selected_channels)}",
        f"Overall ACTIVE window fraction (ACTIVE + REST only): {overall_active_fraction:.3f}",
        f"ACTIVE fraction among top-RMS windows: {top_active_fraction:.3f}",
        f"Median aggregate RMS in ACTIVE windows: {float(active_rms.median()):.3f} uV",
        f"Median aggregate RMS in REST windows: {float(rest_rms.median()):.3f} uV",
        f"ACTIVE segments averaged: {int(average_profile['segment_count'])}",
        f"Median ACTIVE duration used for average plot: {float(average_profile['median_duration_sec']):.2f} s",
        f"Broad RMS peaks detected in the average ACTIVE profile: {int(len(average_profile['peak_indices']))}",
    ]
    if len(average_profile["peak_indices"]):
        peak_times = average_profile["time_axis_sec"][average_profile["peak_indices"]].round(2).tolist()
        lines.append(f"Average-profile RMS peak times (segment-relative seconds): {peak_times}")
    return lines


def plot_full_session(
    window_df: pd.DataFrame,
    segments: Sequence[Dict[str, Any]],
    spec,
    top_mask: pd.Series,
    threshold: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 6))
    add_segment_shading(ax, segments, spec)
    ax.plot(
        window_df["center_time_sec"],
        window_df["aggregate_rms"],
        color="#1f4e79",
        linewidth=1.8,
        label="Aggregate RMS",
    )
    top_windows = window_df.loc[top_mask]
    ax.scatter(
        top_windows["center_time_sec"],
        top_windows["aggregate_rms"],
        color="#c00000",
        edgecolor="white",
        linewidth=0.4,
        s=26,
        label="Top RMS windows",
        zorder=3,
    )
    ax.axhline(
        threshold,
        color="#c00000",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        label=f"Top-window threshold ({threshold:.2f} uV)",
    )
    ax.set_title("Left/Right Session RMS Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Aggregate RMS (uV)")
    ax.grid(alpha=0.2)
    legend_handles = [
        Patch(facecolor="#f6b26b", alpha=0.18, label="ACTIVE"),
        Patch(facecolor="#b6d7a8", alpha=0.14, label="REST"),
        Patch(facecolor="#d9d9d9", alpha=0.15, label="BASELINE"),
        Line2D([0], [0], color="#1f4e79", linewidth=1.8, label="Aggregate RMS"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#c00000",
            markeredgecolor="white",
            markersize=7,
            label="Top RMS windows",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_average_active_segment(
    average_profile: Dict[str, Any],
    filename: str,
    output_path: Path,
) -> None:
    time_axis = average_profile["time_axis_sec"]
    peak_indices = average_profile["peak_indices"]
    top_channels = average_profile["top_channels"]

    fig, (ax_raw, ax_rms) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.2]},
    )

    colors = ["#1f77b4", "#d62728"]
    for color, channel in zip(colors, top_channels):
        mean_abs = average_profile["per_channel_mean_abs"][channel]
        std_abs = average_profile["per_channel_std_abs"][channel]
        ax_raw.plot(time_axis, mean_abs, color=color, linewidth=1.4, label=f"{channel} mean |signal|")
        ax_raw.fill_between(time_axis, mean_abs - std_abs, mean_abs + std_abs, color=color, alpha=0.15)
    ax_raw.set_ylabel("Mean absolute amplitude (uV)")
    ax_raw.set_title(f"Average ACTIVE Segment Profile: {filename}")
    ax_raw.grid(alpha=0.2)
    ax_raw.legend(loc="upper right", framealpha=0.95)

    ax_rms.plot(
        time_axis,
        average_profile["mean_aggregate_rms"],
        color="#1f4e79",
        linewidth=2.0,
        label="Mean aggregate RMS",
    )
    ax_rms.fill_between(
        time_axis,
        average_profile["mean_aggregate_rms"] - average_profile["std_aggregate_rms"],
        average_profile["mean_aggregate_rms"] + average_profile["std_aggregate_rms"],
        color="#1f4e79",
        alpha=0.18,
        label="±1 std",
    )
    if len(peak_indices):
        ax_rms.scatter(
            time_axis[peak_indices],
            average_profile["mean_aggregate_rms"][peak_indices],
            color="#c00000",
            s=45,
            zorder=3,
            label="Broad RMS peaks",
        )
    ax_rms.set_xlabel("Segment-relative time (s)")
    ax_rms.set_ylabel("Aggregate RMS (uV)")
    ax_rms.grid(alpha=0.2)
    ax_rms.legend(loc="upper right", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def analyze_session(
    audit: Dict[str, Any],
    spec,
    selected_channels: Sequence[str],
    excluded_channels: Dict[str, str],
    window_sec: float,
    overlap: float,
    top_fraction: float,
    output_dir: Path,
) -> Dict[str, Any]:
    raw_df = load_openbci_csv(Path(audit["file_path"]))
    fs_hz = float(audit["sampling"]["fs_used_hz"])
    signals = preprocess_session_signals(raw_df, "left_right", selected_channels, fs_hz)
    sample_labels = build_sample_labels(len(signals), audit["segments"], spec)
    window_df = compute_rms_windows(signals, sample_labels, fs_hz, window_sec, overlap)
    top_mask, threshold = top_window_mask(window_df, top_fraction)
    average_profile = build_average_active_profile(signals, audit["segments"], spec, fs_hz)

    session_stem = safe_stem(audit["filename"])
    full_plot_path = output_dir / f"rms_vs_time_visualization_{session_stem}.png"
    average_plot_path = output_dir / f"rms_vs_time_average_active_{session_stem}.png"
    windows_csv_path = output_dir / f"rms_vs_time_windows_{session_stem}.csv"
    summary_md_path = output_dir / f"rms_vs_time_summary_{session_stem}.md"

    plot_full_session(window_df, audit["segments"], spec, top_mask, threshold, full_plot_path)
    plot_average_active_segment(average_profile, audit["filename"], average_plot_path)

    export_df = window_df.copy()
    export_df["top_rms_window"] = top_mask.astype(int)
    export_df.to_csv(windows_csv_path, index=False)

    summary_lines = [
        "# RMS vs Time Validation",
        "",
        f"- Session: `{audit['filename']}`",
        f"- Selected channels: `{', '.join(selected_channels)}`",
        f"- Excluded channels from training-screen logic: `{excluded_channels}`",
        f"- Window length: `{window_sec:.2f}` s",
        f"- Overlap: `{overlap:.2f}`",
        f"- Top RMS fraction highlighted: `{top_fraction:.2f}`",
        "",
        "## Summary",
        "",
    ]
    for line in build_summary(window_df, top_mask, average_profile, selected_channels):
        summary_lines.append(f"- {line}")
    write_markdown(summary_md_path, "\n".join(summary_lines))

    active_rest = window_df[window_df["mapped_label"].isin([LABEL_ACTIVE, LABEL_REST])].copy()
    top_windows = active_rest.loc[top_mask.loc[active_rest.index]]
    overall_active_fraction = float((active_rest["mapped_label"] == LABEL_ACTIVE).mean())
    top_active_fraction = float((top_windows["mapped_label"] == LABEL_ACTIVE).mean()) if len(top_windows) else float("nan")
    active_median = float(active_rest.loc[active_rest["mapped_label"] == LABEL_ACTIVE, "aggregate_rms"].median())
    rest_median = float(active_rest.loc[active_rest["mapped_label"] == LABEL_REST, "aggregate_rms"].median())

    return {
        "filename": audit["filename"],
        "selected_channels": list(selected_channels),
        "full_plot_path": full_plot_path,
        "average_plot_path": average_plot_path,
        "windows_csv_path": windows_csv_path,
        "summary_md_path": summary_md_path,
        "top_active_fraction": top_active_fraction,
        "overall_active_fraction": overall_active_fraction,
        "active_median_rms": active_median,
        "rest_median_rms": rest_median,
        "average_active_peak_count": int(len(average_profile["peak_indices"])),
        "average_active_peak_times_sec": average_profile["time_axis_sec"][average_profile["peak_indices"]].round(2).tolist(),
        "active_segment_count": int(average_profile["segment_count"]),
        "average_active_duration_sec": float(average_profile["median_duration_sec"]),
    }


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    audits = audit_all_sessions()
    spec = family_spec("left_right")
    train_audits, _ = split_train_test_audits(audits, "left_right")
    selected_channels, excluded_channels = select_model_channels(train_audits)
    available = [row for row in audits if row["family"] == "left_right"]
    if args.filename:
        requested = set(args.filename)
        selected_audits = [row for row in available if row["filename"] in requested]
        missing = sorted(requested - {row["filename"] for row in selected_audits})
        if missing:
            available_names = [row["filename"] for row in available]
            raise ValueError(f"Could not find session(s) {missing}. Available left/right sessions: {available_names}")
    else:
        selected_audits = available

    all_results: List[Dict[str, Any]] = []
    combined_summary_lines = [
        "# RMS vs Time Validation Across Left/Right Sessions",
        "",
        f"- Selected channels: `{', '.join(selected_channels)}`",
        f"- Excluded channels from training-screen logic: `{excluded_channels}`",
        f"- Window length: `{args.window_sec:.2f}` s",
        f"- Overlap: `{args.overlap:.2f}`",
        f"- Top RMS fraction highlighted: `{args.top_fraction:.2f}`",
        "",
    ]

    for audit in selected_audits:
        result = analyze_session(
            audit=audit,
            spec=spec,
            selected_channels=selected_channels,
            excluded_channels=excluded_channels,
            window_sec=args.window_sec,
            overlap=args.overlap,
            top_fraction=args.top_fraction,
            output_dir=output_dir,
        )
        all_results.append(result)

        combined_summary_lines.extend(
            [
                f"## {audit['filename']}",
                "",
                f"- Full-session RMS plot: `{result['full_plot_path']}`",
                f"- Average ACTIVE profile plot: `{result['average_plot_path']}`",
                f"- Window table: `{result['windows_csv_path']}`",
                f"- High-RMS windows inside ACTIVE: `{result['top_active_fraction']:.3f}` "
                f"(overall ACTIVE rate `{result['overall_active_fraction']:.3f}`)",
                f"- Median aggregate RMS: `ACTIVE={result['active_median_rms']:.3f} uV`, "
                f"`REST={result['rest_median_rms']:.3f} uV`",
                f"- ACTIVE segments averaged: `{result['active_segment_count']}` "
                f"(median duration `{result['average_active_duration_sec']:.2f} s`)",
                f"- Broad RMS peaks in average ACTIVE profile: `{result['average_active_peak_count']}` "
                f"at `{result['average_active_peak_times_sec']}`",
                "",
            ]
        )

    combined_summary_path = output_dir / "rms_vs_time_summary_all_sessions.md"
    write_markdown(combined_summary_path, "\n".join(combined_summary_lines))

    print(f"Selected channels: {selected_channels}")
    print(f"Saved combined summary to {combined_summary_path}")
    print()
    for result in all_results:
        print(f"Session: {result['filename']}")
        print(f"- Saved full-session plot to {result['full_plot_path']}")
        print(f"- Saved average ACTIVE plot to {result['average_plot_path']}")
        print(f"- High-RMS windows inside ACTIVE: {result['top_active_fraction']:.3f} "
              f"(overall ACTIVE rate: {result['overall_active_fraction']:.3f})")
        print(f"- Median aggregate RMS: ACTIVE={result['active_median_rms']:.3f} uV, "
              f"REST={result['rest_median_rms']:.3f} uV")
        print(f"- Broad RMS peaks in average ACTIVE profile: {result['average_active_peak_count']} "
              f"at {result['average_active_peak_times_sec']}")
        print()


if __name__ == "__main__":
    main()
