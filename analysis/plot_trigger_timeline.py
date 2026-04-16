from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

DEFAULT_MPLCONFIGDIR = Path(__file__).resolve().parent / "outputs" / ".mplconfig"
DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.utils import ensure_output_dir
else:
    from .utils import ensure_output_dir


COARSE_COLORS = {
    "BASELINE": "#bdbdbd",
    "REST": "#e0e0e0",
    "HOLD": "#c8e6c9",
    "REPEATED": "#ffe0b2",
}

EVENT_COLORS = {
    "INACTIVE": "#f0f0f0",
    "ONSET": "#9ecae1",
    "ACTIVE": "#a1d99b",
    "OFFSET": "#fcae91",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot replay trigger timelines for representative jaw segments.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory containing replay outputs.",
    )
    return parser.parse_args()


def _choose_segments(segment_df: pd.DataFrame) -> dict:
    test_file = "HR-3-15-26-(03).csv"
    hold_clean = segment_df[(segment_df["filename"] == test_file) & (segment_df["coarse_label"] == "HOLD")].head(1)
    repeated_multi = segment_df[
        (segment_df["filename"] == test_file)
        & (segment_df["coarse_label"] == "REPEATED")
        & (segment_df["fallback_used"] == False)
        & (segment_df["detected_event_count"] >= 3)
    ].head(1)
    ambiguous = segment_df[(segment_df["coarse_label"] == "REPEATED") & (segment_df["fallback_used"] == True)].head(1)
    return {
        "hold_clean": hold_clean.iloc[0].to_dict() if not hold_clean.empty else None,
        "repeated_multi": repeated_multi.iloc[0].to_dict() if not repeated_multi.empty else None,
        "ambiguous_case": ambiguous.iloc[0].to_dict() if not ambiguous.empty else None,
    }


def _add_label_spans(ax, frame: pd.DataFrame, label_column: str, color_map: dict, alpha: float) -> None:
    labels = frame[label_column].to_numpy()
    times = frame["time_sec"].to_numpy(dtype=float)
    if len(times) == 0:
        return
    start_index = 0
    current_label = str(labels[0])
    for index in range(1, len(labels)):
        if labels[index] != current_label:
            if current_label in color_map:
                ax.axvspan(times[start_index], times[index - 1], color=color_map[current_label], alpha=alpha, linewidth=0.0)
            start_index = index
            current_label = str(labels[index])
    if current_label in color_map:
        ax.axvspan(times[start_index], times[-1], color=color_map[current_label], alpha=alpha, linewidth=0.0)


def _plot_segment(
    sample_df: pd.DataFrame,
    score_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    click_df: pd.DataFrame,
    thresholds_df: pd.DataFrame,
    segment_row: dict,
    title: str,
    output_path: Path,
) -> None:
    margin_sec = 0.75
    filename = segment_row["filename"]
    start_sec = float(segment_row["start_time_sec"]) - margin_sec
    end_sec = float(segment_row["end_time_sec"]) + margin_sec

    sample_slice = sample_df[
        (sample_df["filename"] == filename)
        & (sample_df["time_sec"] >= start_sec)
        & (sample_df["time_sec"] <= end_sec)
    ].copy()
    score_slice = score_df[
        (score_df["filename"] == filename)
        & (score_df["center_time_sec"] >= start_sec)
        & (score_df["center_time_sec"] <= end_sec)
    ].copy()
    reference_slice = reference_df[
        (reference_df["filename"] == filename)
        & (reference_df["reference_onset_time_sec"] >= start_sec)
        & (reference_df["reference_onset_time_sec"] <= end_sec)
    ].copy()
    click_slice = click_df[
        (click_df["filename"] == filename)
        & (click_df["center_time_sec"] >= start_sec)
        & (click_df["center_time_sec"] <= end_sec)
    ].copy()

    threshold_row = thresholds_df[thresholds_df["filename"] == filename].iloc[0]

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    ax_signal, ax_probs, ax_labels = axes

    _add_label_spans(ax_signal, sample_slice, "coarse_label", COARSE_COLORS, alpha=0.18)
    _add_label_spans(ax_signal, sample_slice, "event_label", EVENT_COLORS, alpha=0.08)
    ax_signal.plot(sample_slice["time_sec"], sample_slice["aggregate_rms"], color="#636363", linewidth=0.8, alpha=0.7, label="aggregate_rms")
    ax_signal.plot(
        sample_slice["time_sec"],
        sample_slice["aggregate_rms_smooth"],
        color="#08519c",
        linewidth=1.6,
        label="aggregate_rms_smooth",
    )
    ax_signal.axhline(float(threshold_row["active_threshold_uv"]), color="#238b45", linestyle="--", linewidth=1.0, label="active_threshold")
    ax_signal.axhline(float(threshold_row["release_threshold_uv"]), color="#cb181d", linestyle="--", linewidth=1.0, label="release_threshold")
    for _, row in reference_slice.iterrows():
        ax_signal.axvline(float(row["reference_onset_time_sec"]), color="#6a3d9a", linestyle=":", linewidth=1.2, alpha=0.9)
    for _, row in click_slice.iterrows():
        ax_signal.axvline(float(row["center_time_sec"]), color="#ff7f00", linestyle="-", linewidth=1.2, alpha=0.9)
    ax_signal.set_ylabel("Jaw Envelope (uV)")
    ax_signal.set_title(title)
    ax_signal.legend(loc="upper right", ncol=2, fontsize=8)

    ax_probs.plot(score_slice["center_time_sec"], score_slice["prob_binary_CLENCH"], color="#1b9e77", label="P(CLENCH)")
    ax_probs.plot(score_slice["center_time_sec"], score_slice["prob_4state_ONSET"], color="#7570b3", label="P(ONSET)")
    ax_probs.plot(score_slice["center_time_sec"], score_slice["prob_4state_ACTIVE"], color="#e7298a", label="P(ACTIVE)")
    for _, row in click_slice.iterrows():
        ax_probs.axvline(float(row["center_time_sec"]), color="#ff7f00", linestyle="-", linewidth=1.0, alpha=0.85)
    ax_probs.set_ylim(0.0, 1.05)
    ax_probs.set_ylabel("Model Prob.")
    ax_probs.legend(loc="upper right", fontsize=8)

    _add_label_spans(ax_labels, sample_slice, "coarse_label", COARSE_COLORS, alpha=0.45)
    _add_label_spans(ax_labels, sample_slice, "event_label", EVENT_COLORS, alpha=0.25)
    ax_labels.set_yticks([])
    ax_labels.set_ylabel("Labels")
    for _, row in reference_slice.iterrows():
        ax_labels.axvline(float(row["reference_onset_time_sec"]), color="#6a3d9a", linestyle=":", linewidth=1.1)
    for _, row in click_slice.iterrows():
        ax_labels.axvline(float(row["center_time_sec"]), color="#ff7f00", linestyle="-", linewidth=1.1)
    ax_labels.set_xlabel("Time (s)")

    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    best_summary = json.loads((output_dir / "jaw_click_best_strategy.json").read_text(encoding="utf-8"))
    strategy_name = best_summary["best_strategy"]["strategy_name"]

    sample_df = pd.read_csv(output_dir / "jaw_event_samples.csv")
    segment_df = pd.read_csv(output_dir / "jaw_event_segment_summary.csv")
    thresholds_df = pd.read_csv(output_dir / "jaw_event_thresholds.csv")
    score_df = pd.read_csv(output_dir / "jaw_click_replay_scores.csv")
    reference_df = pd.read_csv(output_dir / "jaw_click_reference_onsets.csv")
    click_df = pd.read_csv(output_dir / f"jaw_click_events_{strategy_name}.csv")
    segments = _choose_segments(segment_df)

    saved_paths = []
    for key, segment_row in segments.items():
        if segment_row is None:
            continue
        output_path = output_dir / f"jaw_trigger_timeline_{key}_{strategy_name}.png"
        _plot_segment(
            sample_df,
            score_df,
            reference_df,
            click_df,
            thresholds_df,
            segment_row,
            title=f"{key} | {strategy_name} | {segment_row['filename']}",
            output_path=output_path,
        )
        saved_paths.append(output_path)
        print(f"Saved {output_path}")

    if not saved_paths:
        print("No representative segments were available for plotting.")


if __name__ == "__main__":
    main()
