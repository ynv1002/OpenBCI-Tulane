from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parent / "outputs" / ".mplconfig")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


EVENT_COLORS = {
    "INACTIVE": "#d9d9d9",
    "ONSET": "#ffb703",
    "ACTIVE": "#2a9d8f",
    "OFFSET": "#e76f51",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save representative jaw event plots.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory containing build_event_labels outputs.",
    )
    return parser.parse_args()


def _choose_segments(segment_df: pd.DataFrame) -> dict[str, pd.Series]:
    hold = segment_df[(segment_df["coarse_label"] == "HOLD") & (segment_df["split_role"] == "test")]
    if hold.empty:
        hold = segment_df[segment_df["coarse_label"] == "HOLD"]

    repeated = segment_df[
        (segment_df["coarse_label"] == "REPEATED")
        & (segment_df["detected_event_count"] >= 3)
        & (~segment_df["fallback_used"])
        & (segment_df["split_role"] == "test")
    ]
    if repeated.empty:
        repeated = segment_df[
            (segment_df["coarse_label"] == "REPEATED")
            & (segment_df["detected_event_count"] >= 3)
            & (~segment_df["fallback_used"])
        ]

    ambiguous = segment_df[
        ((segment_df["fallback_used"]) | (segment_df["filename"].str.contains("3-8-26")))
        & (segment_df["coarse_label"] == "REPEATED")
    ]
    if ambiguous.empty:
        ambiguous = segment_df[segment_df["coarse_label"] == "REPEATED"]

    return {
        "hold_clean": hold.iloc[0],
        "repeated_multi": repeated.iloc[0],
        "ambiguous_case": ambiguous.iloc[0],
    }


def _shade_event_states(ax: plt.Axes, segment_samples: pd.DataFrame) -> None:
    start = 0
    labels = segment_samples["event_label"].tolist()
    times = segment_samples["time_sec"].to_numpy()
    current = labels[0]
    for idx in range(1, len(labels)):
        if labels[idx] != current:
            ax.axvspan(times[start], times[idx - 1], color=EVENT_COLORS.get(current, "#cccccc"), alpha=0.18)
            start = idx
            current = labels[idx]
    ax.axvspan(times[start], times[-1], color=EVENT_COLORS.get(current, "#cccccc"), alpha=0.18)


def _plot_segment(segment_row: pd.Series, sample_df: pd.DataFrame, output_path: Path) -> None:
    segment_samples = sample_df[
        (sample_df["filename"] == segment_row["filename"])
        & (sample_df["sample_row"] >= segment_row["start_sample"])
        & (sample_df["sample_row"] <= segment_row["end_sample"])
    ].copy()
    signal_columns = [column for column in segment_samples.columns if column.startswith("signal_Channel_")]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(segment_samples["time_sec"], segment_samples["aggregate_rms"], color="#4d4d4d", alpha=0.5, label="aggregate_rms")
    axes[0].plot(
        segment_samples["time_sec"],
        segment_samples["aggregate_rms_smooth"],
        color="#1f77b4",
        linewidth=1.6,
        label="aggregate_rms_smooth",
    )
    axes[0].axhline(float(segment_samples["active_threshold"].iloc[0]), color="#2a9d8f", linestyle="--", label="active_threshold")
    axes[0].axhline(float(segment_samples["release_threshold"].iloc[0]), color="#e76f51", linestyle=":", label="release_threshold")
    _shade_event_states(axes[0], segment_samples)
    axes[0].set_ylabel("Envelope (uV-like)")
    axes[0].legend(loc="upper right", ncol=4)
    axes[0].set_title(
        f"{segment_row['filename']} | {segment_row['coarse_label']} | mode={segment_row['detection_mode']} | events={segment_row['detected_event_count']}"
    )

    for signal_column in signal_columns:
        axes[1].plot(segment_samples["time_sec"], segment_samples[signal_column], linewidth=0.9, alpha=0.8, label=signal_column.replace("signal_", ""))
    _shade_event_states(axes[1], segment_samples)
    axes[1].set_ylabel("Filtered channel")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="upper right", ncol=3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    sample_df = pd.read_csv(input_dir / "jaw_event_samples.csv")
    segment_df = pd.read_csv(input_dir / "jaw_event_segment_summary.csv")
    chosen = _choose_segments(segment_df)

    outputs = {
        "hold_clean": input_dir / "jaw_event_example_hold_clean.png",
        "repeated_multi": input_dir / "jaw_event_example_repeated_multi.png",
        "ambiguous_case": input_dir / "jaw_event_example_ambiguous.png",
    }
    for key, row in chosen.items():
        _plot_segment(row, sample_df, outputs[key])
        print(f"Wrote {outputs[key]}")


if __name__ == "__main__":
    main()
