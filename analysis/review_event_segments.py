from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.utils import dataframe_to_markdown, ensure_output_dir
else:
    from .utils import dataframe_to_markdown, ensure_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review saved jaw event segment summaries.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory containing build_event_labels outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = ensure_output_dir(args.input_dir)

    segment_df = pd.read_csv(input_dir / "jaw_event_segment_summary.csv")
    threshold_df = pd.read_csv(input_dir / "jaw_event_thresholds.csv")
    sample_df = pd.read_csv(input_dir / "jaw_event_samples.csv", usecols=["filename", "event_label", "coarse_label"])

    per_file_state_df = (
        sample_df.groupby(["filename", "event_label"]).size().reset_index(name="sample_count")
    )
    repeated_df = (
        segment_df[segment_df["coarse_label"] == "REPEATED"]
        .groupby("filename")[["detected_event_count", "fallback_used"]]
        .agg({"detected_event_count": "sum", "fallback_used": "sum"})
        .reset_index()
    )
    review_md = input_dir / "jaw_event_review.md"

    lines = [
        "# Jaw Event Review",
        "",
        "## Thresholds",
        "",
        dataframe_to_markdown(
            threshold_df[
                [
                    "filename",
                    "inactive_reference_uv",
                    "active_reference_uv",
                    "active_threshold_uv",
                    "release_threshold_uv",
                    "peak_prominence_uv",
                ]
            ].round(3),
            include_index=False,
        ),
        "",
        "## Repeated Segment Summary",
        "",
        dataframe_to_markdown(repeated_df, include_index=False),
        "",
        "## Per-File Event Counts",
        "",
        dataframe_to_markdown(per_file_state_df, include_index=False),
        "",
    ]
    review_md.write_text("\n".join(lines), encoding="utf-8")

    print(threshold_df.round(3).to_string(index=False))
    print("\nRepeated segment summary:")
    print(repeated_df.to_string(index=False))
    print(f"\nWrote {review_md}")


if __name__ == "__main__":
    main()
