from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.event_utils import JawEventConfig, build_jaw_event_label_bundle, summarize_jaw_event_labels
    from analysis.utils import audit_all_sessions, dataframe_to_markdown, ensure_output_dir, write_json
else:
    from .event_utils import JawEventConfig, build_jaw_event_label_bundle, summarize_jaw_event_labels
    from .utils import audit_all_sessions, dataframe_to_markdown, ensure_output_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sample-level jaw event labels.")
    parser.add_argument("--smoothing-sec", type=float, default=JawEventConfig().smoothing_sec)
    parser.add_argument("--onset-sec", type=float, default=JawEventConfig().onset_duration_sec)
    parser.add_argument("--offset-sec", type=float, default=JawEventConfig().offset_duration_sec)
    parser.add_argument(
        "--minimum-active-sec",
        type=float,
        default=JawEventConfig().minimum_active_duration_sec,
    )
    parser.add_argument(
        "--minimum-peak-distance-sec",
        type=float,
        default=JawEventConfig().minimum_peak_distance_sec,
    )
    parser.add_argument("--inactive-quantile", type=float, default=JawEventConfig().inactive_quantile)
    parser.add_argument("--active-quantile", type=float, default=JawEventConfig().active_quantile)
    parser.add_argument("--threshold-mix", type=float, default=JawEventConfig().threshold_mix)
    parser.add_argument(
        "--release-threshold-mix",
        type=float,
        default=JawEventConfig().release_threshold_mix,
    )
    parser.add_argument(
        "--minimum-reference-gap-uv",
        type=float,
        default=JawEventConfig().minimum_reference_gap_uv,
    )
    parser.add_argument(
        "--peak-prominence-scale",
        type=float,
        default=JawEventConfig().peak_prominence_scale,
    )
    parser.add_argument(
        "--minimum-peak-prominence-uv",
        type=float,
        default=JawEventConfig().minimum_peak_prominence_uv,
    )
    parser.add_argument("--manual-active-threshold-uv", type=float, default=None)
    parser.add_argument("--manual-release-threshold-uv", type=float, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for event-label outputs.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> JawEventConfig:
    return JawEventConfig(
        smoothing_sec=args.smoothing_sec,
        onset_duration_sec=args.onset_sec,
        offset_duration_sec=args.offset_sec,
        minimum_active_duration_sec=args.minimum_active_sec,
        minimum_peak_distance_sec=args.minimum_peak_distance_sec,
        inactive_quantile=args.inactive_quantile,
        active_quantile=args.active_quantile,
        threshold_mix=args.threshold_mix,
        release_threshold_mix=args.release_threshold_mix,
        minimum_reference_gap_uv=args.minimum_reference_gap_uv,
        peak_prominence_scale=args.peak_prominence_scale,
        minimum_peak_prominence_uv=args.minimum_peak_prominence_uv,
        manual_active_threshold_uv=args.manual_active_threshold_uv,
        manual_release_threshold_uv=args.manual_release_threshold_uv,
    )


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    config = build_config(args)
    audits = audit_all_sessions()

    bundle = build_jaw_event_label_bundle(audits, config)
    summary = summarize_jaw_event_labels(bundle)

    sample_csv = output_dir / "jaw_event_samples.csv"
    interval_csv = output_dir / "jaw_event_intervals.csv"
    segment_csv = output_dir / "jaw_event_segment_summary.csv"
    threshold_csv = output_dir / "jaw_event_thresholds.csv"
    summary_json = output_dir / "jaw_event_summary.json"
    summary_md = output_dir / "jaw_event_summary.md"

    bundle["sample_frame"].to_csv(sample_csv, index=False)
    bundle["interval_frame"].to_csv(interval_csv, index=False)
    bundle["coarse_segment_frame"].to_csv(segment_csv, index=False)
    bundle["threshold_frame"].to_csv(threshold_csv, index=False)

    payload = {
        "selected_channels": bundle["selected_channels"],
        "excluded_channels": bundle["excluded_channels"],
        "train_files": [audit["filename"] for audit in bundle["train_audits"]],
        "test_file": bundle["test_audit"]["filename"],
        "config": asdict(config),
        "summary": summary,
    }
    write_json(summary_json, payload)

    event_counts_df = pd.DataFrame(
        sorted(summary["event_state_sample_counts"].items()), columns=["event_label", "sample_count"]
    )
    repeated_df = bundle["coarse_segment_frame"]
    repeated_df = repeated_df[repeated_df["coarse_label"] == "REPEATED"][
        [
            "filename",
            "coarse_segment_index",
            "duration_sec",
            "detection_mode",
            "detected_peak_count",
            "detected_event_count",
            "fallback_used",
            "notes",
        ]
    ].reset_index(drop=True)

    lines = [
        "# Jaw Event Labels",
        "",
        f"- Train files: `{', '.join(payload['train_files'])}`",
        f"- Test file: `{payload['test_file']}`",
        f"- Selected channels: `{bundle['selected_channels']}`",
        f"- Excluded channels: `{' | '.join(f'{key}: {value}' for key, value in bundle['excluded_channels'].items())}`",
        f"- Config: `{config}`",
        "",
        "## Event State Counts",
        "",
        dataframe_to_markdown(event_counts_df, include_index=False),
        "",
        "## Repeated Segment Detection",
        "",
        f"- Repeated segments: `{summary['repeated_segment_count']}`",
        f"- Repeated fallback count: `{summary['repeated_fallback_count']}`",
        f"- Total detected repeated events: `{summary['repeated_detected_event_count_total']}`",
        f"- Median detected events per repeated segment: `{summary['repeated_detected_event_count_median']:.2f}`",
        "",
        dataframe_to_markdown(repeated_df.head(20), include_index=False),
        "",
    ]
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    pd.set_option("display.max_colwidth", 120)
    print(event_counts_df.to_string(index=False))
    print("\nRepeated detection summary:")
    print(repeated_df.head(20).to_string(index=False))
    print(f"\nWrote {sample_csv}")
    print(f"Wrote {interval_csv}")
    print(f"Wrote {segment_csv}")
    print(f"Wrote {threshold_csv}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
