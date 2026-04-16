from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import pandas as pd

from audit_utils import (
    assess_raw_signal_scale,
    build_sanity_window,
    dataframe_to_markdown,
    ensure_output_dir,
    load_emg_csv,
    write_json,
)
from config import AuditConfig
from metrics import build_mode_comparison, compute_channel_diagnostics, per_label_summary
from plots import save_all_plots
from signal_stages import OpenBCIEMGJoystick1DProcessor, build_mode_signals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the OpenBCI EMG processing audit across multiple preprocessing modes."
    )
    parser.add_argument("--csv", type=Path, default=AuditConfig().csv_path, help="Input CSV path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=AuditConfig().output_dir,
        help="Directory for audit outputs.",
    )
    parser.add_argument("--fs", type=float, default=AuditConfig().fs_hz, help="Sampling rate in Hz.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AuditConfig:
    base = AuditConfig()
    return AuditConfig(
        csv_path=args.csv,
        output_dir=args.output_dir,
        fs_hz=args.fs,
        label_column=base.label_column,
        all_channels=base.all_channels,
        tracked_channels=base.tracked_channels,
        channel_map=base.channel_map,
        aggregation_methods=base.aggregation_methods,
        modes=base.modes,
        mode_labels=base.mode_labels,
        openbci=base.openbci,
        filtering=base.filtering,
        plots=base.plots,
        count_to_uv=base.count_to_uv,
        sample_rate_note=base.sample_rate_note,
        supported_labels=base.supported_labels,
        active_labels=base.active_labels,
        direction_thresholds=base.direction_thresholds,
        activation_thresholds=base.activation_thresholds,
        sanity_channel=base.sanity_channel,
        sanity_window_samples=base.sanity_window_samples,
        suspicious_raw_rail=base.suspicious_raw_rail,
        raw_rail_nominal=base.raw_rail_nominal,
        near_zero_normalized_threshold=base.near_zero_normalized_threshold,
        near_one_normalized_threshold=base.near_one_normalized_threshold,
    )


def write_audit_summary(
    path: Path,
    config: AuditConfig,
    scale_assessment: Dict[str, object],
    filter_metadata: Dict[str, object],
    diagnostics_df: pd.DataFrame,
    mode_comparison_df: pd.DataFrame,
    summary_payload: Dict[str, object],
) -> None:
    best_direction = summary_payload["mode_summary"]["best_direction_mode_method"]
    best_sensible = summary_payload["mode_summary"]["best_processor_sensibility_mode_method"]
    best_jaws = summary_payload["mode_summary"]["best_jaws_center_mode_method"]
    raw_best = mode_comparison_df.loc[
        (mode_comparison_df["mode"] == "raw") & (mode_comparison_df["method"] == "mean")
    ].iloc[0]
    filtered_best = mode_comparison_df.loc[
        (mode_comparison_df["mode"] == "filtered") & (mode_comparison_df["method"] == "mean")
    ].iloc[0]

    flagged_channels = diagnostics_df.loc[
        diagnostics_df["mode"] == "raw",
        [
            "channel",
            "flag_suspicious_saturation",
            "flag_near_constant",
            "flag_unusually_noisy",
            "flag_large_drift",
        ],
    ].copy()

    preprocessing_helped = filtered_best["direction_macro_f1"] > raw_best["direction_macro_f1"] + 0.05
    mapping_suspicious = not bool(best_direction.get("expected_direction_order", True))
    filtered_desc = filter_metadata["filtered"]["filtering"]
    if isinstance(filtered_desc, dict):
        filtered_desc_text = ", ".join(filtered_desc.get("pipeline", []))
        filtered_desc_text = f"{filtered_desc.get('filter_backend', 'unknown')}: {filtered_desc_text}"
    else:
        filtered_desc_text = str(filtered_desc)

    lines = [
        "# OpenBCI EMG Processing Audit",
        "",
        "## Why This Audit Exists",
        "",
        "The original baseline applied the OpenBCI-style EMG joystick logic closely, but left/right results were weak.",
        "This audit checks whether the weakness is explained by preprocessing and signal assumptions before the algorithm itself is blamed or the collection protocol is redesigned.",
        "",
        "## Input And Assumptions",
        "",
        f"- CSV: `{config.csv_path}`",
        f"- Sample rate: `{config.fs_hz:.2f} Hz`",
        f"- Sample-rate note: {config.sample_rate_note}",
        f"- Tracked channels: `{', '.join(config.tracked_channels)}`",
        f"- Left mapping: `{', '.join(config.channel_map['left'])}`",
        f"- Right mapping: `{', '.join(config.channel_map['right'])}`",
        f"- Count-to-uV assumption tested: `{config.count_to_uv:.5f} uV/count`",
        "",
        "## Audit Questions",
        "",
        f"1. Likely input type: `{scale_assessment['likely_signal_kind']}`.",
        f"2. `0.02235 uV/count` plausible: `{scale_assessment['count_to_uv_plausible']}`.",
        f"3. Channels near the nominal rail: `{', '.join(scale_assessment['channels_near_rail']) if scale_assessment['channels_near_rail'] else 'none'}`.",
        f"4. Filter pipeline used for Mode D: `{filtered_desc_text}`.",
        "",
        "## High-Level Findings",
        "",
        f"- Best Jleft vs Jright separation in this audit: mode `{best_direction['mode']}` with `{best_direction['method']}` aggregation "
        f"(macro F1 `{best_direction['direction_macro_f1']:.4f}`, ROC AUC `{best_direction['roc_auc_x_smooth']:.4f}`).",
        f"- Most sensible OpenBCI threshold behavior by heuristic: mode `{best_sensible['mode']}` with `{best_sensible['method']}` aggregation "
        f"(clip fraction `{best_sensible['average_uv_clip_fraction_mean']:.4f}`, normalized mid fraction `{best_sensible['normalized_mid_fraction_mean']:.4f}`).",
        f"- Best Jaws near-center behavior among non-degenerate modes: mode `{best_jaws['mode']}` with `{best_jaws['method']}` aggregation "
        f"(Jaws center score `{best_jaws['jaws_center_score']:.4f}`).",
        "",
        "## Preprocessing Interpretation",
        "",
        (
            "- Filtered preprocessing materially improved the left/right threshold metric over raw mode, which is evidence that preprocessing was part of the failure."
            if preprocessing_helped
            else "- Filtered preprocessing did not materially rescue the left/right metric over raw mode, which suggests preprocessing is not the whole explanation."
        ),
        (
            "- The best-performing mode still has a suspicious direction ordering, so channel mapping or label-side alignment should remain under suspicion."
            if mapping_suspicious
            else "- The best-performing mode preserves the expected left/right sign ordering."
        ),
        "- If all modes remain weak after thresholds and normalization look reasonable, that points more toward dataset/protocol mismatch than simple preprocessing failure.",
        "",
        "## Raw Channel Flags",
        "",
        dataframe_to_markdown(flagged_channels, include_index=False),
        "",
        "## Mode Comparison",
        "",
        dataframe_to_markdown(
            mode_comparison_df[
                [
                    "mode",
                    "method",
                    "direction_macro_f1",
                    "roc_auc_x_smooth",
                    "average_uv_clip_fraction_mean",
                    "normalized_mid_fraction_mean",
                    "activation_macro_f1",
                    "jaws_center_score",
                ]
            ].round(4),
            include_index=False,
        ),
        "",
        "## What To Inspect First",
        "",
        "1. `audit_summary.md` for the mode ranking and interpretation.",
        "2. `mode_comparison.csv` for the numeric comparison across modes.",
        "3. `tracked_channels_filtered.png`, `threshold_dynamics_filtered.png`, and `normalized_outputs_filtered.png` to see whether filtering makes the processor behave more sensibly.",
        "4. `x_smooth_histograms_mean.png` and `activation_scatter_mean.png` to compare label structure across modes.",
        "5. `sanity_window_ch1.csv` to hand-check stage values and threshold updates over a short sample window.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = build_config(args)
    output_dir = ensure_output_dir(config.output_dir)

    raw_df = load_emg_csv(config.csv_path, config)
    scale_assessment = assess_raw_signal_scale(raw_df, config)
    mode_signals, mode_metadata = build_mode_signals(raw_df, config)

    processed_by_mode: Dict[str, pd.DataFrame] = {}
    label_summaries = []

    for mode, signal_df in mode_signals.items():
        processor_input = raw_df[["sample_index", "time_sec", "label_original", "label_clean"]].copy()
        processor_input["mode"] = mode
        processor_input["signal_units"] = mode_metadata[mode]["units"]
        for channel in config.all_channels:
            processor_input[f"signal_{channel}"] = signal_df[channel].to_numpy()
        for channel in config.tracked_channels:
            processor_input[f"{channel}_signal_uv"] = signal_df[channel].to_numpy()

        processor = OpenBCIEMGJoystick1DProcessor(config)
        processor_outputs = processor.process(processor_input)
        processed_df = pd.concat([processor_input, processor_outputs], axis=1)
        processed_df.to_csv(output_dir / f"processed_{mode}.csv", index=False)
        processed_by_mode[mode] = processed_df
        label_summaries.append(per_label_summary(processed_df, config))

    channel_diagnostics = compute_channel_diagnostics(raw_df, mode_signals, config)
    channel_diagnostics.to_csv(output_dir / "channel_diagnostics.csv", index=False)

    label_summary_df = pd.concat(label_summaries, ignore_index=True)
    label_summary_df.to_csv(output_dir / "label_level_summary.csv", index=False)

    mode_comparison_df, mode_summary = build_mode_comparison(processed_by_mode, config)
    mode_comparison_df.to_csv(output_dir / "mode_comparison.csv", index=False)

    for mode, payload in mode_summary["per_mode"].items():
        for method, method_payload in payload["methods"].items():
            method_payload["direction_scan"].to_csv(
                output_dir / f"{mode}_{method}_direction_threshold_scan.csv", index=False
            )
            method_payload["activation_scan"].to_csv(
                output_dir / f"{mode}_{method}_activation_threshold_scan.csv", index=False
            )
            method_payload["direction_confusion"].to_csv(
                output_dir / f"{mode}_{method}_jleft_jright_confusion.csv"
            )
            method_payload["activation_confusion"].to_csv(
                output_dir / f"{mode}_{method}_norm_active_confusion.csv"
            )

    sanity_window_df = build_sanity_window(raw_df, mode_signals, processed_by_mode, config)
    sanity_window_df.to_csv(output_dir / f"sanity_window_{config.sanity_channel}.csv", index=False)

    plot_paths = []
    if not args.skip_plots:
        plot_paths = [
            str(path)
            for path in save_all_plots(
                raw_df, mode_signals, processed_by_mode, channel_diagnostics, config, output_dir
            )
        ]

    summary_payload = {
        "config": asdict(config),
        "scale_assessment": scale_assessment,
        "mode_metadata": mode_metadata,
        "mode_summary": mode_summary,
        "plot_paths": plot_paths,
        "output_files": {
            "channel_diagnostics_csv": str(output_dir / "channel_diagnostics.csv"),
            "mode_comparison_csv": str(output_dir / "mode_comparison.csv"),
            "sanity_window_csv": str(output_dir / f"sanity_window_{config.sanity_channel}.csv"),
        },
    }
    write_json(output_dir / "audit_summary.json", summary_payload)
    write_audit_summary(
        output_dir / "audit_summary.md",
        config,
        scale_assessment,
        mode_metadata,
        channel_diagnostics,
        mode_comparison_df,
        summary_payload,
    )


if __name__ == "__main__":
    main()
