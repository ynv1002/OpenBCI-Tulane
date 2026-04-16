from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import pandas as pd

from config import ExperimentConfig
from metrics import (
    label_counts,
    left_right_separability,
    per_label_summary,
    scan_activation_thresholds,
    scan_direction_thresholds,
    summarize_jaws,
)
from openbci_emg import OpenBCIEMGJoystick1DProcessor
from plots import save_all_plots
from utils import dataframe_to_markdown, ensure_output_dir, infer_scale_and_transform, load_emg_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the OpenBCI-style 1D EMG joystick baseline experiment."
    )
    parser.add_argument("--csv", type=Path, default=ExperimentConfig().csv_path, help="Input CSV path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ExperimentConfig().output_dir,
        help="Directory for processed outputs.",
    )
    parser.add_argument("--fs", type=float, default=ExperimentConfig().fs_hz, help="Sampling rate in Hz.")
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip PNG generation and only write tabular outputs.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    base = ExperimentConfig()
    return ExperimentConfig(
        csv_path=args.csv,
        output_dir=args.output_dir,
        fs_hz=args.fs,
        sample_rate_note=base.sample_rate_note,
        label_column=base.label_column,
        tracked_channels=base.tracked_channels,
        channel_map=base.channel_map,
        aggregation_methods=base.aggregation_methods,
        primary_aggregation=base.primary_aggregation,
        openbci=base.openbci,
        scaling=base.scaling,
        plots=base.plots,
        direction_thresholds=base.direction_thresholds,
        activation_thresholds=base.activation_thresholds,
        keep_aux_columns=base.keep_aux_columns,
        supported_labels=base.supported_labels,
        active_labels=base.active_labels,
    )


def write_markdown_report(
    path: Path,
    config: ExperimentConfig,
    scale_report: Dict[str, object],
    counts: Dict[str, int],
    label_summary_df: pd.DataFrame,
    summary_payload: Dict[str, object],
) -> None:
    lines = [
        "# OpenBCI EMG Left/Right Baseline Report",
        "",
        "## Framing",
        "",
        "This is a first-pass offline baseline that applies OpenBCI-style EMG joystick logic to an older hold-based dataset.",
        "The dataset contains sustained `Jaws`, `Jleft`, and `Jright` holds, which is only a partial fit for joystick-style directional control.",
        "That mismatch matters: OpenBCI joystick logic is usually more natural with distinct dynamic left/right motions than long sustained holds.",
        "This report should therefore be read as an algorithm-transfer baseline, not as a statement that the current collection protocol is already ideal.",
        "",
        "## Input And Assumptions",
        "",
        f"- CSV: `{config.csv_path}`",
        f"- Sample rate: `{config.fs_hz:.2f} Hz`",
        f"- Sample-rate note: {config.sample_rate_note}",
        f"- Left channels mapped to X-: `{', '.join(config.channel_map['left'])}`",
        f"- Right channels mapped to X+: `{', '.join(config.channel_map['right'])}`",
        f"- Aggregations compared: `{', '.join(config.aggregation_methods)}`",
        f"- OpenBCI window: `{config.openbci.window_seconds:.2f} s`",
        f"- OpenBCI uvLimit: `{config.openbci.uv_limit:.2f} uV`",
        f"- OpenBCI smoothing: `{config.openbci.smoothing:.2f}`",
        "",
        "## Scale Inspection",
        "",
        f"- Applied scale mode: `{scale_report['mode_applied']}`",
        f"- Scale comment: {scale_report['data_scale_comment']}",
        f"- Detection reasons: {', '.join(scale_report['detection_reasons']) if scale_report['detection_reasons'] else 'none'}",
        "",
        "## Sample Counts",
        "",
    ]

    counts_df = pd.DataFrame({"label": list(counts.keys()), "count": list(counts.values())})
    lines.append(dataframe_to_markdown(counts_df, include_index=False))
    lines.extend(
        [
            "",
            "## Per-Label Summary",
            "",
            dataframe_to_markdown(label_summary_df.round(4), include_index=False),
            "",
            "## Aggregation Results",
            "",
        ]
    )

    for method in config.aggregation_methods:
        method_summary = summary_payload["methods"][method]
        direction_confusion_df = pd.DataFrame.from_dict(
            method_summary["direction_confusion"], orient="index"
        )
        activation_confusion_df = pd.DataFrame.from_dict(
            method_summary["activation_confusion"], orient="index"
        )
        lines.extend(
            [
                f"### `{method}` aggregation",
                "",
                f"- Jleft vs Jright separability: Cohen's d `{method_summary['separability']['cohens_d_x_smooth']:.4f}`, "
                f"overlap `{method_summary['separability']['empirical_overlap_x_smooth']:.4f}`, "
                f"ROC AUC `{method_summary['separability']['roc_auc_x_smooth']:.4f}`.",
                f"- Best directional threshold on `x_smooth`: `{method_summary['direction_threshold_best']['threshold']:.2f}` "
                f"(macro F1 `{method_summary['direction_threshold_best']['macro_f1']:.4f}`, "
                f"coverage `{method_summary['direction_threshold_best']['coverage']:.4f}`, "
                f"accuracy-all `{method_summary['direction_threshold_best']['accuracy_all']:.4f}`).",
                f"- Best norm vs active threshold on `total_activation`: `{method_summary['activation_threshold_best']['threshold']:.2f}` "
                f"(macro F1 `{method_summary['activation_threshold_best']['macro_f1']:.4f}`, "
                f"accuracy `{method_summary['activation_threshold_best']['accuracy']:.4f}`).",
                f"- Jaws summary: mean total activation `{method_summary['jaws_summary']['jaws_mean_total_activation']:.4f}`, "
                f"mean abs direction `{method_summary['jaws_summary']['jaws_mean_abs_direction']:.4f}`, "
                f"activation-vs-norm ratio `{method_summary['jaws_summary']['jaws_total_vs_norm_ratio']:.4f}`, "
                f"abs-direction-vs-lateral ratio `{method_summary['jaws_summary']['jaws_abs_direction_vs_lateral_ratio']:.4f}`.",
                "",
                "Directional confusion table:",
                "",
                dataframe_to_markdown(direction_confusion_df, include_index=True),
                "",
                "Norm vs active confusion table:",
                "",
                dataframe_to_markdown(activation_confusion_df, include_index=True),
                "",
            ]
        )

    lines.extend(
        [
            "## What To Inspect First",
            "",
            "1. `summary_report.md` for the scale assumption, threshold scan winners, and Jaws behavior.",
            "2. `x_smooth_histograms.png` to see whether `Jleft`, `Jright`, `Jaws`, and `norm` separate cleanly.",
            "3. `activation_direction_scatter.png` to see whether `Jaws` is active but near-center.",
            "4. `x_raw_x_smooth_window.png` and `tracked_channel_thresholds_window.png` to see whether the adaptive thresholds behave sensibly during label transitions.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = build_config(args)
    output_dir = ensure_output_dir(config.output_dir)

    raw_df = load_emg_csv(config.csv_path, config)
    scaled_df, scale_report = infer_scale_and_transform(raw_df, config)

    processor = OpenBCIEMGJoystick1DProcessor(config)
    processed_metrics_df = processor.process(scaled_df)
    processed_df = pd.concat([scaled_df, processed_metrics_df], axis=1)

    processed_csv_path = output_dir / "processed_timeseries.csv"
    processed_df.to_csv(processed_csv_path, index=False)

    counts = label_counts(processed_df)
    label_summary_df = per_label_summary(processed_df, config)
    label_summary_csv_path = output_dir / "label_summary_stats.csv"
    label_summary_df.to_csv(label_summary_csv_path, index=False)

    direction_scans: Dict[str, pd.DataFrame] = {}
    activation_scans: Dict[str, pd.DataFrame] = {}
    methods_payload: Dict[str, object] = {}

    for method in config.aggregation_methods:
        separability = left_right_separability(processed_df, method)
        direction_scan_df, direction_best, direction_confusion = scan_direction_thresholds(
            processed_df, method, config.direction_thresholds
        )
        activation_scan_df, activation_best, activation_confusion = scan_activation_thresholds(
            processed_df, method, config.activation_thresholds
        )
        jaws_summary = summarize_jaws(processed_df, method)

        direction_scans[method] = direction_scan_df
        activation_scans[method] = activation_scan_df
        direction_scan_df.to_csv(output_dir / f"{method}_direction_threshold_scan.csv", index=False)
        activation_scan_df.to_csv(output_dir / f"{method}_activation_threshold_scan.csv", index=False)
        direction_confusion.to_csv(output_dir / f"{method}_jleft_jright_confusion.csv")
        activation_confusion.to_csv(output_dir / f"{method}_norm_active_confusion.csv")

        methods_payload[method] = {
            "separability": separability,
            "direction_threshold_best": direction_best.to_dict(),
            "activation_threshold_best": activation_best.to_dict(),
            "direction_confusion": direction_confusion.to_dict(),
            "activation_confusion": activation_confusion.to_dict(),
            "jaws_summary": jaws_summary,
        }

    plot_paths = []
    if not args.skip_plots:
        plot_paths = [str(path) for path in save_all_plots(processed_df, config, output_dir, direction_scans, activation_scans)]

    summary_payload = {
        "config": asdict(config),
        "scale_report": scale_report,
        "label_counts": counts,
        "methods": methods_payload,
        "plot_paths": plot_paths,
        "processed_csv": str(processed_csv_path),
        "label_summary_csv": str(label_summary_csv_path),
    }
    write_json(output_dir / "summary_metrics.json", summary_payload)
    write_markdown_report(
        output_dir / "summary_report.md",
        config,
        scale_report,
        counts,
        label_summary_df,
        summary_payload,
    )


if __name__ == "__main__":
    main()
