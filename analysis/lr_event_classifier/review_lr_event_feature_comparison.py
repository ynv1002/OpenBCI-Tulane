from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.lr_event_classifier.review_lr_event_classifier import (
    DEFAULT_FS_HZ,
    DEFAULT_VALIDATION_DIR,
    HIGH_TRUST_FILES,
    _build_event_feature_table,
    _compute_common_channels,
    _ensure_validation_outputs,
    _event_features,
    _fit_fold,
    _load_validation_bundle,
    _pooled_results,
    _window_bounds,
)
from analysis.utils import (
    dataframe_to_markdown,
    ensure_output_dir,
    extract_window_features,
    load_openbci_csv,
    preprocess_session_signals,
    write_json,
    write_markdown,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "second_pass"
MATERIAL_IMPROVEMENT_DELTA = 0.03


@dataclass(frozen=True)
class FeatureConfig:
    name: str
    window_sec: float
    include_spectral: bool
    include_spectral_asymmetry: bool
    write_feature_table: bool


FEATURE_CONFIGS = [
    FeatureConfig(
        name="baseline_time_0p50",
        window_sec=0.50,
        include_spectral=False,
        include_spectral_asymmetry=False,
        write_feature_table=False,
    ),
    FeatureConfig(
        name="baseline_plus_spectral_0p50",
        window_sec=0.50,
        include_spectral=True,
        include_spectral_asymmetry=True,
        write_feature_table=True,
    ),
    FeatureConfig(
        name="baseline_plus_spectral_0p75",
        window_sec=0.75,
        include_spectral=True,
        include_spectral_asymmetry=True,
        write_feature_table=True,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline and spectral event-level LEFT vs RIGHT feature sets on the two high-trust Yaniv runs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for second-pass comparison outputs.",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=DEFAULT_VALIDATION_DIR,
        help="Directory containing Yaniv LR validation outputs.",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=DEFAULT_FS_HZ,
        help="Authoritative sample rate for event timing and feature windows.",
    )
    return parser.parse_args()


def _split_spectral_features(
    window,
    channel_columns: list[str],
    fs_hz: float,
    include_asymmetry: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    spectral_features = extract_window_features(
        window,
        channel_columns,
        fs_hz=fs_hz,
        feature_mode="spectral_only",
        with_asymmetry=False,
    )
    asymmetry_features: dict[str, float] = {}
    if include_asymmetry:
        spectral_with_asym = extract_window_features(
            window,
            channel_columns,
            fs_hz=fs_hz,
            feature_mode="spectral_only",
            with_asymmetry=True,
        )
        asymmetry_features = {
            key: value
            for key, value in spectral_with_asym.items()
            if key.startswith(("mu_asym_", "beta_asym_"))
        }
    return spectral_features, asymmetry_features


def _build_feature_table_for_config(
    validation_bundles: list[dict[str, Any]],
    common_channels: list[str],
    fs_hz: float,
    config: FeatureConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if (
        config.name == "baseline_time_0p50"
        and abs(config.window_sec - 0.50) < 1e-9
        and not config.include_spectral
        and not config.include_spectral_asymmetry
    ):
        event_features_df, feature_columns = _build_event_feature_table(validation_bundles, common_channels, fs_hz)
        metadata = {
            "feature_columns": feature_columns,
            "baseline_feature_columns": feature_columns,
            "spectral_feature_columns": [],
            "asymmetry_feature_columns": [],
        }
        return event_features_df, metadata

    window_samples = max(3, int(round(config.window_sec * fs_hz)))
    rows: list[dict[str, Any]] = []
    baseline_feature_columns: list[str] | None = None
    spectral_feature_columns: list[str] | None = None
    asymmetry_feature_columns: list[str] | None = None

    for bundle in validation_bundles:
        filename = bundle["filename"]
        csv_path = REPO_ROOT / "OPENBCI_runs"
        csv_path = Path("/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/EEG_LR") / filename
        raw_df = load_openbci_csv(csv_path)
        processed = preprocess_session_signals(raw_df, "left_right", common_channels, fs_hz)
        block_lookup = {
            int(row["block_id"]): row for row in bundle["block_df"].to_dict(orient="records")
        }

        for event in bundle["event_df"].to_dict(orient="records"):
            block_id = int(event["assigned_block_id"])
            block = block_lookup[block_id]
            peak_sample = int(round(float(event["event_time_sec"]) * fs_hz))
            block_start_sample = int(round(float(block["start_time_sec"]) * fs_hz))
            block_end_sample = int(round(float(block["end_time_sec"]) * fs_hz))
            window_start, window_end = _window_bounds(
                peak_sample=peak_sample,
                block_start_sample=block_start_sample,
                block_end_sample=block_end_sample,
                window_samples=window_samples,
            )
            window = processed.iloc[window_start : window_end + 1].to_numpy(dtype=float)

            baseline_features = _event_features(window, common_channels)
            if baseline_feature_columns is None:
                baseline_feature_columns = list(baseline_features.keys())

            feature_row = dict(baseline_features)
            if config.include_spectral:
                spectral_features, asymmetry_features = _split_spectral_features(
                    window,
                    common_channels,
                    fs_hz=fs_hz,
                    include_asymmetry=config.include_spectral_asymmetry,
                )
                if spectral_feature_columns is None:
                    spectral_feature_columns = list(spectral_features.keys())
                if asymmetry_feature_columns is None:
                    asymmetry_feature_columns = list(asymmetry_features.keys())
                feature_row.update(spectral_features)
                feature_row.update(asymmetry_features)

            row = {
                "file": filename,
                "event_id": int(event["event_id"]),
                "event_time_sec": float(event["event_time_sec"]),
                "event_time_relative_sec": float(event["event_time_relative_sec"]),
                "block_id": block_id,
                "label": str(event["assigned_side"]),
                "block_start_time_sec": float(block["start_time_sec"]),
                "block_end_time_sec": float(block["end_time_sec"]),
                "peak_value": float(event["peak_value"]),
                "peak_prominence": float(event["peak_prominence"]),
                "window_start_time_sec": float(window_start / fs_hz),
                "window_end_time_sec": float(window_end / fs_hz),
                "window_sec": float((window_end - window_start + 1) / fs_hz),
            }
            row.update(feature_row)
            rows.append(row)

    event_features_df = pd.DataFrame(rows).sort_values(["file", "event_time_sec"]).reset_index(drop=True)
    metadata_columns = {
        "file",
        "event_id",
        "event_time_sec",
        "event_time_relative_sec",
        "block_id",
        "label",
        "block_start_time_sec",
        "block_end_time_sec",
        "peak_value",
        "peak_prominence",
        "window_start_time_sec",
        "window_end_time_sec",
        "window_sec",
    }
    feature_columns = [column for column in event_features_df.columns if column not in metadata_columns]
    metadata = {
        "feature_columns": feature_columns,
        "baseline_feature_columns": baseline_feature_columns or [],
        "spectral_feature_columns": spectral_feature_columns or [],
        "asymmetry_feature_columns": asymmetry_feature_columns or [],
    }
    return event_features_df, metadata


def _run_feature_config(
    event_features_df: pd.DataFrame,
    feature_columns: list[str],
    config: FeatureConfig,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    config_dir = ensure_output_dir(output_dir / config.name)
    for test_file in HIGH_TRUST_FILES:
        train_df = event_features_df[event_features_df["file"] != test_file].reset_index(drop=True)
        test_df = event_features_df[event_features_df["file"] == test_file].reset_index(drop=True)
        fold_results_df, fold_predictions_df = _fit_fold(
            train_df=train_df,
            test_df=test_df,
            feature_columns=feature_columns,
            output_dir=config_dir,
        )
        fold_frames.append(fold_results_df)
        prediction_frames.append(fold_predictions_df)

    results_df = pd.concat(fold_frames, ignore_index=True)
    prediction_df = pd.concat(prediction_frames, ignore_index=True)
    pooled_df = _pooled_results(prediction_df, event_features_df, config_dir)
    results_df = pd.concat([results_df, pooled_df], ignore_index=True)
    return results_df, prediction_df


def _build_summary(
    results_df: pd.DataFrame,
    feature_metadata_by_config: dict[str, dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    pooled = results_df[results_df["train_file"] == "__pooled_cross_session__"].copy()
    pooled["is_spectral"] = pooled["feature_config"] != "baseline_time_0p50"
    baseline_best = pooled[pooled["feature_config"] == "baseline_time_0p50"].sort_values(
        ["macro_f1", "accuracy"],
        ascending=False,
    ).iloc[0]
    spectral_best = pooled[pooled["is_spectral"]].sort_values(
        ["macro_f1", "accuracy"],
        ascending=False,
    ).iloc[0]
    best_overall = pooled.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]
    delta_macro_f1 = float(spectral_best["macro_f1"] - baseline_best["macro_f1"])
    delta_accuracy = float(spectral_best["accuracy"] - baseline_best["accuracy"])
    materially_improved = bool(delta_macro_f1 >= MATERIAL_IMPROVEMENT_DELTA)

    summary = {
        "files_used": HIGH_TRUST_FILES,
        "feature_configs": {
            config_name: {
                "window_sec": float(
                    pooled.loc[pooled["feature_config"] == config_name, "window_sec"].iloc[0]
                ),
                "feature_count": int(metadata["feature_count"]),
                "baseline_feature_count": int(metadata["baseline_feature_count"]),
                "spectral_feature_count": int(metadata["spectral_feature_count"]),
                "asymmetry_feature_count": int(metadata["asymmetry_feature_count"]),
            }
            for config_name, metadata in feature_metadata_by_config.items()
        },
        "baseline_best": baseline_best.to_dict(),
        "spectral_best": spectral_best.to_dict(),
        "best_overall": best_overall.to_dict(),
        "delta_macro_f1_vs_baseline": delta_macro_f1,
        "delta_accuracy_vs_baseline": delta_accuracy,
        "materially_improved": materially_improved,
        "csp_status": "Skipped: there is no clean CSP implementation path in the repo.",
    }
    write_json(output_dir / "summary.json", summary)

    pooled_preview = pooled.loc[
        :,
        [
            "feature_config",
            "window_sec",
            "model_name",
            "feature_count",
            "baseline_feature_count",
            "spectral_feature_count",
            "asymmetry_feature_count",
            "accuracy",
            "macro_f1",
        ],
    ].copy()
    for column in ("accuracy", "macro_f1"):
        pooled_preview[column] = pooled_preview[column].map(lambda value: f"{float(value):.3f}")

    lines = [
        "# Second-Pass LR Event Feature Comparison",
        "",
        "## Setup",
        f"- Files used: `{', '.join(HIGH_TRUST_FILES)}`",
        "- Evaluation: `two-fold leave-one-session-out cross-session`",
        "- Labeled events: `375 total` (`183 LEFT`, `192 RIGHT`)",
        "- Baseline config reuses the existing event-level baseline feature builder exactly.",
        "- Spectral features use the repo-native `mu_power`, `beta_power`, and pairwise `mu_asym_*` / `beta_asym_*` features from `analysis.utils.extract_window_features`.",
        "- CSP: skipped because there is no clean existing CSP implementation path in the repo.",
        "",
        "## Pooled Results",
        dataframe_to_markdown(pooled_preview),
        "",
        "## Interpretation",
        (
            f"- Best baseline config: `{baseline_best['feature_config']}` / `{baseline_best['model_name']}` "
            f"with accuracy `{float(baseline_best['accuracy']):.3f}` and macro-F1 `{float(baseline_best['macro_f1']):.3f}`."
        ),
        (
            f"- Best spectral config: `{spectral_best['feature_config']}` / `{spectral_best['model_name']}` "
            f"with accuracy `{float(spectral_best['accuracy']):.3f}` and macro-F1 `{float(spectral_best['macro_f1']):.3f}`."
        ),
        (
            f"- Delta vs baseline: accuracy `{delta_accuracy:+.3f}`, macro-F1 `{delta_macro_f1:+.3f}`."
        ),
    ]
    if materially_improved:
        lines.append("- The added spectral/spatial features produced a material cross-session improvement over the first baseline.")
    elif delta_macro_f1 > 0:
        lines.append("- The added spectral/spatial features helped slightly, but not enough to count as a material improvement.")
    else:
        lines.append("- The added spectral/spatial features did not materially improve the first baseline.")

    spectral_050 = pooled[pooled["feature_config"] == "baseline_plus_spectral_0p50"].sort_values(
        ["macro_f1", "accuracy"],
        ascending=False,
    ).iloc[0]
    spectral_075 = pooled[pooled["feature_config"] == "baseline_plus_spectral_0p75"].sort_values(
        ["macro_f1", "accuracy"],
        ascending=False,
    ).iloc[0]
    if float(spectral_075["macro_f1"]) > float(spectral_050["macro_f1"]):
        lines.append("- The wider `0.75 s` event window helped the spectral configuration more than the strict `0.50 s` comparison window.")
    elif float(spectral_075["macro_f1"]) < float(spectral_050["macro_f1"]):
        lines.append("- Widening the event window to `0.75 s` did not help the spectral configuration.")
    else:
        lines.append("- The `0.50 s` and `0.75 s` spectral windows were effectively tied.")

    if float(best_overall["macro_f1"]) >= 0.65:
        lines.append("- Event-level LEFT vs RIGHT now looks promising enough to justify a next refinement pass.")
    else:
        lines.append("- Event-level LEFT vs RIGHT still looks weak cross-session and likely needs a better event representation or session calibration.")

    write_markdown(output_dir / "feature_set_comparison_summary.md", "\n".join(lines))
    return summary


def run_feature_comparison(
    output_dir: Path,
    validation_dir: Path,
    fs_hz: float,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    validation_dir = ensure_output_dir(validation_dir)
    _ensure_validation_outputs(validation_dir, fs_hz)

    file_paths = [Path("/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/EEG_LR") / filename for filename in HIGH_TRUST_FILES]
    common_channels = _compute_common_channels(file_paths)
    validation_bundles = [_load_validation_bundle(validation_dir, filename) for filename in HIGH_TRUST_FILES]

    all_results_frames: list[pd.DataFrame] = []
    feature_metadata_by_config: dict[str, dict[str, Any]] = {}

    for config in FEATURE_CONFIGS:
        event_features_df, metadata = _build_feature_table_for_config(
            validation_bundles=validation_bundles,
            common_channels=common_channels,
            fs_hz=fs_hz,
            config=config,
        )
        feature_columns = metadata["feature_columns"]
        results_df, _ = _run_feature_config(
            event_features_df=event_features_df,
            feature_columns=feature_columns,
            config=config,
            output_dir=output_dir,
        )
        results_df.insert(0, "feature_config", config.name)
        results_df.insert(1, "window_sec", float(config.window_sec))
        results_df.insert(6, "feature_count", int(len(feature_columns)))
        results_df.insert(7, "baseline_feature_count", int(len(metadata["baseline_feature_columns"])))
        results_df.insert(8, "spectral_feature_count", int(len(metadata["spectral_feature_columns"])))
        results_df.insert(9, "asymmetry_feature_count", int(len(metadata["asymmetry_feature_columns"])))
        all_results_frames.append(results_df)

        feature_metadata_by_config[config.name] = {
            "feature_count": len(feature_columns),
            "baseline_feature_count": len(metadata["baseline_feature_columns"]),
            "spectral_feature_count": len(metadata["spectral_feature_columns"]),
            "asymmetry_feature_count": len(metadata["asymmetry_feature_columns"]),
        }

        if config.write_feature_table:
            table_path = output_dir / f"event_feature_table_{config.name}.csv"
            event_features_df.to_csv(table_path, index=False)

    feature_set_results_df = pd.concat(all_results_frames, ignore_index=True)
    ordered_columns = [
        "feature_config",
        "window_sec",
        "model_name",
        "train_file",
        "test_file",
        "train_samples",
        "test_samples",
        "feature_count",
        "baseline_feature_count",
        "spectral_feature_count",
        "asymmetry_feature_count",
        "train_class_counts",
        "test_class_counts",
        "accuracy",
        "macro_f1",
        "confusion_matrix_csv",
    ]
    feature_set_results_df = feature_set_results_df.loc[:, ordered_columns].copy()
    feature_set_results_path = output_dir / "feature_set_results.csv"
    feature_set_results_df.to_csv(feature_set_results_path, index=False)

    summary = _build_summary(
        results_df=feature_set_results_df,
        feature_metadata_by_config=feature_metadata_by_config,
        output_dir=output_dir,
    )

    pooled_preview = feature_set_results_df[
        feature_set_results_df["train_file"] == "__pooled_cross_session__"
    ].loc[
        :,
        ["feature_config", "model_name", "accuracy", "macro_f1"],
    ].copy()
    pooled_preview["accuracy"] = pooled_preview["accuracy"].map(lambda value: f"{float(value):.3f}")
    pooled_preview["macro_f1"] = pooled_preview["macro_f1"].map(lambda value: f"{float(value):.3f}")
    print("Second-pass LR event feature comparison")
    print(f"  Output directory: {output_dir}")
    print(f"  Files used: {', '.join(HIGH_TRUST_FILES)}")
    print(f"  Selected channels: {', '.join(common_channels)}")
    print("")
    print("Pooled cross-session results")
    print(dataframe_to_markdown(pooled_preview))
    print("")
    print(
        f"Best overall: {summary['best_overall']['feature_config']} / {summary['best_overall']['model_name']} "
        f"(accuracy={float(summary['best_overall']['accuracy']):.3f}, macro_f1={float(summary['best_overall']['macro_f1']):.3f})"
    )

    return {
        "feature_set_results_path": feature_set_results_path,
        "summary": summary,
    }


def main() -> None:
    args = parse_args()
    run_feature_comparison(
        output_dir=args.output_dir,
        validation_dir=args.validation_dir,
        fs_hz=float(args.fs),
    )


if __name__ == "__main__":
    main()
