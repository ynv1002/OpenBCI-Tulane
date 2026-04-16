from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Sequence

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.run_active_vs_rest_cross_session_threshold import TEST_FILENAME, TRAIN_FILENAME
    from analysis.run_active_vs_rest_experiment import ACTIVE_LABEL, model_bank, remap_active_vs_rest
    from analysis.run_active_vs_rest_threshold_sweep import evaluate_thresholds
    from analysis.utils import LABEL_BASELINE, LABEL_REST, audit_all_sessions, dataframe_to_markdown, ensure_output_dir, write_markdown
    from analysis.run_active_vs_rest_experiment import build_bundle_for_subset, resolve_left_right_subset
else:
    from .run_active_vs_rest_cross_session_threshold import TEST_FILENAME, TRAIN_FILENAME
    from .run_active_vs_rest_experiment import ACTIVE_LABEL, model_bank, remap_active_vs_rest
    from .run_active_vs_rest_threshold_sweep import evaluate_thresholds
    from .utils import LABEL_BASELINE, LABEL_REST, audit_all_sessions, dataframe_to_markdown, ensure_output_dir, write_markdown
    from .run_active_vs_rest_experiment import build_bundle_for_subset, resolve_left_right_subset


DEFAULT_THRESHOLD = 0.4
DEFAULT_CALIBRATION_SEC = 45.0
NO_NORM_RESULTS_PATH = Path(__file__).resolve().parent / "outputs" / "active_vs_rest_cross_session_threshold_comparison.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate cross-session ACTIVE vs REST with session-specific baseline calibration normalization."
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--calibration-sec", type=float, default=DEFAULT_CALIBRATION_SEC)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for calibration-normalization outputs.",
    )
    return parser.parse_args()


def calibration_subset(
    frame: pd.DataFrame,
    calibration_sec: float,
) -> pd.DataFrame:
    mask = (
        frame["label"].isin([LABEL_BASELINE, LABEL_REST])
        & (frame["start_time_sec"] >= 0.0)
        & (frame["end_time_sec"] <= calibration_sec)
    )
    return frame.loc[mask].copy()


def compute_calibration_stats(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    calibration_sec: float,
) -> tuple[pd.Series, pd.Series, int]:
    calib = calibration_subset(frame, calibration_sec)
    if calib.empty:
        raise ValueError("No BASELINE/REST calibration windows were found in the requested calibration period.")
    mean = calib.loc[:, list(feature_columns)].mean(axis=0)
    std = calib.loc[:, list(feature_columns)].std(axis=0, ddof=0)
    std = std.replace(0.0, 1.0).fillna(1.0)
    return mean, std, int(len(calib))


def normalize_with_calibration(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    mean: pd.Series,
    std: pd.Series,
) -> pd.DataFrame:
    out = frame.copy()
    out.loc[:, list(feature_columns)] = (out.loc[:, list(feature_columns)] - mean) / std
    return out


def load_no_norm_row(path: Path, threshold: float) -> pd.Series | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    match = df.loc[np.isclose(df["threshold"], threshold)]
    if match.empty:
        return None
    return match.iloc[0]


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    audits = audit_all_sessions()
    subset_audits = resolve_left_right_subset(audits, [TRAIN_FILENAME, TEST_FILENAME])
    bundle = build_bundle_for_subset(
        subset_audits,
        window_sec=2.0,
        overlap=0.5,
        with_asymmetry=True,
    )

    train_mean, train_std, train_calib_windows = compute_calibration_stats(
        bundle["train_windows"],
        bundle["feature_columns"],
        args.calibration_sec,
    )
    test_mean, test_std, test_calib_windows = compute_calibration_stats(
        bundle["test_windows"],
        bundle["feature_columns"],
        args.calibration_sec,
    )

    train_norm = normalize_with_calibration(bundle["train_windows"], bundle["feature_columns"], train_mean, train_std)
    test_norm = normalize_with_calibration(bundle["test_windows"], bundle["feature_columns"], test_mean, test_std)

    train_df = remap_active_vs_rest(train_norm)
    test_df = remap_active_vs_rest(test_norm)

    model = model_bank()["RandomForest"]
    X_train = train_df.loc[:, bundle["feature_columns"]].to_numpy(dtype=float)
    X_test = test_df.loc[:, bundle["feature_columns"]].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy()
    y_test = pd.Series(test_df["label"].to_numpy(), name="label")

    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    active_index = list(model.classes_).index(ACTIVE_LABEL)
    active_prob = pd.Series(probs[:, active_index], index=test_df.index, name="p_active")

    calibrated_df = evaluate_thresholds(y_test, active_prob, [args.threshold])
    calibrated_row = calibrated_df.iloc[0]

    no_norm_row = load_no_norm_row(NO_NORM_RESULTS_PATH, args.threshold)

    comparison_rows = [
        {
            "condition": "calibrated_per_session_baseline",
            "threshold": args.threshold,
            "accuracy": float(calibrated_row["accuracy"]),
            "macro_f1": float(calibrated_row["macro_f1"]),
            "active_precision": float(calibrated_row["active_precision"]),
            "active_recall": float(calibrated_row["active_recall"]),
            "misses": int(calibrated_row["misses"]),
            "false_triggers": int(calibrated_row["false_triggers"]),
            "train_calibration_windows": train_calib_windows,
            "test_calibration_windows": test_calib_windows,
        }
    ]
    if no_norm_row is not None:
        comparison_rows.insert(
            0,
            {
                "condition": "no_normalization",
                "threshold": float(no_norm_row["threshold"]),
                "accuracy": float(no_norm_row["accuracy"]),
                "macro_f1": float(no_norm_row["macro_f1"]),
                "active_precision": float(no_norm_row["active_precision"]),
                "active_recall": float(no_norm_row["active_recall"]),
                "misses": int(no_norm_row["misses"]),
                "false_triggers": int(no_norm_row["false_triggers"]),
                "train_calibration_windows": train_calib_windows,
                "test_calibration_windows": test_calib_windows,
            },
        )

    export_df = pd.DataFrame(comparison_rows)
    results_csv = output_dir / "active_vs_rest_cross_session_calibrated_comparison.csv"
    export_df.to_csv(results_csv, index=False)

    confusion_csv = output_dir / "confusion_active_vs_rest_cross_session_calibrated_threshold_0_4.csv"
    calibrated_row["confusion_matrix"].to_csv(confusion_csv)

    delta_lines: list[str] = []
    if no_norm_row is not None:
        delta_lines.extend(
            [
                f"- ACTIVE recall change vs no normalization: `{float(calibrated_row['active_recall']) - float(no_norm_row['active_recall']):+.3f}`",
                f"- Macro-F1 change vs no normalization: `{float(calibrated_row['macro_f1']) - float(no_norm_row['macro_f1']):+.3f}`",
                f"- False trigger change vs no normalization: `{int(calibrated_row['false_triggers']) - int(no_norm_row['false_triggers']):+d}`",
                f"- Miss change vs no normalization: `{int(calibrated_row['misses']) - int(no_norm_row['misses']):+d}`",
            ]
        )

    summary_lines = [
        "# ACTIVE vs REST Cross-Session Calibration Normalization",
        "",
        f"- Train file: `{TRAIN_FILENAME}`",
        f"- Test file: `{TEST_FILENAME}`",
        f"- Model: `RandomForest`",
        f"- Features: `combined + asymmetry`",
        f"- Threshold: `{args.threshold:.1f}`",
        f"- Calibration period: first `{args.calibration_sec:.0f}` seconds",
        f"- Train calibration windows used: `{train_calib_windows}`",
        f"- Test calibration windows used: `{test_calib_windows}`",
        f"- Output CSV: `{results_csv}`",
        "",
        "## Results",
        "",
        dataframe_to_markdown(export_df),
        "",
        "## Comparison",
        "",
        *delta_lines,
        "",
        "## Interpretation",
        "",
        f"- Calibrated ACTIVE recall: `{float(calibrated_row['active_recall']):.3f}`",
        f"- Calibrated macro-F1: `{float(calibrated_row['macro_f1']):.3f}`",
        f"- Calibrated false triggers: `{int(calibrated_row['false_triggers'])}`",
    ]
    summary_md = output_dir / "active_vs_rest_cross_session_calibrated_comparison.md"
    write_markdown(summary_md, "\n".join(summary_lines))

    print(dataframe_to_markdown(export_df))
    print("\nComparison")
    if delta_lines:
        for line in delta_lines:
            print(line)
    else:
        print("- No prior no-normalization result was found for comparison.")

    print("\nInterpretation")
    if no_norm_row is not None:
        if float(calibrated_row["active_recall"]) > float(no_norm_row["active_recall"]):
            print("- Recall improved with per-session calibration normalization.")
        else:
            print("- Recall did not improve with per-session calibration normalization.")
        if float(calibrated_row["macro_f1"]) > float(no_norm_row["macro_f1"]):
            print("- Macro-F1 improved with per-session calibration normalization.")
        else:
            print("- Macro-F1 did not improve with per-session calibration normalization.")
        if int(calibrated_row["false_triggers"]) < int(no_norm_row["false_triggers"]):
            print("- Calibration reduced false triggers, which suggests some session mismatch was reduced.")
        elif int(calibrated_row["false_triggers"]) > int(no_norm_row["false_triggers"]):
            print("- Calibration increased false triggers, so session mismatch was not cleanly reduced.")
        else:
            print("- Calibration left false triggers unchanged.")
    else:
        print("- No no-normalization baseline was available for direct comparison.")

    print(f"\nWrote {results_csv}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
