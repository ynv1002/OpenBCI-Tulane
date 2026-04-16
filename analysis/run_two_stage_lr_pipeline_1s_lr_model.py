from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.run_active_vs_rest_cross_session_calibrated import (
        DEFAULT_CALIBRATION_SEC,
        TEST_FILENAME,
        TRAIN_FILENAME,
        compute_calibration_stats,
        normalize_with_calibration,
    )
    from analysis.run_active_vs_rest_experiment import build_bundle_for_subset, resolve_left_right_subset
    from analysis.run_active_vs_rest_experiment import remap_active_vs_rest
    from analysis.run_two_stage_lr_pipeline import (
        ACTIVE_THRESHOLD,
        FINAL_LABELS,
        LEFT_LABEL,
        REST_LABEL,
        RIGHT_LABEL,
        fit_active_gate,
        fit_left_right_model,
        normalize_windows_by_session,
    )
    from analysis.utils import audit_all_sessions, dataframe_to_markdown, ensure_output_dir, write_markdown
else:
    from .run_active_vs_rest_cross_session_calibrated import (
        DEFAULT_CALIBRATION_SEC,
        TEST_FILENAME,
        TRAIN_FILENAME,
        compute_calibration_stats,
        normalize_with_calibration,
    )
    from .run_active_vs_rest_experiment import build_bundle_for_subset, resolve_left_right_subset
    from .run_active_vs_rest_experiment import remap_active_vs_rest
    from .run_two_stage_lr_pipeline import (
        ACTIVE_THRESHOLD,
        FINAL_LABELS,
        LEFT_LABEL,
        REST_LABEL,
        RIGHT_LABEL,
        fit_active_gate,
        fit_left_right_model,
        normalize_windows_by_session,
    )
    from .utils import audit_all_sessions, dataframe_to_markdown, ensure_output_dir, write_markdown


GATE_WINDOW_SEC = 2.0
LR_WINDOW_SEC = 1.0
OVERLAP = 0.5
BASELINE_RESULTS_PATH = Path(__file__).resolve().parent / "outputs" / "two_stage_lr_pipeline_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the calibrated two-stage pipeline with a 1.0s LEFT/RIGHT model."
    )
    parser.add_argument("--calibration-sec", type=float, default=DEFAULT_CALIBRATION_SEC)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for 1.0s LEFT/RIGHT pipeline outputs.",
    )
    return parser.parse_args()


def project_gate_probabilities_to_lr_windows(
    gate_test_frame: pd.DataFrame,
    p_active: pd.Series,
    lr_test_frame: pd.DataFrame,
) -> pd.Series:
    gate_windows = gate_test_frame.loc[:, ["start_time_sec", "end_time_sec"]].copy()
    gate_windows["p_active"] = p_active.to_numpy(dtype=float)

    projected: List[float] = []
    for _, row in lr_test_frame.loc[:, ["start_time_sec", "end_time_sec"]].iterrows():
        center_time = 0.5 * (float(row["start_time_sec"]) + float(row["end_time_sec"]))
        overlaps = gate_windows.loc[
            (gate_windows["start_time_sec"] <= center_time) & (gate_windows["end_time_sec"] >= center_time),
            "p_active",
        ]
        if overlaps.empty:
            nearest_idx = (
                (0.5 * (gate_windows["start_time_sec"] + gate_windows["end_time_sec"]) - center_time)
                .abs()
                .idxmin()
            )
            projected.append(float(gate_windows.loc[nearest_idx, "p_active"]))
        else:
            projected.append(float(overlaps.max()))
    return pd.Series(projected, index=lr_test_frame.index, name="projected_p_active")


def load_baseline_results(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df.iloc[0]


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    audits = audit_all_sessions()

    gate_subset_audits = resolve_left_right_subset(audits, [TRAIN_FILENAME, TEST_FILENAME])
    gate_bundle = build_bundle_for_subset(
        gate_subset_audits,
        window_sec=GATE_WINDOW_SEC,
        overlap=OVERLAP,
        with_asymmetry=True,
    )

    lr_subset_audits = resolve_left_right_subset(audits, None)
    lr_bundle = build_bundle_for_subset(
        lr_subset_audits,
        window_sec=LR_WINDOW_SEC,
        overlap=OVERLAP,
        with_asymmetry=True,
    )

    gate_train_mean, gate_train_std, gate_train_calib_windows = compute_calibration_stats(
        gate_bundle["train_windows"], gate_bundle["feature_columns"], args.calibration_sec
    )
    gate_test_mean, gate_test_std, gate_test_calib_windows = compute_calibration_stats(
        gate_bundle["test_windows"], gate_bundle["feature_columns"], args.calibration_sec
    )

    gate_train_norm = normalize_with_calibration(
        gate_bundle["train_windows"], gate_bundle["feature_columns"], gate_train_mean, gate_train_std
    ).reset_index(drop=True)
    gate_test_norm = normalize_with_calibration(
        gate_bundle["test_windows"], gate_bundle["feature_columns"], gate_test_mean, gate_test_std
    ).reset_index(drop=True)

    lr_train_norm, lr_train_calibration_counts = normalize_windows_by_session(
        lr_bundle["train_windows"], lr_bundle["feature_columns"], args.calibration_sec
    )
    lr_test_norm, lr_test_calibration_counts = normalize_windows_by_session(
        lr_bundle["test_windows"], lr_bundle["feature_columns"], args.calibration_sec
    )

    gate_train = remap_active_vs_rest(gate_train_norm)
    _, gate_test_p_active = fit_active_gate(gate_train, gate_test_norm, gate_bundle["feature_columns"])
    projected_p_active = project_gate_probabilities_to_lr_windows(gate_test_norm, gate_test_p_active, lr_test_norm)
    gate_open = projected_p_active >= ACTIVE_THRESHOLD

    lr_pred = fit_left_right_model(lr_train_norm, lr_test_norm, lr_bundle["feature_columns"])

    final_pred = pd.Series(REST_LABEL, index=lr_test_norm.index, name="final_pred")
    final_pred.loc[gate_open] = lr_pred.loc[gate_open]

    y_true = lr_test_norm["label"].copy()
    y_true = y_true.where(y_true.isin([LEFT_LABEL, RIGHT_LABEL]), REST_LABEL)

    cm = pd.DataFrame(
        confusion_matrix(y_true, final_pred, labels=FINAL_LABELS),
        index=[f"true_{label}" for label in FINAL_LABELS],
        columns=[f"pred_{label}" for label in FINAL_LABELS],
    )

    active_true_mask = y_true.isin([LEFT_LABEL, RIGHT_LABEL])
    active_detection_rate = float(gate_open.mean())
    active_recall = float(gate_open.loc[active_true_mask].mean()) if active_true_mask.any() else float("nan")
    lr_accuracy_on_active_windows = float(
        accuracy_score(y_true.loc[active_true_mask], final_pred.loc[active_true_mask])
    ) if active_true_mask.any() else float("nan")
    lr_accuracy_when_detected = float(
        accuracy_score(
            y_true.loc[active_true_mask & gate_open],
            lr_pred.loc[active_true_mask & gate_open],
        )
    ) if (active_true_mask & gate_open).any() else float("nan")

    results_row = {
        "train_file": TRAIN_FILENAME,
        "test_file": TEST_FILENAME,
        "gate_window_sec": GATE_WINDOW_SEC,
        "left_right_window_sec": LR_WINDOW_SEC,
        "overlap": OVERLAP,
        "calibration_sec": args.calibration_sec,
        "gate_threshold": ACTIVE_THRESHOLD,
        "accuracy": float(accuracy_score(y_true, final_pred)),
        "macro_f1": float(f1_score(y_true, final_pred, average="macro")),
        "active_detection_rate": active_detection_rate,
        "active_recall": active_recall,
        "left_right_accuracy_on_active_windows": lr_accuracy_on_active_windows,
        "left_right_accuracy_when_detected": lr_accuracy_when_detected,
        "gate_selected_channels": ", ".join(gate_bundle["selected_channels"]),
        "left_right_selected_channels": ", ".join(lr_bundle["selected_channels"]),
        "gate_train_calibration_windows": gate_train_calib_windows,
        "gate_test_calibration_windows": gate_test_calib_windows,
        "left_right_train_calibration_windows": lr_train_calibration_counts,
        "left_right_test_calibration_windows": lr_test_calibration_counts,
    }

    results_df = pd.DataFrame([results_row])
    baseline_row = load_baseline_results(BASELINE_RESULTS_PATH)
    comparison_df = results_df.copy()
    if baseline_row is not None:
        comparison_df.insert(0, "comparison_target", "new_1s_lr_model")

    results_csv = output_dir / "two_stage_lr_pipeline_1s_lr_model.csv"
    confusion_csv = output_dir / "two_stage_lr_pipeline_1s_lr_model_confusion_matrix.csv"
    predictions_csv = output_dir / "two_stage_lr_pipeline_1s_lr_model_predictions.csv"
    summary_md = output_dir / "two_stage_lr_pipeline_1s_lr_model_summary.md"

    results_df.to_csv(results_csv, index=False)
    cm.to_csv(confusion_csv)

    prediction_table = lr_test_norm.loc[:, ["filename", "start_time_sec", "end_time_sec", "label"]].copy()
    prediction_table["true_label"] = y_true
    prediction_table["projected_p_active"] = projected_p_active
    prediction_table["gate_open"] = gate_open.astype(int)
    prediction_table["left_right_pred"] = lr_pred
    prediction_table["final_pred"] = final_pred
    prediction_table.to_csv(predictions_csv, index=False)

    comparison_lines: List[str] = []
    if baseline_row is not None:
        comparison_lines.extend(
            [
                f"- Previous pipeline macro-F1 (2.0s LR): `{float(baseline_row['macro_f1']):.3f}`",
                f"- New pipeline macro-F1 (1.0s LR): `{results_row['macro_f1']:.3f}`",
                f"- Macro-F1 change: `{results_row['macro_f1'] - float(baseline_row['macro_f1']):+.3f}`",
                f"- Previous LEFT/RIGHT accuracy on true ACTIVE windows: `{float(baseline_row['left_right_accuracy_on_active_windows']):.3f}`",
                f"- New LEFT/RIGHT accuracy on true ACTIVE windows: `{results_row['left_right_accuracy_on_active_windows']:.3f}`",
                f"- LEFT/RIGHT-on-active change: `{results_row['left_right_accuracy_on_active_windows'] - float(baseline_row['left_right_accuracy_on_active_windows']):+.3f}`",
                f"- Previous LEFT/RIGHT accuracy when gate opens: `{float(baseline_row['left_right_accuracy_when_detected']):.3f}`",
                f"- New LEFT/RIGHT accuracy when gate opens: `{results_row['left_right_accuracy_when_detected']:.3f}`",
                f"- LEFT/RIGHT-when-detected change: `{results_row['left_right_accuracy_when_detected'] - float(baseline_row['left_right_accuracy_when_detected']):+.3f}`",
                f"- Previous ACTIVE recall: `{float(baseline_row['active_recall']):.3f}`",
                f"- New ACTIVE recall: `{results_row['active_recall']:.3f}`",
                f"- ACTIVE recall change: `{results_row['active_recall'] - float(baseline_row['active_recall']):+.3f}`",
            ]
        )

    report_lines = [
        "# Two-Stage LEFT/RIGHT Pipeline With 1.0s LR Model",
        "",
        f"- Gate model remains unchanged: calibrated `RandomForest` at `{GATE_WINDOW_SEC:.1f}` s with threshold `{ACTIVE_THRESHOLD:.1f}`",
        f"- LEFT/RIGHT model retrained at `{LR_WINDOW_SEC:.1f}` s windows",
        f"- Output CSV: `{results_csv}`",
        f"- Confusion matrix: `{confusion_csv}`",
        "",
        "## Results",
        "",
        dataframe_to_markdown(results_df),
        "",
        "## Confusion Matrix",
        "",
        dataframe_to_markdown(cm.reset_index().rename(columns={"index": "label"})),
        "",
        "## Comparison vs Previous 2.0s LR Pipeline",
        "",
        *comparison_lines,
    ]
    write_markdown(summary_md, "\n".join(report_lines))

    print(dataframe_to_markdown(results_df))
    print("\nConfusion matrix")
    print(dataframe_to_markdown(cm.reset_index().rename(columns={"index": "label"})))
    if comparison_lines:
        print("\nComparison vs previous 2.0s LR pipeline")
        for line in comparison_lines:
            print(line)
    print(f"\nWrote {results_csv}")
    print(f"Wrote {confusion_csv}")
    print(f"Wrote {predictions_csv}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
