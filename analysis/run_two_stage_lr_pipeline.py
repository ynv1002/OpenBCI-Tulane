from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Sequence

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
    from analysis.run_active_vs_rest_experiment import ACTIVE_LABEL, model_bank, remap_active_vs_rest
    from analysis.run_active_vs_rest_experiment import build_bundle_for_subset, resolve_left_right_subset
    from analysis.utils import audit_all_sessions, dataframe_to_markdown, ensure_output_dir, write_markdown
else:
    from .run_active_vs_rest_cross_session_calibrated import (
        DEFAULT_CALIBRATION_SEC,
        TEST_FILENAME,
        TRAIN_FILENAME,
        compute_calibration_stats,
        normalize_with_calibration,
    )
    from .run_active_vs_rest_experiment import ACTIVE_LABEL, model_bank, remap_active_vs_rest
    from .run_active_vs_rest_experiment import build_bundle_for_subset, resolve_left_right_subset
    from .utils import audit_all_sessions, dataframe_to_markdown, ensure_output_dir, write_markdown


LEFT_LABEL = "LEFT"
RIGHT_LABEL = "RIGHT"
REST_LABEL = "REST"
FINAL_LABELS = [REST_LABEL, LEFT_LABEL, RIGHT_LABEL]
ACTIVE_THRESHOLD = 0.3
LR_MODEL_NAME = "RandomForest"
GATE_MODEL_NAME = "RandomForest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the calibrated two-stage LEFT/RIGHT pipeline."
    )
    parser.add_argument("--calibration-sec", type=float, default=DEFAULT_CALIBRATION_SEC)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for two-stage pipeline outputs.",
    )
    return parser.parse_args()


def fit_active_gate(
    train_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    feature_columns: List[str],
) -> tuple[object, pd.Series]:
    gate_model = model_bank()[GATE_MODEL_NAME]
    X_train = train_df.loc[:, feature_columns].to_numpy(dtype=float)
    X_test = prediction_df.loc[:, feature_columns].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy()
    gate_model.fit(X_train, y_train)
    probs = gate_model.predict_proba(X_test)
    active_index = list(gate_model.classes_).index(ACTIVE_LABEL)
    return gate_model, pd.Series(probs[:, active_index], index=prediction_df.index, name="p_active")


def fit_left_right_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str],
) -> pd.Series:
    lr_train = train_df[train_df["label"].isin([LEFT_LABEL, RIGHT_LABEL])].copy()
    lr_model = model_bank()[LR_MODEL_NAME]
    X_train = lr_train.loc[:, feature_columns].to_numpy(dtype=float)
    y_train = lr_train["label"].to_numpy()
    X_test = test_df.loc[:, feature_columns].to_numpy(dtype=float)
    lr_model.fit(X_train, y_train)
    return pd.Series(lr_model.predict(X_test), index=test_df.index, name="lr_pred")


def normalize_windows_by_session(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    calibration_sec: float,
) -> tuple[pd.DataFrame, Dict[str, int]]:
    normalized_frames: List[pd.DataFrame] = []
    calibration_counts: Dict[str, int] = {}
    for filename, session_df in frame.groupby("filename", sort=False):
        mean, std, calib_count = compute_calibration_stats(session_df, feature_columns, calibration_sec)
        normalized_frames.append(normalize_with_calibration(session_df, feature_columns, mean, std))
        calibration_counts[str(filename)] = calib_count
    normalized = pd.concat(normalized_frames, ignore_index=True) if normalized_frames else pd.DataFrame()
    return normalized.reset_index(drop=True), calibration_counts


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    audits = audit_all_sessions()
    gate_subset_audits = resolve_left_right_subset(audits, [TRAIN_FILENAME, TEST_FILENAME])
    gate_bundle = build_bundle_for_subset(
        gate_subset_audits,
        window_sec=2.0,
        overlap=0.5,
        with_asymmetry=True,
    )
    lr_subset_audits = resolve_left_right_subset(audits, None)
    lr_bundle = build_bundle_for_subset(
        lr_subset_audits,
        window_sec=2.0,
        overlap=0.5,
        with_asymmetry=True,
    )

    train_mean, train_std, train_calib_windows = compute_calibration_stats(
        gate_bundle["train_windows"],
        gate_bundle["feature_columns"],
        args.calibration_sec,
    )
    test_mean, test_std, test_calib_windows = compute_calibration_stats(
        gate_bundle["test_windows"],
        gate_bundle["feature_columns"],
        args.calibration_sec,
    )

    gate_train_norm = normalize_with_calibration(
        gate_bundle["train_windows"], gate_bundle["feature_columns"], train_mean, train_std
    ).reset_index(drop=True)
    gate_test_norm = normalize_with_calibration(
        gate_bundle["test_windows"], gate_bundle["feature_columns"], test_mean, test_std
    ).reset_index(drop=True)

    lr_train_norm, lr_train_calibration_counts = normalize_windows_by_session(
        lr_bundle["train_windows"], lr_bundle["feature_columns"], args.calibration_sec
    )
    lr_test_norm, lr_test_calibration_counts = normalize_windows_by_session(
        lr_bundle["test_windows"], lr_bundle["feature_columns"], args.calibration_sec
    )

    gate_train = remap_active_vs_rest(gate_train_norm)
    _, p_active = fit_active_gate(gate_train, gate_test_norm, gate_bundle["feature_columns"])
    gate_active = p_active >= ACTIVE_THRESHOLD

    lr_pred = fit_left_right_model(lr_train_norm, lr_test_norm, lr_bundle["feature_columns"])

    final_pred = pd.Series(REST_LABEL, index=gate_test_norm.index, name="final_pred")
    final_pred.loc[gate_active] = lr_pred.loc[gate_active]

    y_true = gate_test_norm["label"].copy()
    y_true = y_true.where(y_true.isin([LEFT_LABEL, RIGHT_LABEL]), REST_LABEL)

    cm = pd.DataFrame(
        confusion_matrix(y_true, final_pred, labels=FINAL_LABELS),
        index=[f"true_{label}" for label in FINAL_LABELS],
        columns=[f"pred_{label}" for label in FINAL_LABELS],
    )

    active_true_mask = y_true.isin([LEFT_LABEL, RIGHT_LABEL])
    active_detection_rate = float(gate_active.mean())
    active_recall = float(gate_active.loc[active_true_mask].mean()) if active_true_mask.any() else float("nan")
    lr_accuracy_on_active_windows = float(
        accuracy_score(y_true.loc[active_true_mask], final_pred.loc[active_true_mask])
    ) if active_true_mask.any() else float("nan")
    lr_accuracy_when_detected = float(
        accuracy_score(
            y_true.loc[active_true_mask & gate_active],
            lr_pred.loc[active_true_mask & gate_active],
        )
    ) if (active_true_mask & gate_active).any() else float("nan")

    metrics = {
        "accuracy": float(accuracy_score(y_true, final_pred)),
        "macro_f1": float(f1_score(y_true, final_pred, average="macro")),
        "active_detection_rate": active_detection_rate,
        "active_recall": active_recall,
        "left_right_accuracy_on_active_windows": lr_accuracy_on_active_windows,
        "left_right_accuracy_when_detected": lr_accuracy_when_detected,
        "gate_train_calibration_windows": train_calib_windows,
        "gate_test_calibration_windows": test_calib_windows,
        "lr_train_calibration_windows": lr_train_calibration_counts,
        "lr_test_calibration_windows": lr_test_calibration_counts,
    }

    results_df = pd.DataFrame(
        [
            {
                "train_file": TRAIN_FILENAME,
                "test_file": TEST_FILENAME,
                "calibration_sec": args.calibration_sec,
                "gate_model": GATE_MODEL_NAME,
                "gate_threshold": ACTIVE_THRESHOLD,
                "left_right_model": LR_MODEL_NAME,
                "gate_selected_channels": ", ".join(gate_bundle["selected_channels"]),
                "left_right_selected_channels": ", ".join(lr_bundle["selected_channels"]),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "active_detection_rate": metrics["active_detection_rate"],
                "active_recall": metrics["active_recall"],
                "left_right_accuracy_on_active_windows": metrics["left_right_accuracy_on_active_windows"],
                "left_right_accuracy_when_detected": metrics["left_right_accuracy_when_detected"],
                "gate_train_calibration_windows": metrics["gate_train_calibration_windows"],
                "gate_test_calibration_windows": metrics["gate_test_calibration_windows"],
                "left_right_train_calibration_windows": metrics["lr_train_calibration_windows"],
                "left_right_test_calibration_windows": metrics["lr_test_calibration_windows"],
            }
        ]
    )

    results_csv = output_dir / "two_stage_lr_pipeline_results.csv"
    confusion_csv = output_dir / "two_stage_lr_pipeline_confusion_matrix.csv"
    predictions_csv = output_dir / "two_stage_lr_pipeline_predictions.csv"
    summary_md = output_dir / "two_stage_lr_pipeline_summary.md"

    results_df.to_csv(results_csv, index=False)
    cm.to_csv(confusion_csv)

    prediction_table = gate_test_norm.loc[
        :,
        ["filename", "start_time_sec", "end_time_sec", "label"],
    ].copy()
    prediction_table["true_label"] = y_true
    prediction_table["p_active"] = p_active
    prediction_table["gate_active"] = gate_active.astype(int)
    prediction_table["left_right_pred"] = lr_pred
    prediction_table["final_pred"] = final_pred
    prediction_table.to_csv(predictions_csv, index=False)

    report_lines = [
        "# Two-Stage LEFT/RIGHT Pipeline",
        "",
        f"- Train file: `{TRAIN_FILENAME}`",
        f"- Test file: `{TEST_FILENAME}`",
        f"- Calibration period: first `{args.calibration_sec:.0f}` seconds",
        f"- Gate model: `{GATE_MODEL_NAME}` with threshold `{ACTIVE_THRESHOLD:.1f}` (trained on `{TRAIN_FILENAME}`)",
        f"- LEFT vs RIGHT model: `{LR_MODEL_NAME}` (trained on `{', '.join(audit['filename'] for audit in lr_bundle['train_audits'])}`)",
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
        "## Interpretation",
        "",
        f"- Overall 3-class macro-F1: `{metrics['macro_f1']:.3f}`",
        f"- ACTIVE detection rate on all test windows: `{metrics['active_detection_rate']:.3f}`",
        f"- ACTIVE recall of the gate on true ACTIVE windows: `{metrics['active_recall']:.3f}`",
        f"- LEFT vs RIGHT accuracy on true ACTIVE windows after gating: `{metrics['left_right_accuracy_on_active_windows']:.3f}`",
        f"- LEFT vs RIGHT accuracy when the gate opens on true ACTIVE windows: `{metrics['left_right_accuracy_when_detected']:.3f}`",
    ]
    write_markdown(summary_md, "\n".join(report_lines))

    print(dataframe_to_markdown(results_df))
    print("\nConfusion matrix")
    print(dataframe_to_markdown(cm.reset_index().rename(columns={"index": "label"})))
    print("\nInterpretation")
    print(f"- Overall 3-class macro-F1: {metrics['macro_f1']:.3f}")
    print(f"- ACTIVE detection rate: {metrics['active_detection_rate']:.3f}")
    print(f"- Gate ACTIVE recall on true ACTIVE windows: {metrics['active_recall']:.3f}")
    print(f"- LEFT vs RIGHT accuracy on true ACTIVE windows after gating: {metrics['left_right_accuracy_on_active_windows']:.3f}")
    print(f"- LEFT vs RIGHT accuracy when gate opens on true ACTIVE windows: {metrics['left_right_accuracy_when_detected']:.3f}")
    print(f"\nWrote {results_csv}")
    print(f"Wrote {confusion_csv}")
    print(f"Wrote {predictions_csv}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
