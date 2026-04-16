from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

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
    from analysis.run_active_vs_rest_experiment import remap_active_vs_rest
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
    from .run_active_vs_rest_experiment import remap_active_vs_rest
    from .utils import audit_all_sessions, dataframe_to_markdown, ensure_output_dir, write_markdown


DEFAULT_WINDOW_SIZES = [0.5, 1.0, 2.0]
DEFAULT_OVERLAP = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the calibrated two-stage LEFT/RIGHT pipeline across window sizes."
    )
    parser.add_argument(
        "--window-sec",
        action="append",
        type=float,
        default=None,
        help="Window size(s) in seconds. Repeat to test multiple values. Defaults to 0.5, 1.0, 2.0.",
    )
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    parser.add_argument("--calibration-sec", type=float, default=DEFAULT_CALIBRATION_SEC)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for window-size experiment outputs.",
    )
    return parser.parse_args()


def run_two_stage_for_window_size(
    audits: List[dict],
    window_sec: float,
    overlap: float,
    calibration_sec: float,
) -> tuple[dict, pd.DataFrame]:
    gate_subset_audits = resolve_left_right_subset(audits, [TRAIN_FILENAME, TEST_FILENAME])
    gate_bundle = build_bundle_for_subset(
        gate_subset_audits,
        window_sec=window_sec,
        overlap=overlap,
        with_asymmetry=True,
    )
    lr_subset_audits = resolve_left_right_subset(audits, None)
    lr_bundle = build_bundle_for_subset(
        lr_subset_audits,
        window_sec=window_sec,
        overlap=overlap,
        with_asymmetry=True,
    )

    train_mean, train_std, train_calib_windows = compute_calibration_stats(
        gate_bundle["train_windows"], gate_bundle["feature_columns"], calibration_sec
    )
    test_mean, test_std, test_calib_windows = compute_calibration_stats(
        gate_bundle["test_windows"], gate_bundle["feature_columns"], calibration_sec
    )

    gate_train_norm = normalize_with_calibration(
        gate_bundle["train_windows"], gate_bundle["feature_columns"], train_mean, train_std
    ).reset_index(drop=True)
    gate_test_norm = normalize_with_calibration(
        gate_bundle["test_windows"], gate_bundle["feature_columns"], test_mean, test_std
    ).reset_index(drop=True)

    lr_train_norm, lr_train_calibration_counts = normalize_windows_by_session(
        lr_bundle["train_windows"], lr_bundle["feature_columns"], calibration_sec
    )
    lr_test_norm, lr_test_calibration_counts = normalize_windows_by_session(
        lr_bundle["test_windows"], lr_bundle["feature_columns"], calibration_sec
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

    row = {
        "window_sec": window_sec,
        "overlap": overlap,
        "train_file": TRAIN_FILENAME,
        "test_file": TEST_FILENAME,
        "gate_threshold": ACTIVE_THRESHOLD,
        "accuracy": float(accuracy_score(y_true, final_pred)),
        "macro_f1": float(f1_score(y_true, final_pred, average="macro")),
        "active_detection_rate": active_detection_rate,
        "active_recall": active_recall,
        "left_right_accuracy_on_active_windows": lr_accuracy_on_active_windows,
        "left_right_accuracy_when_detected": lr_accuracy_when_detected,
        "gate_selected_channels": ", ".join(gate_bundle["selected_channels"]),
        "left_right_selected_channels": ", ".join(lr_bundle["selected_channels"]),
        "gate_train_calibration_windows": train_calib_windows,
        "gate_test_calibration_windows": test_calib_windows,
        "left_right_train_calibration_windows": lr_train_calibration_counts,
        "left_right_test_calibration_windows": lr_test_calibration_counts,
    }
    return row, cm


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    window_sizes = sorted(set(args.window_sec or DEFAULT_WINDOW_SIZES))
    audits = audit_all_sessions()

    rows: List[dict] = []
    confusion_paths: List[tuple[float, Path]] = []
    for window_sec in window_sizes:
        row, cm = run_two_stage_for_window_size(
            audits=audits,
            window_sec=window_sec,
            overlap=args.overlap,
            calibration_sec=args.calibration_sec,
        )
        rows.append(row)
        confusion_csv = output_dir / f"two_stage_lr_pipeline_confusion_matrix_{str(window_sec).replace('.', '_')}.csv"
        cm.to_csv(confusion_csv)
        confusion_paths.append((window_sec, confusion_csv))
        print(
            f"[window={window_sec:.1f}s] accuracy={row['accuracy']:.3f} "
            f"macro_f1={row['macro_f1']:.3f} "
            f"active_recall={row['active_recall']:.3f} "
            f"gate_open_rate={row['active_detection_rate']:.3f} "
            f"lr_acc_active={row['left_right_accuracy_on_active_windows']:.3f}"
        )

    results_df = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=[False, False]).reset_index(drop=True)
    results_csv = output_dir / "window_size_experiment.csv"
    results_df.to_csv(results_csv, index=False)

    best_macro = results_df.sort_values(["macro_f1", "accuracy"], ascending=[False, False]).iloc[0]
    best_recall = results_df.sort_values(["active_recall", "macro_f1"], ascending=[False, False]).iloc[0]
    balance_df = results_df.copy()
    balance_df["balance_score"] = (
        balance_df["macro_f1"]
        + 0.20 * balance_df["active_recall"]
        + 0.10 * balance_df["left_right_accuracy_when_detected"]
    )
    best_balance = balance_df.sort_values(["balance_score", "macro_f1"], ascending=[False, False]).iloc[0]

    summary_md = output_dir / "window_size_experiment.md"
    report_lines = [
        "# Two-Stage Window Size Experiment",
        "",
        f"- Train file: `{TRAIN_FILENAME}`",
        f"- Test file: `{TEST_FILENAME}`",
        f"- Calibration period: first `{args.calibration_sec:.0f}` seconds",
        f"- Gate threshold: `{ACTIVE_THRESHOLD:.1f}`",
        f"- Output CSV: `{results_csv}`",
        "",
        "## Results",
        "",
        dataframe_to_markdown(
            results_df.loc[
                :,
                [
                    "window_sec",
                    "accuracy",
                    "macro_f1",
                    "active_recall",
                    "active_detection_rate",
                    "left_right_accuracy_on_active_windows",
                    "left_right_accuracy_when_detected",
                ],
            ]
        ),
        "",
        "## Interpretation",
        "",
        f"- Best macro-F1 window: `{best_macro['window_sec']:.1f}` s with macro-F1 `{best_macro['macro_f1']:.3f}`",
        f"- Best ACTIVE recall window: `{best_recall['window_sec']:.1f}` s with ACTIVE recall `{best_recall['active_recall']:.3f}`",
        f"- Best practical balance: `{best_balance['window_sec']:.1f}` s with macro-F1 `{best_balance['macro_f1']:.3f}`, ACTIVE recall `{best_balance['active_recall']:.3f}`, and LEFT/RIGHT accuracy when detected `{best_balance['left_right_accuracy_when_detected']:.3f}`",
        "",
        "## Confusion Matrices",
        "",
    ]
    for window_sec, path in confusion_paths:
        report_lines.append(f"- `{window_sec:.1f}` s: `{path}`")
    write_markdown(summary_md, "\n".join(report_lines))

    print("\nResults")
    print(
        dataframe_to_markdown(
            results_df.loc[
                :,
                [
                    "window_sec",
                    "accuracy",
                    "macro_f1",
                    "active_recall",
                    "active_detection_rate",
                    "left_right_accuracy_on_active_windows",
                    "left_right_accuracy_when_detected",
                ],
            ]
        )
    )
    print("\nInterpretation")
    print(f"- Best macro-F1 window: {best_macro['window_sec']:.1f}s")
    print(f"- Best ACTIVE recall window: {best_recall['window_sec']:.1f}s")
    print(
        f"- Best practical balance: {best_balance['window_sec']:.1f}s "
        f"(macro-F1={best_balance['macro_f1']:.3f}, ACTIVE recall={best_balance['active_recall']:.3f}, "
        f"LEFT/RIGHT when detected={best_balance['left_right_accuracy_when_detected']:.3f})"
    )
    print(f"\nWrote {results_csv}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
