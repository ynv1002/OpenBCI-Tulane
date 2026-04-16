from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

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
    from analysis.run_active_vs_rest_threshold_sweep import evaluate_thresholds
    from analysis.utils import audit_all_sessions, dataframe_to_markdown, ensure_output_dir, write_markdown
    from analysis.run_active_vs_rest_experiment import build_bundle_for_subset, resolve_left_right_subset
else:
    from .run_active_vs_rest_cross_session_calibrated import (
        DEFAULT_CALIBRATION_SEC,
        TEST_FILENAME,
        TRAIN_FILENAME,
        compute_calibration_stats,
        normalize_with_calibration,
    )
    from .run_active_vs_rest_experiment import ACTIVE_LABEL, model_bank, remap_active_vs_rest
    from .run_active_vs_rest_threshold_sweep import evaluate_thresholds
    from .utils import audit_all_sessions, dataframe_to_markdown, ensure_output_dir, write_markdown
    from .run_active_vs_rest_experiment import build_bundle_for_subset, resolve_left_right_subset


DEFAULT_THRESHOLDS = [0.2, 0.3, 0.4, 0.5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep thresholds for the calibrated cross-session ACTIVE vs REST RandomForest."
    )
    parser.add_argument(
        "--threshold",
        action="append",
        type=float,
        default=None,
        help="Decision threshold(s) for P(ACTIVE). Defaults to 0.2, 0.3, 0.4, 0.5.",
    )
    parser.add_argument("--calibration-sec", type=float, default=DEFAULT_CALIBRATION_SEC)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for calibrated threshold sweep outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = sorted(set(args.threshold or DEFAULT_THRESHOLDS))
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

    sweep_df = evaluate_thresholds(y_test, active_prob, thresholds)
    export_df = sweep_df.loc[
        :,
        ["threshold", "accuracy", "macro_f1", "active_precision", "active_recall", "misses", "false_triggers"],
    ].copy()
    results_csv = output_dir / "active_vs_rest_calibrated_threshold_sweep.csv"
    export_df.to_csv(results_csv, index=False)

    for _, row in sweep_df.iterrows():
        confusion_csv = output_dir / (
            f"confusion_active_vs_rest_calibrated_threshold_sweep_{str(row['threshold']).replace('.', '_')}.csv"
        )
        row["confusion_matrix"].to_csv(confusion_csv)

    best_macro = export_df.sort_values(["macro_f1", "accuracy"], ascending=[False, False]).iloc[0]
    best_recall = export_df.sort_values(["active_recall", "macro_f1"], ascending=[False, False]).iloc[0]
    balance_df = export_df.copy()
    balance_df["balance_score"] = (
        balance_df["macro_f1"] - 0.001 * balance_df["false_triggers"] + 0.001 * balance_df["active_recall"] * 100.0
    )
    best_balance = balance_df.sort_values(["balance_score", "macro_f1"], ascending=[False, False]).iloc[0]

    summary_md = output_dir / "active_vs_rest_calibrated_threshold_sweep.md"
    report_lines = [
        "# ACTIVE vs REST Calibrated Threshold Sweep",
        "",
        f"- Train file: `{TRAIN_FILENAME}`",
        f"- Test file: `{TEST_FILENAME}`",
        f"- Model: `RandomForest`",
        f"- Features: `combined + asymmetry`",
        f"- Calibration period: first `{args.calibration_sec:.0f}` seconds",
        f"- Train calibration windows used: `{train_calib_windows}`",
        f"- Test calibration windows used: `{test_calib_windows}`",
        f"- Output CSV: `{results_csv}`",
        "",
        "## Results",
        "",
        dataframe_to_markdown(export_df),
        "",
        "## Interpretation",
        "",
        f"- Best macro-F1 threshold: `{best_macro['threshold']:.1f}` with macro-F1 `{best_macro['macro_f1']:.3f}`",
        f"- Best recall threshold: `{best_recall['threshold']:.1f}` with ACTIVE recall `{best_recall['active_recall']:.3f}`",
        f"- Best practical balance: `{best_balance['threshold']:.1f}` with macro-F1 `{best_balance['macro_f1']:.3f}`, ACTIVE recall `{best_balance['active_recall']:.3f}`, and `{int(best_balance['false_triggers'])}` false triggers",
    ]
    write_markdown(summary_md, "\n".join(report_lines))

    print(dataframe_to_markdown(export_df))
    print("\nInterpretation")
    print(f"- Best macro-F1 threshold: {best_macro['threshold']:.1f}")
    print(f"- Best recall threshold: {best_recall['threshold']:.1f}")
    print(
        f"- Best practical balance: {best_balance['threshold']:.1f} "
        f"(macro-F1={best_balance['macro_f1']:.3f}, recall={best_balance['active_recall']:.3f}, "
        f"false_triggers={int(best_balance['false_triggers'])})"
    )
    print(f"\nWrote {results_csv}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
