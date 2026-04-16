from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.run_active_vs_rest_experiment import ACTIVE_LABEL, model_bank, remap_active_vs_rest
    from analysis.run_active_vs_rest_threshold_sweep import evaluate_thresholds
    from analysis.utils import audit_all_sessions, dataframe_to_markdown, ensure_output_dir, write_markdown
    from analysis.run_active_vs_rest_experiment import build_bundle_for_subset, resolve_left_right_subset
else:
    from .run_active_vs_rest_experiment import ACTIVE_LABEL, model_bank, remap_active_vs_rest
    from .run_active_vs_rest_threshold_sweep import evaluate_thresholds
    from .utils import audit_all_sessions, dataframe_to_markdown, ensure_output_dir, write_markdown
    from .run_active_vs_rest_experiment import build_bundle_for_subset, resolve_left_right_subset


TRAIN_FILENAME = "LR-2-27-26-(01).csv"
TEST_FILENAME = "LR-3-15-26-(04).csv"
DEFAULT_THRESHOLDS = [0.4, 0.5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed RandomForest ACTIVE probability thresholds cross-session."
    )
    parser.add_argument(
        "--threshold",
        action="append",
        type=float,
        default=None,
        help="Decision threshold(s) for P(ACTIVE). Defaults to 0.4 and 0.5.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for cross-session threshold outputs.",
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

    train_df = remap_active_vs_rest(bundle["train_windows"])
    test_df = remap_active_vs_rest(bundle["test_windows"])

    model = model_bank()["RandomForest"]
    X_train = train_df.loc[:, bundle["feature_columns"]].to_numpy(dtype=float)
    X_test = test_df.loc[:, bundle["feature_columns"]].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy()
    y_test = pd.Series(test_df["label"].to_numpy(), name="label")

    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    active_index = list(model.classes_).index(ACTIVE_LABEL)
    active_prob = pd.Series(probs[:, active_index], index=test_df.index, name="p_active")

    results_df = evaluate_thresholds(y_test, active_prob, thresholds)
    export_df = results_df.loc[
        :,
        ["threshold", "accuracy", "macro_f1", "active_precision", "active_recall", "misses", "false_triggers"],
    ].copy()
    results_csv = output_dir / "active_vs_rest_cross_session_threshold_comparison.csv"
    export_df.to_csv(results_csv, index=False)

    for _, row in results_df.iterrows():
        confusion_csv = output_dir / f"confusion_active_vs_rest_cross_session_threshold_{str(row['threshold']).replace('.', '_')}.csv"
        row["confusion_matrix"].to_csv(confusion_csv)

    baseline_row = export_df.loc[export_df["threshold"] == 0.5].iloc[0]
    tuned_row = export_df.loc[export_df["threshold"] == 0.4].iloc[0]

    summary_lines = [
        "# ACTIVE vs REST Cross-Session Threshold Comparison",
        "",
        f"- Train file: `{TRAIN_FILENAME}`",
        f"- Test file: `{TEST_FILENAME}`",
        f"- Model: `RandomForest`",
        f"- Features: `combined + asymmetry`",
        f"- Output CSV: `{results_csv}`",
        "",
        "## Results",
        "",
        dataframe_to_markdown(export_df),
        "",
        "## Comparison to Previous 0.5 Baseline",
        "",
        f"- ACTIVE recall change at `0.4` vs `0.5`: `{float(tuned_row['active_recall']) - float(baseline_row['active_recall']):+.3f}`",
        f"- False trigger change at `0.4` vs `0.5`: `{int(tuned_row['false_triggers']) - int(baseline_row['false_triggers']):+d}`",
        f"- Macro-F1 change at `0.4` vs `0.5`: `{float(tuned_row['macro_f1']) - float(baseline_row['macro_f1']):+.3f}`",
        "",
        "## Interpretation",
        "",
        f"- Threshold `0.4` ACTIVE recall: `{tuned_row['active_recall']:.3f}`",
        f"- Threshold `0.5` ACTIVE recall: `{baseline_row['active_recall']:.3f}`",
        f"- Threshold `0.4` false triggers: `{int(tuned_row['false_triggers'])}`",
        f"- Threshold `0.5` false triggers: `{int(baseline_row['false_triggers'])}`",
    ]

    summary_md = output_dir / "active_vs_rest_cross_session_threshold_comparison.md"
    write_markdown(summary_md, "\n".join(summary_lines))

    print(dataframe_to_markdown(export_df))
    print("\nComparison")
    print(
        f"- ACTIVE recall change at 0.4 vs 0.5: "
        f"{float(tuned_row['active_recall']) - float(baseline_row['active_recall']):+.3f}"
    )
    print(
        f"- False trigger change at 0.4 vs 0.5: "
        f"{int(tuned_row['false_triggers']) - int(baseline_row['false_triggers']):+d}"
    )
    print(
        f"- Macro-F1 change at 0.4 vs 0.5: "
        f"{float(tuned_row['macro_f1']) - float(baseline_row['macro_f1']):+.3f}"
    )

    acceptable = int(tuned_row["false_triggers"]) <= 30 and float(tuned_row["active_recall"]) >= 0.20
    print("\nInterpretation")
    if float(tuned_row["active_recall"]) > float(baseline_row["active_recall"]):
        print("- Recall improved with threshold 0.4.")
    else:
        print("- Recall did not improve with threshold 0.4.")
    if int(tuned_row["false_triggers"]) > int(baseline_row["false_triggers"]):
        print("- False triggers increased at threshold 0.4.")
    else:
        print("- False triggers did not increase at threshold 0.4.")
    if acceptable:
        print("- The threshold 0.4 operating point looks more usable than 0.5 for control, but it is still limited by cross-session shift.")
    else:
        print("- The threshold 0.4 operating point does not look comfortably acceptable for control in this cross-session setting.")

    print(f"\nWrote {results_csv}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
