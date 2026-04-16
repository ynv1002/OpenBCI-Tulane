from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.run_eeg_direction_base_up_calibrated import (
        CALIBRATION_SEC,
        FEATURE_MODE,
        OVERLAP,
        TASK_LEFT_VS_ALL,
        TASK_RIGHT_VS_ALL,
        TASKS,
        WINDOW_SEC,
        WITH_ASYMMETRY,
        append_coverage_note,
        compute_calibration_stats,
        fit_and_score,
        normalize_with_calibration,
        remap_task,
        task_note,
    )
    from analysis.utils import (
        LABEL_BASELINE,
        WINDOW_METADATA_COLUMNS,
        audit_all_sessions,
        build_windows_for_audit,
        dataframe_to_markdown,
        ensure_output_dir,
        select_model_channels,
        write_markdown,
    )
else:
    from .run_eeg_direction_base_up_calibrated import (
        CALIBRATION_SEC,
        FEATURE_MODE,
        OVERLAP,
        TASK_LEFT_VS_ALL,
        TASK_RIGHT_VS_ALL,
        TASKS,
        WINDOW_SEC,
        WITH_ASYMMETRY,
        append_coverage_note,
        compute_calibration_stats,
        fit_and_score,
        normalize_with_calibration,
        remap_task,
        task_note,
    )
    from .utils import (
        LABEL_BASELINE,
        WINDOW_METADATA_COLUMNS,
        audit_all_sessions,
        build_windows_for_audit,
        dataframe_to_markdown,
        ensure_output_dir,
        select_model_channels,
        write_markdown,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run calibrated cross-session EEG direction generalization for LEFT-vs-ALL and RIGHT-vs-ALL."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for cross-session EEG direction outputs.",
    )
    return parser.parse_args()


def build_session_windows(audit: Dict[str, object], selected_channels: Sequence[str], split_role: str) -> Dict[str, object]:
    windows = build_windows_for_audit(
        audit,
        selected_channels,
        split_role=split_role,
        window_sec=WINDOW_SEC,
        overlap=OVERLAP,
        feature_mode=FEATURE_MODE,
        with_asymmetry=WITH_ASYMMETRY,
    )
    feature_columns = [column for column in windows.columns if column not in WINDOW_METADATA_COLUMNS]
    mean, std, calibration_windows = compute_calibration_stats(windows, feature_columns, CALIBRATION_SEC)
    normalized = normalize_with_calibration(windows, feature_columns, mean, std)
    remaining = normalized.loc[
        (normalized["start_time_sec"] >= CALIBRATION_SEC) & (normalized["label"] != LABEL_BASELINE)
    ].copy()
    if remaining.empty:
        raise ValueError(f"No post-calibration windows remained for {audit['filename']}.")
    return {
        "frame": remaining,
        "feature_columns": feature_columns,
        "calibration_windows": calibration_windows,
    }


def safe_variance(series: pd.Series) -> float:
    if len(series) <= 1:
        return 0.0
    return float(series.var(ddof=0))


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    eeg_audits = [audit for audit in audit_all_sessions() if audit["family"] == "left_right"]

    rows: List[Dict[str, object]] = []
    for train_audit, test_audit in permutations(eeg_audits, 2):
        selected_channels, excluded_channels = select_model_channels([train_audit])
        if not selected_channels:
            raise ValueError(f"No usable channels survived for training session {train_audit['filename']}.")

        train_bundle = build_session_windows(train_audit, selected_channels, split_role="train")
        test_bundle = build_session_windows(test_audit, selected_channels, split_role="test")

        if train_bundle["feature_columns"] != test_bundle["feature_columns"]:
            raise ValueError("Feature columns differ between train and test after applying a shared channel set.")

        for task_name in [TASK_LEFT_VS_ALL, TASK_RIGHT_VS_ALL]:
            train_df, label_order = remap_task(train_bundle["frame"], task_name)
            test_df, _ = remap_task(test_bundle["frame"], task_name)
            metrics = fit_and_score(train_df, test_df, train_bundle["feature_columns"], label_order)
            note = append_coverage_note(
                task_note(task_name, metrics["macro_f1"], metrics["confusion_matrix"]),
                test_df,
                label_order,
            )

            confusion_name = (
                f"confusion_cross_session_{train_audit['filename'].replace('.', '_').replace('(', '').replace(')', '')}"
                f"_to_{test_audit['filename'].replace('.', '_').replace('(', '').replace(')', '')}_{task_name}.csv"
            )
            confusion_csv = output_dir / confusion_name
            metrics["confusion_matrix"].to_csv(confusion_csv)

            rows.append(
                {
                    "train_session": train_audit["filename"],
                    "test_session": test_audit["filename"],
                    "task": task_name,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "notes": note,
                    "selected_channels": ", ".join(selected_channels),
                    "excluded_channels": " | ".join(f"{key}: {value}" for key, value in excluded_channels.items()),
                    "train_calibration_windows": train_bundle["calibration_windows"],
                    "test_calibration_windows": test_bundle["calibration_windows"],
                    "train_class_counts": train_df["label"].value_counts().sort_index().to_dict(),
                    "test_class_counts": test_df["label"].value_counts().sort_index().to_dict(),
                    "confusion_matrix_csv": str(confusion_csv),
                }
            )

    results_df = pd.DataFrame(rows).sort_values(
        ["task", "macro_f1", "accuracy", "train_session", "test_session"],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)
    results_csv = output_dir / "eeg_direction_cross_session.csv"
    results_df.to_csv(results_csv, index=False)

    task_summary = (
        results_df.groupby("task")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_variance=("accuracy", safe_variance),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_variance=("macro_f1", safe_variance),
        )
        .reset_index()
        .sort_values(["macro_f1_mean", "accuracy_mean"], ascending=[False, False])
        .reset_index(drop=True)
    )
    best_task = task_summary.iloc[0]

    pair_summary = (
        results_df.groupby(["train_session", "test_session"], as_index=False)[["accuracy", "macro_f1"]]
        .mean()
        .sort_values(["macro_f1", "accuracy"], ascending=[False, False])
        .reset_index(drop=True)
    )

    recommendation = (
        "Use RIGHT vs ALL as the more stable cross-session direction model."
        if best_task["task"] == TASK_RIGHT_VS_ALL
        else "Use LEFT vs ALL as the more stable cross-session direction model."
    )

    summary_md = output_dir / "eeg_direction_cross_session.md"
    report_lines = [
        "# EEG Direction Cross-Session Summary",
        "",
        "- Ordered session pairs are evaluated independently: train on one session, test on a different unseen session.",
        f"- Calibration uses the first `{CALIBRATION_SEC:.0f}` seconds of each session separately.",
        f"- Windowing: `{WINDOW_SEC:.1f}` s, overlap `{OVERLAP:.1f}`.",
        f"- Features: `{FEATURE_MODE} + asymmetry`.",
        "- Model: `RandomForest`.",
        "- Tasks: `LEFT vs ALL`, `RIGHT vs ALL`.",
        "",
        f"- Output CSV: `{results_csv}`",
        "",
        "## Per-Pair Results",
        "",
        dataframe_to_markdown(results_df.loc[:, ["train_session", "test_session", "task", "accuracy", "macro_f1", "notes"]]),
        "",
        "## Task Summary",
        "",
        dataframe_to_markdown(task_summary),
        "",
        "## Session-Pair Summary",
        "",
        dataframe_to_markdown(pair_summary),
        "",
        "## Recommendation",
        "",
        f"- Best generalizing task: `{best_task['task']}`",
        f"- {TASK_LEFT_VS_ALL}: mean macro-F1 `{float(task_summary.loc[task_summary['task'] == TASK_LEFT_VS_ALL, 'macro_f1_mean'].iloc[0]):.3f}`, variance `{float(task_summary.loc[task_summary['task'] == TASK_LEFT_VS_ALL, 'macro_f1_variance'].iloc[0]):.4f}`",
        f"- {TASK_RIGHT_VS_ALL}: mean macro-F1 `{float(task_summary.loc[task_summary['task'] == TASK_RIGHT_VS_ALL, 'macro_f1_mean'].iloc[0]):.3f}`, variance `{float(task_summary.loc[task_summary['task'] == TASK_RIGHT_VS_ALL, 'macro_f1_variance'].iloc[0]):.4f}`",
        f"- Recommended direction model: {recommendation}",
    ]
    write_markdown(summary_md, "\n".join(report_lines))

    print("Task summary")
    print(dataframe_to_markdown(task_summary))
    print("\nSession-pair summary")
    print(dataframe_to_markdown(pair_summary))
    print("\nSummary")
    print(f"- Best generalizing task: {best_task['task']} (mean macro-F1={best_task['macro_f1_mean']:.3f})")
    for _, row in task_summary.iterrows():
        print(
            f"- {row['task']}: mean macro-F1={row['macro_f1_mean']:.3f}, "
            f"macro-F1 variance={row['macro_f1_variance']:.4f}"
        )
    print(f"- Recommended direction model: {recommendation}")
    print(f"\nWrote {results_csv}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
