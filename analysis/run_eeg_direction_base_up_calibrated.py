from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.utils import (
        WINDOW_METADATA_COLUMNS,
        LABEL_BASELINE,
        LABEL_REST,
        audit_all_sessions,
        build_windows_for_audit,
        dataframe_to_markdown,
        ensure_output_dir,
        select_model_channels,
        write_markdown,
    )
else:
    from .utils import (
        WINDOW_METADATA_COLUMNS,
        LABEL_BASELINE,
        LABEL_REST,
        audit_all_sessions,
        build_windows_for_audit,
        dataframe_to_markdown,
        ensure_output_dir,
        select_model_channels,
        write_markdown,
    )


CALIBRATION_SEC = 45.0
TRAIN_FRACTION = 0.70
WINDOW_SEC = 1.0
OVERLAP = 0.5
FEATURE_MODE = "combined"
WITH_ASYMMETRY = True
TASK_LEFT_VS_ALL = "left_vs_all"
TASK_RIGHT_VS_ALL = "right_vs_all"
TASK_LEFT_VS_RIGHT = "left_vs_right"
TASKS = [TASK_LEFT_VS_ALL, TASK_RIGHT_VS_ALL, TASK_LEFT_VS_RIGHT]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run calibrated within-session EEG direction baselines with 1.0 s combined+asymmetry features."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for calibrated EEG direction outputs.",
    )
    return parser.parse_args()


def calibration_subset(frame: pd.DataFrame, calibration_sec: float) -> pd.DataFrame:
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
) -> Tuple[pd.Series, pd.Series, int]:
    calib = calibration_subset(frame, calibration_sec)
    if calib.empty:
        raise ValueError("No baseline/rest calibration windows were found in the first 45 seconds.")
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


def chronological_split(frame: pd.DataFrame, train_fraction: float = TRAIN_FRACTION) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values(["start_time_sec", "start_sample"]).reset_index(drop=True)
    split_index = max(1, min(len(ordered) - 1, int(round(len(ordered) * train_fraction))))
    return (
        ordered.iloc[:split_index].copy().reset_index(drop=True),
        ordered.iloc[split_index:].copy().reset_index(drop=True),
    )


def remap_task(frame: pd.DataFrame, task_name: str) -> Tuple[pd.DataFrame, List[str]]:
    out = frame.copy()
    out = out[out["label"].isin(["LEFT", "RIGHT", LABEL_REST])].copy()
    if task_name == TASK_LEFT_VS_ALL:
        out["label"] = out["label"].map({"LEFT": "LEFT", "RIGHT": "ALL", LABEL_REST: "ALL"})
        return out.reset_index(drop=True), ["ALL", "LEFT"]
    if task_name == TASK_RIGHT_VS_ALL:
        out["label"] = out["label"].map({"LEFT": "ALL", "RIGHT": "RIGHT", LABEL_REST: "ALL"})
        return out.reset_index(drop=True), ["ALL", "RIGHT"]
    if task_name == TASK_LEFT_VS_RIGHT:
        out = out[out["label"].isin(["LEFT", "RIGHT"])].copy()
        return out.reset_index(drop=True), ["LEFT", "RIGHT"]
    raise ValueError(f"Unsupported task {task_name}")


def fit_and_score(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: Sequence[str],
    label_order: Sequence[str],
) -> Dict[str, object]:
    if train_df["label"].nunique() < 2:
        raise ValueError("Training split does not contain both classes for this task.")

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
    )
    X_train = train_df.loc[:, list(feature_columns)].to_numpy(dtype=float)
    X_test = test_df.loc[:, list(feature_columns)].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=list(label_order))
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in label_order],
        columns=[f"pred_{label}" for label in label_order],
    )
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, labels=list(label_order), average="macro", zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, labels=list(label_order), average="macro", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, labels=list(label_order), average="macro", zero_division=0)),
        "confusion_matrix": cm_df,
        "test_predictions": y_pred,
        "test_truth": y_test,
    }


def task_note(task_name: str, macro_f1: float, confusion_df: pd.DataFrame) -> str:
    if macro_f1 >= 0.85:
        base = "strong separation"
    elif macro_f1 >= 0.70:
        base = "usable separation"
    elif macro_f1 >= 0.50:
        base = "mixed performance"
    else:
        base = "weak separation"

    if task_name in {TASK_LEFT_VS_ALL, TASK_RIGHT_VS_ALL}:
        negative_label = "ALL"
        positive_label = "LEFT" if task_name == TASK_LEFT_VS_ALL else "RIGHT"
        fp = int(confusion_df.loc[f"true_{negative_label}", f"pred_{positive_label}"])
        tp = int(confusion_df.loc[f"true_{positive_label}", f"pred_{positive_label}"])
        if fp > tp:
            return f"{base}; false triggers exceed hits"
    if task_name == TASK_LEFT_VS_RIGHT:
        left_correct = int(confusion_df.loc["true_LEFT", "pred_LEFT"])
        right_correct = int(confusion_df.loc["true_RIGHT", "pred_RIGHT"])
        if abs(left_correct - right_correct) > max(5, int(0.15 * (left_correct + right_correct + 1))):
            return f"{base}; directional bias"
    return base


def append_coverage_note(note: str, test_df: pd.DataFrame, label_order: Sequence[str]) -> str:
    present = set(test_df["label"].unique())
    missing = [label for label in label_order if label not in present]
    if not missing:
        return note
    return f"{note}; test missing {', '.join(missing)}"


def build_session_bundle(audit: Dict[str, object]) -> Dict[str, object]:
    selected_channels, excluded_channels = select_model_channels([audit])
    if not selected_channels:
        raise ValueError(f"No usable channels survived for {audit['filename']}.")

    windows = build_windows_for_audit(
        audit,
        selected_channels,
        split_role="within_session",
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

    train_all, test_all = chronological_split(remaining, TRAIN_FRACTION)
    return {
        "selected_channels": selected_channels,
        "excluded_channels": excluded_channels,
        "feature_columns": feature_columns,
        "calibration_windows": calibration_windows,
        "train_all": train_all,
        "test_all": test_all,
    }


def evaluate_session_task(
    audit: Dict[str, object],
    bundle: Dict[str, object],
    task_name: str,
    output_dir: Path,
) -> Dict[str, object]:
    train_df, label_order = remap_task(bundle["train_all"], task_name)
    test_df, _ = remap_task(bundle["test_all"], task_name)
    metrics = fit_and_score(train_df, test_df, bundle["feature_columns"], label_order)
    note = append_coverage_note(
        task_note(task_name, metrics["macro_f1"], metrics["confusion_matrix"]),
        test_df,
        label_order,
    )

    confusion_csv = output_dir / f"confusion_{audit['filename'].replace('.', '_').replace('(', '').replace(')', '')}_{task_name}.csv"
    metrics["confusion_matrix"].to_csv(confusion_csv)

    return {
        "dataset": "EEG",
        "session_name": audit["filename"],
        "task": task_name,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "notes": note,
        "selected_channels": ", ".join(bundle["selected_channels"]),
        "excluded_channels": " | ".join(f"{key}: {value}" for key, value in bundle["excluded_channels"].items()),
        "calibration_windows": bundle["calibration_windows"],
        "window_sec": WINDOW_SEC,
        "overlap": OVERLAP,
        "feature_mode": FEATURE_MODE,
        "with_asymmetry": WITH_ASYMMETRY,
        "train_class_counts": train_df["label"].value_counts().sort_index().to_dict(),
        "test_class_counts": test_df["label"].value_counts().sort_index().to_dict(),
        "confusion_matrix_csv": str(confusion_csv),
    }


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    audits = [audit for audit in audit_all_sessions() if audit["family"] == "left_right"]

    rows: List[Dict[str, object]] = []
    for audit in audits:
        bundle = build_session_bundle(audit)
        for task_name in TASKS:
            rows.append(evaluate_session_task(audit, bundle, task_name, output_dir))

    results_df = pd.DataFrame(rows).sort_values(
        ["task", "macro_f1", "accuracy", "session_name"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    results_csv = output_dir / "eeg_direction_base_up_calibrated.csv"
    results_df.to_csv(results_csv, index=False)

    task_summary = (
        results_df.groupby("task")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
        )
        .reset_index()
    )
    task_summary = task_summary.sort_values(["macro_f1_mean", "accuracy_mean"], ascending=[False, False]).reset_index(drop=True)
    best_task = task_summary.iloc[0]

    session_summary = (
        results_df.groupby("session_name", as_index=False)[["macro_f1", "accuracy"]]
        .mean()
        .sort_values("macro_f1", ascending=False)
        .reset_index(drop=True)
    )

    consistency_lines = [
        f"- `{row['task']}`: mean macro-F1 `{row['macro_f1_mean']:.3f}`, std `{row['macro_f1_std']:.3f}`"
        for _, row in task_summary.iterrows()
    ]

    if best_task["task"] == TASK_LEFT_VS_RIGHT:
        recommendation = "Use LEFT vs RIGHT as the primary direction model once activity is already known."
    elif best_task["task"] == TASK_LEFT_VS_ALL:
        recommendation = "Use LEFT vs ALL as the most reliable calibrated direction detector."
    else:
        recommendation = "Use RIGHT vs ALL as the most reliable calibrated direction detector."

    summary_md = output_dir / "eeg_direction_base_up_calibrated.md"
    report_lines = [
        "# EEG Direction Base-Up Calibrated Summary",
        "",
        "- Sessions are evaluated independently.",
        f"- Calibration uses the first `{CALIBRATION_SEC:.0f}` seconds of BASELINE/REST windows.",
        f"- Remaining data is split chronologically: first `{TRAIN_FRACTION:.0%}` train, last `{1.0 - TRAIN_FRACTION:.0%}` test.",
        f"- Windowing: `{WINDOW_SEC:.1f}` s, overlap `{OVERLAP:.1f}`.",
        f"- Features: `{FEATURE_MODE} + asymmetry`.",
        "- Model: `RandomForest`.",
        "",
        f"- Output CSV: `{results_csv}`",
        "",
        "## Per-Run Results",
        "",
        dataframe_to_markdown(results_df.loc[:, ["session_name", "task", "accuracy", "macro_f1", "precision", "recall", "notes"]]),
        "",
        "## Task Averages",
        "",
        dataframe_to_markdown(task_summary),
        "",
        "## Session Averages",
        "",
        dataframe_to_markdown(session_summary),
        "",
        "## Summary",
        "",
        f"- Best performing task: `{best_task['task']}`",
        *consistency_lines,
        f"- Recommended direction model: {recommendation}",
    ]
    write_markdown(summary_md, "\n".join(report_lines))

    print("Task averages")
    print(dataframe_to_markdown(task_summary))
    print("\nSession averages")
    print(dataframe_to_markdown(session_summary))
    print("\nSummary")
    print(f"- Best performing task: {best_task['task']} (mean macro-F1={best_task['macro_f1_mean']:.3f})")
    print("- Consistency across sessions:")
    for line in consistency_lines:
        print(f"  {line}")
    print(f"- Recommended direction model: {recommendation}")
    print(f"\nWrote {results_csv}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
