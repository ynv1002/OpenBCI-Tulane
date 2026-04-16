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
    from analysis.event_utils import (
        EVENT_LABELS_4STATE,
        EventWindowConfig,
        JawEventConfig,
        build_jaw_event_labels_for_audit,
    )
    from analysis.utils import (
        DEFAULT_OVERLAP,
        DEFAULT_WINDOW_SEC,
        WINDOW_METADATA_COLUMNS,
        audit_all_sessions,
        build_windows_for_audit,
        dataframe_to_markdown,
        ensure_output_dir,
        extract_window_features,
        select_model_channels,
        write_markdown,
    )
else:
    from .event_utils import (
        EVENT_LABELS_4STATE,
        EventWindowConfig,
        JawEventConfig,
        build_jaw_event_labels_for_audit,
    )
    from .utils import (
        DEFAULT_OVERLAP,
        DEFAULT_WINDOW_SEC,
        WINDOW_METADATA_COLUMNS,
        audit_all_sessions,
        build_windows_for_audit,
        dataframe_to_markdown,
        ensure_output_dir,
        extract_window_features,
        select_model_channels,
        write_markdown,
    )


COARSE_TASK_WINDOW_SEC = DEFAULT_WINDOW_SEC
COARSE_TASK_OVERLAP = DEFAULT_OVERLAP
FEATURE_MODE = "combined"
WITH_ASYMMETRY = False
TRAIN_FRACTION = 0.70

TASK_JAW_VS_REST = "jaw_vs_rest"
TASK_JAW_4STATE = "jaw_4state"
TASK_LR_2CLASS = "left_vs_right"
TASK_LR_3CLASS = "left_vs_right_vs_rest"
TASK_ACTIVE_VS_REST = "active_vs_rest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline within-session RandomForest evaluations across jaw and EEG tasks."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for baseline within-session outputs.",
    )
    return parser.parse_args()


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
    sort_columns: Sequence[str] = ("start_time_sec", "start_sample"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values(list(sort_columns)).reset_index(drop=True)
    split_index = max(1, min(len(ordered) - 1, int(round(len(ordered) * train_fraction))))
    train_df = ordered.iloc[:split_index].copy().reset_index(drop=True)
    test_df = ordered.iloc[split_index:].copy().reset_index(drop=True)
    return train_df, test_df


def fit_random_forest(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: Sequence[str],
    label_order: Sequence[str],
) -> Dict[str, object]:
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
    per_class_recall = {}
    for index, label in enumerate(label_order):
        denom = float(cm[index].sum())
        per_class_recall[label] = float(cm[index, index] / denom) if denom else 0.0
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": cm_df,
        "per_class_recall": per_class_recall,
        "train_class_counts": train_df["label"].value_counts().sort_index().to_dict(),
        "test_class_counts": test_df["label"].value_counts().sort_index().to_dict(),
    }


def task_note(task_name: str, macro_f1: float, per_class_recall: Dict[str, float]) -> str:
    if macro_f1 >= 0.85:
        base = "strong separation"
    elif macro_f1 >= 0.70:
        base = "usable separation"
    elif macro_f1 >= 0.50:
        base = "mixed performance"
    else:
        base = "weak separation"

    if not per_class_recall:
        return base

    hardest_label = min(per_class_recall, key=per_class_recall.get)
    hardest_recall = per_class_recall[hardest_label]

    if task_name == TASK_JAW_4STATE and hardest_label in {"ONSET", "OFFSET"} and hardest_recall < 0.5:
        return f"{base}; transition states weak"
    if "REST" in per_class_recall:
        non_rest = [value for label, value in per_class_recall.items() if label != "REST"]
        if non_rest and per_class_recall["REST"] - float(np.mean(non_rest)) > 0.20:
            return f"{base}; rest dominates"
    if task_name == TASK_LR_2CLASS:
        recall_values = list(per_class_recall.values())
        if max(recall_values) - min(recall_values) > 0.20:
            return f"{base}; directional bias"
    if hardest_recall < 0.40:
        return f"{base}; weakest class={hardest_label}"
    return base


def build_coarse_session_windows(audit: Dict[str, object]) -> Dict[str, object]:
    selected_channels, excluded_channels = select_model_channels([audit])
    if not selected_channels:
        raise ValueError(f"No usable channels survived for {audit['filename']}.")
    frame = build_windows_for_audit(
        audit,
        selected_channels,
        split_role="within_session",
        window_sec=COARSE_TASK_WINDOW_SEC,
        overlap=COARSE_TASK_OVERLAP,
        feature_mode=FEATURE_MODE,
        with_asymmetry=WITH_ASYMMETRY,
    )
    feature_columns = [column for column in frame.columns if column not in WINDOW_METADATA_COLUMNS]
    return {
        "frame": frame,
        "feature_columns": feature_columns,
        "selected_channels": selected_channels,
        "excluded_channels": excluded_channels,
    }


def remap_coarse_task(frame: pd.DataFrame, task_name: str) -> Tuple[pd.DataFrame, List[str]]:
    out = frame.copy()
    if task_name == TASK_JAW_VS_REST:
        out = out[out["label"].isin(["HOLD", "REPEATED", "REST"])].copy()
        out["label"] = out["label"].map({"HOLD": "JAW", "REPEATED": "JAW", "REST": "REST"})
        return out.reset_index(drop=True), ["REST", "JAW"]
    if task_name == TASK_LR_2CLASS:
        out = out[out["label"].isin(["LEFT", "RIGHT"])].copy()
        return out.reset_index(drop=True), ["LEFT", "RIGHT"]
    if task_name == TASK_LR_3CLASS:
        out = out[out["label"].isin(["LEFT", "RIGHT", "REST"])].copy()
        return out.reset_index(drop=True), ["REST", "LEFT", "RIGHT"]
    if task_name == TASK_ACTIVE_VS_REST:
        out = out[out["label"].isin(["LEFT", "RIGHT", "REST"])].copy()
        out["label"] = out["label"].map({"LEFT": "ACTIVE", "RIGHT": "ACTIVE", "REST": "REST"})
        return out.reset_index(drop=True), ["REST", "ACTIVE"]
    raise ValueError(f"Unknown coarse task: {task_name}")


def build_event_windows_combined(
    sample_frame: pd.DataFrame,
    signal_columns: Sequence[str],
    window_config: EventWindowConfig,
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    for filename, session_df in sample_frame.groupby("filename", sort=False):
        session_df = session_df.reset_index(drop=True)
        if len(session_df) < 2:
            continue
        time_values = session_df["time_sec"].to_numpy(dtype=float)
        fs_hz = 1.0 / float(np.median(np.diff(time_values)))
        window_samples = max(1, int(round(window_config.window_sec * fs_hz)))
        hop_samples = max(1, int(round(window_samples * (1.0 - window_config.overlap))))
        if len(session_df) < window_samples:
            continue

        signal_matrix = session_df.loc[:, list(signal_columns)].to_numpy(dtype=float)
        for start in range(0, len(session_df) - window_samples + 1, hop_samples):
            end = start + window_samples
            center = start + window_samples // 2
            window = signal_matrix[start:end, :]
            row = {
                "filename": filename,
                "start_sample": int(session_df.iloc[start]["sample_row"]),
                "end_sample": int(session_df.iloc[end - 1]["sample_row"]),
                "center_time_sec": float(session_df.iloc[center]["time_sec"]),
                "event_label": str(session_df.iloc[center]["event_label"]),
            }
            row.update(
                extract_window_features(
                    window,
                    signal_columns,
                    fs_hz=fs_hz,
                    feature_mode=FEATURE_MODE,
                    with_asymmetry=WITH_ASYMMETRY,
                )
            )
            rows.append(row)

    frame = pd.DataFrame(rows)
    metadata_columns = {"filename", "start_sample", "end_sample", "center_time_sec", "event_label"}
    feature_columns = [column for column in frame.columns if column not in metadata_columns]
    return {"frame": frame, "feature_columns": feature_columns}


def run_jaw_4state_task(audit: Dict[str, object]) -> Dict[str, object]:
    selected_channels, excluded_channels = select_model_channels([audit])
    if not selected_channels:
        raise ValueError(f"No usable channels survived for {audit['filename']}.")

    event_bundle = build_jaw_event_labels_for_audit(audit, selected_channels, JawEventConfig())
    signal_columns = [f"signal_{channel}" for channel in event_bundle["selected_channels"]]
    window_bundle = build_event_windows_combined(
        event_bundle["sample_frame"],
        signal_columns,
        EventWindowConfig(),
    )
    frame = window_bundle["frame"].copy()
    frame["label"] = frame["event_label"]
    train_df, test_df = chronological_split(frame, sort_columns=("center_time_sec", "start_sample"))
    metrics = fit_random_forest(train_df, test_df, window_bundle["feature_columns"], EVENT_LABELS_4STATE)
    return {
        "dataset": "jaw",
        "session_name": audit["filename"],
        "task": TASK_JAW_4STATE,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "notes": task_note(TASK_JAW_4STATE, metrics["macro_f1"], metrics["per_class_recall"]),
        "selected_channels": ", ".join(selected_channels),
        "window_sec": EventWindowConfig().window_sec,
        "overlap": EventWindowConfig().overlap,
        "train_class_counts": metrics["train_class_counts"],
        "test_class_counts": metrics["test_class_counts"],
    }


def run_coarse_task(audit: Dict[str, object], dataset_name: str, task_name: str) -> Dict[str, object]:
    bundle = build_coarse_session_windows(audit)
    frame, label_order = remap_coarse_task(bundle["frame"], task_name)
    train_df, test_df = chronological_split(frame)
    metrics = fit_random_forest(train_df, test_df, bundle["feature_columns"], label_order)
    return {
        "dataset": dataset_name,
        "session_name": audit["filename"],
        "task": task_name,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "notes": task_note(task_name, metrics["macro_f1"], metrics["per_class_recall"]),
        "selected_channels": ", ".join(bundle["selected_channels"]),
        "window_sec": COARSE_TASK_WINDOW_SEC,
        "overlap": COARSE_TASK_OVERLAP,
        "train_class_counts": metrics["train_class_counts"],
        "test_class_counts": metrics["test_class_counts"],
    }


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    audits = audit_all_sessions()

    rows: List[Dict[str, object]] = []
    for audit in audits:
        family = audit["family"]
        if family == "jaw":
            for task_name in [TASK_JAW_VS_REST]:
                rows.append(run_coarse_task(audit, "jaw", task_name))
            rows.append(run_jaw_4state_task(audit))
        elif family == "left_right":
            for task_name in [TASK_LR_2CLASS, TASK_LR_3CLASS, TASK_ACTIVE_VS_REST]:
                rows.append(run_coarse_task(audit, "EEG", task_name))

    summary_df = pd.DataFrame(rows).sort_values(
        ["dataset", "task", "macro_f1", "accuracy"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)
    summary_csv = output_dir / "baseline_within_session_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    best_row = summary_df.sort_values(["macro_f1", "accuracy"], ascending=[False, False]).iloc[0]
    task_averages = (
        summary_df.groupby("task", as_index=False)[["accuracy", "macro_f1", "precision", "recall"]]
        .mean()
        .sort_values("macro_f1", ascending=False)
        .reset_index(drop=True)
    )
    dataset_averages = (
        summary_df.groupby("dataset", as_index=False)[["accuracy", "macro_f1", "precision", "recall"]]
        .mean()
        .sort_values("macro_f1", ascending=False)
        .reset_index(drop=True)
    )
    strongest_dataset = dataset_averages.iloc[0]
    weakest_task = task_averages.sort_values("macro_f1", ascending=True).iloc[0]

    jaw_vs_rest_avg = task_averages.loc[task_averages["task"] == TASK_JAW_VS_REST]
    jaw_4state_avg = task_averages.loc[task_averages["task"] == TASK_JAW_4STATE]
    eeg_direction_avg = task_averages.loc[task_averages["task"] == TASK_LR_2CLASS]
    eeg_activity_avg = task_averages.loc[task_averages["task"] == TASK_ACTIVE_VS_REST]

    jaw_role = "jaw is the strongest simple control signal" if not jaw_vs_rest_avg.empty and float(jaw_vs_rest_avg.iloc[0]["macro_f1"]) >= 0.75 else "jaw is usable but needs more task-specific tuning"
    if not jaw_4state_avg.empty and float(jaw_4state_avg.iloc[0]["macro_f1"]) < 0.65:
        jaw_role += "; 4-state jaw decoding is notably weaker than binary jaw vs rest"

    eeg_direction_role = (
        "EEG direction is viable within-session" if not eeg_direction_avg.empty and float(eeg_direction_avg.iloc[0]["macro_f1"]) >= 0.65 else "EEG direction is only moderate within-session"
    )
    eeg_activity_role = (
        "EEG activity is usable as a coarse gate" if not eeg_activity_avg.empty and float(eeg_activity_avg.iloc[0]["macro_f1"]) >= 0.65 else "EEG activity is the weakest role and should not be the first-stage gate without adaptation"
    )

    summary_md = output_dir / "baseline_within_session_summary.md"
    report_lines = [
        "# Baseline Within-Session Summary",
        "",
        "- Coarse tasks use 2.0 s windows with 0.5 overlap.",
        "- Jaw 4-state uses the existing event-scale windowing so onset/offset labels remain meaningful.",
        "- BASELINE windows are excluded from the coarse tasks to keep label targets aligned with the requested task definitions.",
        "",
        f"- Output CSV: `{summary_csv}`",
        "",
        "## Best Overall Run",
        "",
        f"- Dataset: `{best_row['dataset']}`",
        f"- Session: `{best_row['session_name']}`",
        f"- Task: `{best_row['task']}`",
        f"- Accuracy: `{best_row['accuracy']:.3f}`",
        f"- Macro-F1: `{best_row['macro_f1']:.3f}`",
        f"- Notes: `{best_row['notes']}`",
        "",
        "## Average By Task",
        "",
        dataframe_to_markdown(task_averages),
        "",
        "## Average By Dataset",
        "",
        dataframe_to_markdown(dataset_averages),
        "",
        "## Recommendations",
        "",
        f"- Jaw: {jaw_role}.",
        f"- EEG direction: {eeg_direction_role}.",
        f"- EEG activity: {eeg_activity_role}.",
    ]
    write_markdown(summary_md, "\n".join(report_lines))

    print("Best overall run")
    print(
        f"- {best_row['dataset']} | {best_row['session_name']} | {best_row['task']} | "
        f"accuracy={best_row['accuracy']:.3f} | macro-F1={best_row['macro_f1']:.3f}"
    )
    print("\nAverage performance per task")
    print(dataframe_to_markdown(task_averages))
    print("\nAverage performance per dataset")
    print(dataframe_to_markdown(dataset_averages))
    print("\nSummary")
    print(f"- Strongest dataset: {strongest_dataset['dataset']} (avg macro-F1={strongest_dataset['macro_f1']:.3f})")
    print(f"- Weakest task: {weakest_task['task']} (avg macro-F1={weakest_task['macro_f1']:.3f})")
    print(f"- Jaw recommendation: {jaw_role}.")
    print(f"- EEG direction recommendation: {eeg_direction_role}.")
    print(f"- EEG activity recommendation: {eeg_activity_role}.")
    print(f"\nWrote {summary_csv}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
