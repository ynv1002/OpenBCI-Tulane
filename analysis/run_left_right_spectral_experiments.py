from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.utils import (
        DEFAULT_OVERLAP,
        DEFAULT_WINDOW_SEC,
        FEATURE_MODES,
        LABEL_REST,
        WINDOW_METADATA_COLUMNS,
        audit_all_sessions,
        audits_by_family,
        build_windows_for_audit,
        dataframe_to_markdown,
        ensure_output_dir,
        select_model_channels,
        write_markdown,
    )
else:
    from .utils import (
        DEFAULT_OVERLAP,
        DEFAULT_WINDOW_SEC,
        FEATURE_MODES,
        LABEL_REST,
        WINDOW_METADATA_COLUMNS,
        audit_all_sessions,
        audits_by_family,
        build_windows_for_audit,
        dataframe_to_markdown,
        ensure_output_dir,
        select_model_channels,
        write_markdown,
    )


LEFT_RIGHT_FAMILY = "left_right"
TASK_TYPES = ("3_class", "2_class")
DATASET_CONDITIONS = ("all_sessions", "clean_sessions")
ASYMMETRY_OPTIONS = (False, True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run left/right EEG experiments with time, spectral, and asymmetry feature variants."
    )
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC)
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for experiment outputs.",
    )
    return parser.parse_args()


def model_bank() -> Dict[str, object]:
    return {
        "LDA": Pipeline([("scaler", StandardScaler()), ("model", LinearDiscriminantAnalysis())]),
        "LogisticRegression": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=3000, class_weight="balanced"))]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        ),
    }


def filtered_left_right_audits(
    all_audits: Sequence[Dict[str, object]],
    dataset_condition: str,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    family_audits = audits_by_family(all_audits)[LEFT_RIGHT_FAMILY]
    if dataset_condition == "all_sessions":
        filtered = family_audits
    elif dataset_condition == "clean_sessions":
        filtered = [audit for audit in family_audits if "3-8-26" not in str(audit["filename"])]
    else:
        raise ValueError(f"Unknown dataset_condition: {dataset_condition}")

    if len(filtered) < 2:
        raise ValueError(f"Dataset condition '{dataset_condition}' leaves fewer than 2 sessions.")
    return filtered[:-1], filtered[-1]


def build_dataset_bundle(
    train_audits: Sequence[Dict[str, object]],
    test_audit: Dict[str, object],
    window_sec: float,
    overlap: float,
    feature_mode: str,
    with_asymmetry: bool,
) -> Dict[str, object]:
    selected_channels, excluded_channels = select_model_channels(train_audits)
    if not selected_channels:
        raise ValueError("No left/right channels survived the training-session quality screen.")

    train_frames = [
        build_windows_for_audit(
            audit,
            selected_channels,
            split_role="train",
            window_sec=window_sec,
            overlap=overlap,
            feature_mode=feature_mode,
            with_asymmetry=with_asymmetry,
        )
        for audit in train_audits
    ]
    test_frame = build_windows_for_audit(
        test_audit,
        selected_channels,
        split_role="test",
        window_sec=window_sec,
        overlap=overlap,
        feature_mode=feature_mode,
        with_asymmetry=with_asymmetry,
    )
    train_frame = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()
    feature_columns = [column for column in train_frame.columns if column not in WINDOW_METADATA_COLUMNS]
    asymmetry_columns = [column for column in feature_columns if column.startswith(("mu_asym_", "beta_asym_"))]

    return {
        "train_windows": train_frame,
        "test_windows": test_frame,
        "feature_columns": feature_columns,
        "asymmetry_columns": asymmetry_columns,
        "selected_channels": selected_channels,
        "excluded_channels": excluded_channels,
        "train_audits": list(train_audits),
        "test_audit": test_audit,
    }


def apply_task(frame: pd.DataFrame, task_type: str) -> Tuple[pd.DataFrame, List[str]]:
    if task_type == "3_class":
        filtered = frame[frame["label"].isin(["LEFT", "RIGHT", LABEL_REST])].copy()
        labels = [LABEL_REST, "LEFT", "RIGHT"]
    elif task_type == "2_class":
        filtered = frame[frame["label"].isin(["LEFT", "RIGHT"])].copy()
        labels = ["LEFT", "RIGHT"]
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    return filtered.reset_index(drop=True), labels


def fit_and_score(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: Sequence[str],
    labels: Sequence[str],
) -> Dict[str, Dict[str, object]]:
    X_train = train_df.loc[:, list(feature_columns)].to_numpy(dtype=float)
    X_test = test_df.loc[:, list(feature_columns)].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    results: Dict[str, Dict[str, object]] = {}
    for model_name, model in model_bank().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm = pd.DataFrame(
            confusion_matrix(y_test, y_pred, labels=list(labels)),
            index=[f"true_{label}" for label in labels],
            columns=[f"pred_{label}" for label in labels],
        )
        results[model_name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
            "confusion_matrix": cm,
        }
    return results


def summarize_interpretation(results_df: pd.DataFrame) -> List[str]:
    best_by_feature = (
        results_df.groupby("feature_mode")["macro_f1"].max().sort_values(ascending=False).to_dict()
    )
    best_with_asym = results_df[results_df["with_asymmetry"]]["macro_f1"].max()
    best_without_asym = results_df[~results_df["with_asymmetry"]]["macro_f1"].max()
    best_all_sessions = results_df[results_df["dataset_condition"] == "all_sessions"]["macro_f1"].max()
    best_clean_sessions = results_df[results_df["dataset_condition"] == "clean_sessions"]["macro_f1"].max()
    best_3_class = results_df[results_df["task_type"] == "3_class"]["macro_f1"].max()
    best_2_class = results_df[results_df["task_type"] == "2_class"]["macro_f1"].max()

    lines = [
        "Interpretation",
        f"- Best macro-F1 by feature mode: {best_by_feature}",
    ]

    if best_by_feature.get("combined", float("-inf")) > best_by_feature.get("time_only", float("-inf")):
        lines.append("- Spectral features helped when combined with time-domain features.")
    elif best_by_feature.get("spectral_only", float("-inf")) > best_by_feature.get("time_only", float("-inf")):
        lines.append("- Spectral features helped, but mainly as a spectral-only representation.")
    else:
        lines.append("- Spectral features did not beat the best time-only configuration.")

    if pd.notna(best_with_asym) and pd.notna(best_without_asym):
        if best_with_asym > best_without_asym:
            lines.append("- Asymmetry improved the best observed result.")
        elif best_with_asym < best_without_asym:
            lines.append("- Asymmetry hurt the best observed result.")
        else:
            lines.append("- Asymmetry made no difference to the best observed result.")

    if best_clean_sessions > best_all_sessions:
        lines.append("- Removing the March 8 sessions improved peak performance.")
    elif best_clean_sessions < best_all_sessions:
        lines.append("- Removing the March 8 sessions did not improve the best result.")
    else:
        lines.append("- Removing the March 8 sessions left the best result unchanged.")

    if best_2_class > best_3_class:
        lines.append("- The 2-class LEFT vs RIGHT task performed better than the 3-class task.")
    elif best_2_class < best_3_class:
        lines.append("- The 3-class task matched or exceeded the 2-class task.")
    else:
        lines.append("- The 2-class and 3-class tasks tied at the best observed result.")
    return lines


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    audits = audit_all_sessions()
    results_rows: List[Dict[str, object]] = []

    for dataset_condition in DATASET_CONDITIONS:
        train_audits, test_audit = filtered_left_right_audits(audits, dataset_condition)
        for feature_mode in FEATURE_MODES:
            for with_asymmetry in ASYMMETRY_OPTIONS:
                dataset_bundle = build_dataset_bundle(
                    train_audits=train_audits,
                    test_audit=test_audit,
                    window_sec=args.window_sec,
                    overlap=args.overlap,
                    feature_mode=feature_mode,
                    with_asymmetry=with_asymmetry,
                )
                for task_type in TASK_TYPES:
                    train_df, labels = apply_task(dataset_bundle["train_windows"], task_type)
                    test_df, _ = apply_task(dataset_bundle["test_windows"], task_type)
                    feature_columns = dataset_bundle["feature_columns"]
                    results = fit_and_score(train_df, test_df, feature_columns, labels)

                    for model_name, metrics in results.items():
                        config_stub = "_".join(
                            [
                                dataset_condition,
                                task_type,
                                feature_mode,
                                "with_asym" if with_asymmetry else "no_asym",
                                model_name,
                            ]
                        )
                        confusion_csv = output_dir / f"confusion_left_right_feature_grid_{config_stub}.csv"
                        metrics["confusion_matrix"].to_csv(confusion_csv)

                        results_rows.append(
                            {
                                "family": LEFT_RIGHT_FAMILY,
                                "dataset_condition": dataset_condition,
                                "task_type": task_type,
                                "feature_mode": feature_mode,
                                "with_asymmetry": with_asymmetry,
                                "effective_asymmetry_feature_count": len(dataset_bundle["asymmetry_columns"]),
                                "model_name": model_name,
                                "train_files": ", ".join(audit["filename"] for audit in train_audits),
                                "test_file": test_audit["filename"],
                                "selected_channels": ", ".join(dataset_bundle["selected_channels"]),
                                "excluded_channels": " | ".join(
                                    f"{key}: {value}" for key, value in dataset_bundle["excluded_channels"].items()
                                ),
                                "window_sec": args.window_sec,
                                "overlap": args.overlap,
                                "feature_count": len(feature_columns),
                                "train_window_count": len(train_df),
                                "test_window_count": len(test_df),
                                "accuracy": metrics["accuracy"],
                                "macro_f1": metrics["macro_f1"],
                                "confusion_matrix_csv": str(confusion_csv),
                            }
                        )

                    best_model_name, best_metrics = max(
                        results.items(),
                        key=lambda item: (item[1]["macro_f1"], item[1]["accuracy"]),
                    )
                    print(
                        f"[{dataset_condition} | {task_type} | {feature_mode} | asym={with_asymmetry}] "
                        f"best={best_model_name} accuracy={best_metrics['accuracy']:.3f} macro_f1={best_metrics['macro_f1']:.3f}"
                    )

    results_df = pd.DataFrame(results_rows).sort_values(
        ["macro_f1", "accuracy"],
        ascending=[False, False],
    ).reset_index(drop=True)
    summary_csv = output_dir / "left_right_spectral_feature_grid_results.csv"
    results_df.to_csv(summary_csv, index=False)

    top_columns = [
        "dataset_condition",
        "task_type",
        "feature_mode",
        "with_asymmetry",
        "model_name",
        "feature_count",
        "accuracy",
        "macro_f1",
        "train_files",
        "test_file",
    ]
    ranked_df = results_df.loc[:, top_columns].head(12).copy()
    interpretation_lines = summarize_interpretation(results_df)

    report_lines = [
        "# Left/Right Spectral Feature Experiment",
        "",
        f"- Output CSV: `{summary_csv}`",
        f"- Total model runs: `{len(results_df)}`",
        f"- Configuration grid size: `{len(DATASET_CONDITIONS) * len(FEATURE_MODES) * len(ASYMMETRY_OPTIONS) * len(TASK_TYPES)}`",
        "",
        "## Top Runs By Macro-F1",
        "",
        dataframe_to_markdown(ranked_df),
        "",
        "## Interpretation",
        "",
        *interpretation_lines[1:],
    ]
    report_path = output_dir / "left_right_spectral_feature_grid_summary.md"
    write_markdown(report_path, "\n".join(report_lines))

    print("\nTop configurations by macro-F1")
    print(ranked_df.to_string(index=False))
    print()
    for line in interpretation_lines:
        print(line)
    print(f"\nWrote {summary_csv}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
