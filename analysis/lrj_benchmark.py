from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .lrj_dataset import (
    DEFAULT_LRJ_FEATURE_MODE,
    DEFAULT_LRJ_FS,
    DEFAULT_LRJ_OVERLAP,
    DEFAULT_LRJ_WINDOW_SEC,
    DEFAULT_LRJ_WITH_ASYMMETRY,
    LRJSessionSpec,
    build_lrj_session_record,
    lrj_session_specs,
)
from .utils import (
    extract_window_features,
    preprocess_session_signals,
    select_model_channels,
    ensure_output_dir,
    dataframe_to_markdown,
    write_json,
)


MOVEMENT_LABELS = ["LEFT", "RIGHT", "JAW"]
COUNT_LABELS = [1, 2, 3, 4, 5, 6]
LRJ_WINDOW_METADATA_COLUMNS = [
    "dataset_track",
    "subject",
    "display_name",
    "filename",
    "file_path",
    "split_role",
    "label",
    "expected_count",
    "observed_count",
    "count_match",
    "count_error",
    "count_filter_family",
    "marker_code",
    "marker_pair_type",
    "trial_index_overall",
    "trial_index_within_label",
    "trial_uid",
    "interval_start_sample",
    "interval_end_sample",
    "interval_start_time_sec",
    "interval_end_time_sec",
    "interval_duration_sec",
    "window_index_within_trial",
    "window_source",
    "start_sample",
    "end_sample",
    "start_time_sec",
    "end_time_sec",
    "actual_window_sec",
    "window_sec",
    "overlap",
    "issues",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the shared offline LRJ benchmark across the Ben and Yaniv count-ramp sessions."
    )
    parser.add_argument("--fs", type=float, default=DEFAULT_LRJ_FS, help="Authoritative sample rate for LRJ reporting.")
    parser.add_argument("--window-sec", type=float, default=DEFAULT_LRJ_WINDOW_SEC, help="Window length in seconds.")
    parser.add_argument("--overlap", type=float, default=DEFAULT_LRJ_OVERLAP, help="Fractional window overlap.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for LRJ benchmark outputs.",
    )
    return parser.parse_args()


def model_bank() -> dict[str, object]:
    return {
        "LDA": Pipeline([("scaler", StandardScaler()), ("model", LinearDiscriminantAnalysis())]),
        "LogisticRegression": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=3000, class_weight="balanced"))]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
        ),
    }


def _probability_columns(prefix: str, classes: Sequence[Any]) -> list[str]:
    return [f"{prefix}_prob_{label}" for label in classes]


def load_lrj_session_records(session_specs: Sequence[LRJSessionSpec], fs_hz: float) -> list[dict[str, Any]]:
    records = [build_lrj_session_record(session_spec=spec, fs_hz=fs_hz) for spec in session_specs]
    for record in records:
        if record["hard_issues"]:
            raise ValueError(
                f"{record['display_name']} has hard structural issues and cannot be used for shared LRJ modeling: "
                + " | ".join(record["hard_issues"])
            )
    return records


def build_lrj_interval_dataset(session_records: Sequence[dict[str, Any]]) -> pd.DataFrame:
    interval_frames: list[pd.DataFrame] = []
    for record in session_records:
        frame = record["trial_summary_df"].copy()
        frame.insert(0, "subject", record["subject"])
        frame.insert(1, "display_name", record["display_name"])
        frame.insert(2, "filename", record["filename"])
        frame.insert(3, "file_path", str(Path(record["file_path"]).resolve()))
        frame.insert(4, "dataset_track", "lrj")
        frame["trial_uid"] = frame["subject"] + "::" + frame["filename"] + "::trial_" + frame["trial_index_overall"].astype(str)
        interval_frames.append(frame)
    return pd.concat(interval_frames, ignore_index=True)


def build_lrj_windows_for_session(
    session_record: dict[str, Any],
    selected_channels: Sequence[str],
    window_sec: float,
    overlap: float,
    feature_mode: str = DEFAULT_LRJ_FEATURE_MODE,
    with_asymmetry: bool = DEFAULT_LRJ_WITH_ASYMMETRY,
    split_role: str = "",
) -> pd.DataFrame:
    raw_df = session_record["raw_df"]
    fs_hz = float(session_record["fs_hz"])
    processed_by_family = {
        family_key: preprocess_session_signals(raw_df, family_key, selected_channels, fs_hz)
        for family_key in ("left_right", "jaw")
    }

    window_samples = max(1, int(round(window_sec * fs_hz)))
    hop_samples = max(1, int(round(window_samples * (1.0 - overlap))))
    trial_summary_df = session_record["trial_summary_df"]
    rows: list[dict[str, Any]] = []

    for trial in trial_summary_df.to_dict(orient="records"):
        family_key = str(trial["count_filter_family"])
        signal_frame = processed_by_family[family_key]
        interval_start = int(trial["start_sample"])
        interval_end = int(trial["end_sample"])
        interval_len = interval_end - interval_start + 1

        if interval_len < window_samples:
            window_starts = [interval_start]
            window_source = "full_interval_fallback"
        else:
            window_starts = list(range(interval_start, interval_end - window_samples + 2, hop_samples))
            window_source = "sliding"

        for window_index, window_start in enumerate(window_starts, start=1):
            if window_source == "sliding":
                window_end = window_start + window_samples - 1
            else:
                window_end = interval_end
            window = signal_frame.iloc[window_start : window_end + 1].to_numpy(dtype=float)
            row = {
                "dataset_track": "lrj",
                "subject": session_record["subject"],
                "display_name": session_record["display_name"],
                "filename": session_record["filename"],
                "file_path": str(Path(session_record["file_path"]).resolve()),
                "split_role": split_role,
                "label": str(trial["label"]),
                "expected_count": int(trial["expected_count"]),
                "observed_count": int(trial["observed_count"]),
                "count_match": bool(trial["count_match"]),
                "count_error": int(trial["count_error"]),
                "count_filter_family": family_key,
                "marker_code": int(trial["marker_code"]),
                "marker_pair_type": str(trial["marker_pair_type"]),
                "trial_index_overall": int(trial["trial_index_overall"]),
                "trial_index_within_label": int(trial["trial_index_within_label"]),
                "trial_uid": f"{session_record['subject']}::{session_record['filename']}::trial_{int(trial['trial_index_overall'])}",
                "interval_start_sample": interval_start,
                "interval_end_sample": interval_end,
                "interval_start_time_sec": float(trial["start_time_sec"]),
                "interval_end_time_sec": float(trial["end_time_sec"]),
                "interval_duration_sec": float(trial["duration_sec"]),
                "window_index_within_trial": window_index,
                "window_source": window_source,
                "start_sample": int(window_start),
                "end_sample": int(window_end),
                "start_time_sec": float(window_start / fs_hz),
                "end_time_sec": float(window_end / fs_hz),
                "actual_window_sec": float((window_end - window_start + 1) / fs_hz),
                "window_sec": float(window_sec),
                "overlap": float(overlap),
                "issues": "" if pd.isna(trial["issues"]) else str(trial["issues"]),
            }
            row.update(
                extract_window_features(
                    window,
                    selected_channels,
                    fs_hz=fs_hz,
                    feature_mode=feature_mode,
                    with_asymmetry=with_asymmetry,
                )
            )
            rows.append(row)
    return pd.DataFrame(rows)


def fit_task_predictions(
    model: object,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    prediction_column: str,
    probability_prefix: str,
) -> tuple[pd.DataFrame, list[Any]]:
    fitted = clone(model)
    X_train = train_df.loc[:, list(feature_columns)].to_numpy(dtype=float)
    X_test = test_df.loc[:, list(feature_columns)].to_numpy(dtype=float)
    y_train = train_df[target_column].to_numpy()
    fitted.fit(X_train, y_train)
    predictions = fitted.predict(X_test)
    classes = list(getattr(fitted, "classes_", sorted(pd.Series(y_train).unique().tolist())))
    probability_df = pd.DataFrame(
        fitted.predict_proba(X_test),
        columns=_probability_columns(probability_prefix, classes),
    )
    output = pd.DataFrame(
        {
            prediction_column: predictions,
        }
    )
    return pd.concat([output, probability_df], axis=1), classes


def aggregate_interval_predictions(
    window_predictions_df: pd.DataFrame,
    movement_classes: Sequence[str],
    count_classes: Sequence[int],
) -> pd.DataFrame:
    movement_prob_columns = _probability_columns("movement", movement_classes)
    count_prob_columns = _probability_columns("count", count_classes)
    metadata_columns = [
        "model_name",
        "fold_name",
        "subject",
        "display_name",
        "filename",
        "file_path",
        "label",
        "expected_count",
        "observed_count",
        "count_match",
        "count_error",
        "count_filter_family",
        "marker_code",
        "marker_pair_type",
        "trial_index_overall",
        "trial_index_within_label",
        "trial_uid",
        "interval_start_sample",
        "interval_end_sample",
        "interval_start_time_sec",
        "interval_end_time_sec",
        "interval_duration_sec",
        "issues",
    ]

    rows: list[dict[str, Any]] = []
    grouped = window_predictions_df.groupby(["model_name", "fold_name", "trial_uid"], sort=False)
    for (_, _, _), group_df in grouped:
        movement_probs = group_df[movement_prob_columns].mean()
        count_probs = group_df[count_prob_columns].mean()
        predicted_movement = max(movement_classes, key=lambda label: float(movement_probs[f"movement_prob_{label}"]))
        predicted_count = max(count_classes, key=lambda label: float(count_probs[f"count_prob_{label}"]))
        base_row = group_df.iloc[0][metadata_columns].to_dict()
        base_row["window_count"] = int(len(group_df))
        base_row["predicted_label"] = predicted_movement
        base_row["predicted_count"] = int(predicted_count)
        base_row["joint_exact_match"] = bool(
            (predicted_movement == base_row["label"]) and (int(predicted_count) == int(base_row["expected_count"]))
        )
        for column in movement_prob_columns:
            base_row[column] = float(movement_probs[column])
        for column in count_prob_columns:
            base_row[column] = float(count_probs[column])
        rows.append(base_row)
    return pd.DataFrame(rows)


def compute_metric_bundle(prediction_df: pd.DataFrame) -> dict[str, float]:
    return {
        "movement_accuracy": float(accuracy_score(prediction_df["label"], prediction_df["predicted_label"])),
        "movement_macro_f1": float(f1_score(prediction_df["label"], prediction_df["predicted_label"], labels=MOVEMENT_LABELS, average="macro", zero_division=0)),
        "count_accuracy": float(accuracy_score(prediction_df["expected_count"], prediction_df["predicted_count"])),
        "count_macro_f1": float(f1_score(prediction_df["expected_count"], prediction_df["predicted_count"], labels=COUNT_LABELS, average="macro", zero_division=0)),
        "joint_exact_match": float(prediction_df["joint_exact_match"].mean()),
    }


def build_results_row(
    evaluation_scope: str,
    fold_name: str,
    model_name: str,
    train_files: Sequence[str],
    test_file: str,
    selected_channels: Sequence[str],
    feature_columns: Sequence[str],
    window_sec: float,
    overlap: float,
    train_window_count: int,
    test_window_count: int,
    train_interval_count: int,
    test_interval_count: int,
    window_metrics: dict[str, float],
    interval_metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "evaluation_scope": evaluation_scope,
        "fold_name": fold_name,
        "model_name": model_name,
        "train_files": ", ".join(train_files),
        "test_file": test_file,
        "selected_channels": ", ".join(selected_channels),
        "feature_count": int(len(feature_columns)),
        "feature_mode": DEFAULT_LRJ_FEATURE_MODE,
        "with_asymmetry": bool(DEFAULT_LRJ_WITH_ASYMMETRY),
        "window_sec": float(window_sec),
        "overlap": float(overlap),
        "train_window_count": int(train_window_count),
        "test_window_count": int(test_window_count),
        "train_interval_count": int(train_interval_count),
        "test_interval_count": int(test_interval_count),
        "movement_window_accuracy": window_metrics["movement_accuracy"],
        "movement_window_macro_f1": window_metrics["movement_macro_f1"],
        "count_window_accuracy": window_metrics["count_accuracy"],
        "count_window_macro_f1": window_metrics["count_macro_f1"],
        "joint_window_exact": window_metrics["joint_exact_match"],
        "movement_interval_accuracy": interval_metrics["movement_accuracy"],
        "movement_interval_macro_f1": interval_metrics["movement_macro_f1"],
        "count_interval_accuracy": interval_metrics["count_accuracy"],
        "count_interval_macro_f1": interval_metrics["count_macro_f1"],
        "joint_interval_exact": interval_metrics["joint_exact_match"],
    }


def build_summary_markdown(
    output_path: Path,
    session_records: Sequence[dict[str, Any]],
    selected_channels: Sequence[str],
    interval_dataset_df: pd.DataFrame,
    window_dataset_df: pd.DataFrame,
    results_df: pd.DataFrame,
) -> None:
    session_table = pd.DataFrame(
        [
            {
                "subject": record["subject"],
                "filename": record["filename"],
                "interval_count": int(len(record["trial_summary_df"])),
                "count_matches": int(record["trial_summary_df"]["count_match"].sum()),
                "count_channels": ", ".join(record["count_channels"]),
                "soft_warnings": " | ".join(record["soft_warnings"]) if record["soft_warnings"] else "",
            }
            for record in session_records
        ]
    )
    fold_results_df = results_df[results_df["evaluation_scope"] == "fold"].copy()
    pooled_results_df = results_df[results_df["evaluation_scope"] == "pooled"].copy()

    best_movement_row = pooled_results_df.sort_values(
        ["movement_interval_macro_f1", "movement_interval_accuracy"],
        ascending=[False, False],
    ).iloc[0]
    best_count_row = pooled_results_df.sort_values(
        ["count_interval_macro_f1", "count_interval_accuracy"],
        ascending=[False, False],
    ).iloc[0]
    best_joint_row = pooled_results_df.sort_values(["joint_interval_exact"], ascending=[False]).iloc[0]

    lines = [
        "# Shared LRJ Offline Benchmark",
        "",
        f"- Sessions: `{', '.join(record['filename'] for record in session_records)}`",
        f"- Shared selected channels: `{', '.join(selected_channels)}`",
        f"- Interval count: `{len(interval_dataset_df)}`",
        f"- Window count: `{len(window_dataset_df)}`",
        f"- Feature mode: `{DEFAULT_LRJ_FEATURE_MODE}`",
        f"- With asymmetry: `{DEFAULT_LRJ_WITH_ASYMMETRY}`",
        "",
        "## Session Summary",
        "",
        dataframe_to_markdown(session_table, include_index=False),
        "",
        "## Per-Fold Results",
        "",
        dataframe_to_markdown(
            fold_results_df[
                [
                    "fold_name",
                    "model_name",
                    "movement_window_macro_f1",
                    "count_window_macro_f1",
                    "joint_window_exact",
                    "movement_interval_macro_f1",
                    "count_interval_macro_f1",
                    "joint_interval_exact",
                ]
            ],
            include_index=False,
        ),
        "",
        "## Pooled LOSO Results",
        "",
        dataframe_to_markdown(
            pooled_results_df[
                [
                    "model_name",
                    "movement_window_macro_f1",
                    "count_window_macro_f1",
                    "joint_window_exact",
                    "movement_interval_macro_f1",
                    "count_interval_macro_f1",
                    "joint_interval_exact",
                ]
            ],
            include_index=False,
        ),
        "",
        "## Recommendation",
        "",
        f"- Best movement model: `{best_movement_row['model_name']}` with pooled interval macro-F1 `{best_movement_row['movement_interval_macro_f1']:.3f}`",
        f"- Best count model: `{best_count_row['model_name']}` with pooled interval macro-F1 `{best_count_row['count_interval_macro_f1']:.3f}`",
        f"- Best joint model: `{best_joint_row['model_name']}` with pooled interval exact-match `{best_joint_row['joint_interval_exact']:.3f}`",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    session_specs = lrj_session_specs()
    session_records = load_lrj_session_records(session_specs, fs_hz=float(args.fs))
    selected_channels, excluded_channels = select_model_channels(session_records)
    if not selected_channels:
        raise ValueError("No shared LRJ channels survived the cross-session quality screen.")

    interval_dataset_df = build_lrj_interval_dataset(session_records)
    window_frames: list[pd.DataFrame] = []
    for record in session_records:
        window_frames.append(
            build_lrj_windows_for_session(
                session_record=record,
                selected_channels=selected_channels,
                window_sec=float(args.window_sec),
                overlap=float(args.overlap),
                split_role=record["subject"].lower(),
            )
        )
    window_dataset_df = pd.concat(window_frames, ignore_index=True)
    feature_columns = [column for column in window_dataset_df.columns if column not in LRJ_WINDOW_METADATA_COLUMNS]

    results_rows: list[dict[str, Any]] = []
    window_prediction_frames: list[pd.DataFrame] = []
    interval_prediction_frames: list[pd.DataFrame] = []

    for test_record in session_records:
        train_records = [record for record in session_records if record["filename"] != test_record["filename"]]
        train_filenames = [record["filename"] for record in train_records]
        fold_name = f"{'+'.join(train_filenames)}_to_{test_record['filename']}"

        train_window_df = window_dataset_df[window_dataset_df["filename"].isin(train_filenames)].reset_index(drop=True)
        test_window_df = window_dataset_df[window_dataset_df["filename"] == test_record["filename"]].reset_index(drop=True)

        for model_name, model in model_bank().items():
            movement_predictions_df, movement_classes = fit_task_predictions(
                model=model,
                train_df=train_window_df,
                test_df=test_window_df,
                feature_columns=feature_columns,
                target_column="label",
                prediction_column="predicted_label",
                probability_prefix="movement",
            )
            count_predictions_df, count_classes = fit_task_predictions(
                model=model,
                train_df=train_window_df,
                test_df=test_window_df,
                feature_columns=feature_columns,
                target_column="expected_count",
                prediction_column="predicted_count",
                probability_prefix="count",
            )

            prediction_df = pd.concat([test_window_df.reset_index(drop=True), movement_predictions_df, count_predictions_df], axis=1)
            prediction_df["fold_name"] = fold_name
            prediction_df["model_name"] = model_name
            prediction_df["joint_exact_match"] = (
                (prediction_df["predicted_label"] == prediction_df["label"])
                & (prediction_df["predicted_count"].astype(int) == prediction_df["expected_count"].astype(int))
            )
            window_prediction_frames.append(prediction_df)

            interval_prediction_df = aggregate_interval_predictions(
                prediction_df,
                movement_classes=movement_classes,
                count_classes=count_classes,
            )
            interval_prediction_frames.append(interval_prediction_df)

            window_metrics = compute_metric_bundle(prediction_df)
            interval_metrics = compute_metric_bundle(interval_prediction_df)
            results_rows.append(
                build_results_row(
                    evaluation_scope="fold",
                    fold_name=fold_name,
                    model_name=model_name,
                    train_files=train_filenames,
                    test_file=test_record["filename"],
                    selected_channels=selected_channels,
                    feature_columns=feature_columns,
                    window_sec=float(args.window_sec),
                    overlap=float(args.overlap),
                    train_window_count=len(train_window_df),
                    test_window_count=len(test_window_df),
                    train_interval_count=sum(len(record["trial_summary_df"]) for record in train_records),
                    test_interval_count=len(test_record["trial_summary_df"]),
                    window_metrics=window_metrics,
                    interval_metrics=interval_metrics,
                )
            )
            print(
                f"[{fold_name} | {model_name}] "
                f"movement_interval_f1={interval_metrics['movement_macro_f1']:.3f} "
                f"count_interval_f1={interval_metrics['count_macro_f1']:.3f} "
                f"joint_interval_exact={interval_metrics['joint_exact_match']:.3f}"
            )

    window_predictions_df = pd.concat(window_prediction_frames, ignore_index=True)
    interval_predictions_df = pd.concat(interval_prediction_frames, ignore_index=True)

    for model_name in model_bank():
        pooled_window_df = window_predictions_df[window_predictions_df["model_name"] == model_name].reset_index(drop=True)
        pooled_interval_df = interval_predictions_df[interval_predictions_df["model_name"] == model_name].reset_index(drop=True)
        results_rows.append(
            build_results_row(
                evaluation_scope="pooled",
                fold_name="pooled_LOSO",
                model_name=model_name,
                train_files=[record["filename"] for record in session_records],
                test_file="pooled_LOSO",
                selected_channels=selected_channels,
                feature_columns=feature_columns,
                window_sec=float(args.window_sec),
                overlap=float(args.overlap),
                train_window_count=int(len(window_dataset_df)),
                test_window_count=int(len(pooled_window_df)),
                train_interval_count=int(len(interval_dataset_df)),
                test_interval_count=int(len(pooled_interval_df)),
                window_metrics=compute_metric_bundle(pooled_window_df),
                interval_metrics=compute_metric_bundle(pooled_interval_df),
            )
        )

    results_df = pd.DataFrame(results_rows).sort_values(
        ["evaluation_scope", "joint_interval_exact", "movement_interval_macro_f1", "count_interval_macro_f1"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    interval_dataset_path = output_dir / "lrj_interval_dataset.csv"
    window_dataset_path = output_dir / "lrj_window_features.csv"
    results_path = output_dir / "lrj_benchmark_results.csv"
    window_predictions_path = output_dir / "lrj_window_predictions.csv"
    interval_predictions_path = output_dir / "lrj_interval_predictions.csv"
    summary_md_path = output_dir / "lrj_benchmark_summary.md"
    summary_json_path = output_dir / "lrj_benchmark_summary.json"

    interval_dataset_df.to_csv(interval_dataset_path, index=False)
    window_dataset_df.to_csv(window_dataset_path, index=False)
    results_df.to_csv(results_path, index=False)
    window_predictions_df.to_csv(window_predictions_path, index=False)
    interval_predictions_df.to_csv(interval_predictions_path, index=False)
    build_summary_markdown(
        output_path=summary_md_path,
        session_records=session_records,
        selected_channels=selected_channels,
        interval_dataset_df=interval_dataset_df,
        window_dataset_df=window_dataset_df,
        results_df=results_df,
    )

    write_json(
        summary_json_path,
        {
            "session_filenames": [record["filename"] for record in session_records],
            "selected_channels": selected_channels,
            "excluded_channels": excluded_channels,
            "feature_columns": feature_columns,
            "feature_mode": DEFAULT_LRJ_FEATURE_MODE,
            "with_asymmetry": DEFAULT_LRJ_WITH_ASYMMETRY,
            "window_sec": float(args.window_sec),
            "overlap": float(args.overlap),
            "results_csv": str(results_path.resolve()),
            "interval_dataset_csv": str(interval_dataset_path.resolve()),
            "window_dataset_csv": str(window_dataset_path.resolve()),
            "window_predictions_csv": str(window_predictions_path.resolve()),
            "interval_predictions_csv": str(interval_predictions_path.resolve()),
            "summary_md": str(summary_md_path.resolve()),
        },
    )

    print(f"\nShared selected channels: {', '.join(selected_channels)}")
    print(f"Wrote {interval_dataset_path}")
    print(f"Wrote {window_dataset_path}")
    print(f"Wrote {results_path}")
    print(f"Wrote {summary_md_path}")


if __name__ == "__main__":
    main()
