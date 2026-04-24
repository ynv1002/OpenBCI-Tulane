from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .event_matrix_builder import (
    DEFAULT_FS_HZ,
    DEFAULT_INCLUDE_FILES as HIGH_TRUST_FILES,
    DEFAULT_VALIDATION_DIR,
    EVENT_WINDOW_SEC,
    LABELS,
    LR_INPUT_DIR,
    LEFT,
    RIGHT,
    _compute_common_channels,
    _ensure_validation_outputs,
    _load_validation_bundle,
    _window_bounds,
    resolve_include_files as _resolve_include_files,
)
from ..utils import dataframe_to_markdown, ensure_output_dir, load_openbci_csv, preprocess_session_signals, write_json, write_markdown


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def _per_channel_features(values: np.ndarray) -> dict[str, float]:
    return {
        "rms": float(np.sqrt(np.mean(values**2))),
        "mav": float(np.mean(np.abs(values))),
        "variance": float(np.var(values)),
        "peak_abs": float(np.max(np.abs(values))),
        "waveform_length": float(np.sum(np.abs(np.diff(values)))) if len(values) > 1 else 0.0,
    }


def _aggregate_rms(window: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(window**2, axis=1))


def _event_features(window: np.ndarray, channel_columns: list[str]) -> dict[str, float]:
    features: dict[str, float] = {}
    channel_rms: dict[str, float] = {}
    channel_mav: dict[str, float] = {}
    for idx, channel in enumerate(channel_columns):
        channel_values = window[:, idx]
        channel_features = _per_channel_features(channel_values)
        channel_rms[channel] = channel_features["rms"]
        channel_mav[channel] = channel_features["mav"]
        for suffix, value in channel_features.items():
            features[f"{channel}_{suffix}"] = value

    mean_rms = float(np.mean(list(channel_rms.values())))
    mean_mav = float(np.mean(list(channel_mav.values())))
    for channel in channel_columns:
        features[f"{channel}_rms_asym"] = float(channel_rms[channel] - mean_rms)
        features[f"{channel}_mav_asym"] = float(channel_mav[channel] - mean_mav)

    aggregate = _aggregate_rms(window)
    features["aggregate_rms_mean"] = float(np.mean(aggregate))
    features["aggregate_rms_std"] = float(np.std(aggregate))
    features["aggregate_rms_max"] = float(np.max(aggregate))
    features["aggregate_rms_waveform_length"] = (
        float(np.sum(np.abs(np.diff(aggregate)))) if len(aggregate) > 1 else 0.0
    )
    return features


def _model_bank() -> dict[str, Any]:
    return {
        "LDA": Pipeline([("scaler", StandardScaler()), ("model", LinearDiscriminantAnalysis())]),
        "LogisticRegression": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=3000, class_weight="balanced"))]
        ),
    }


def _build_event_feature_table(
    validation_bundles: list[dict[str, Any]],
    common_channels: list[str],
    fs_hz: float,
) -> tuple[pd.DataFrame, list[str]]:
    window_samples = max(3, int(round(EVENT_WINDOW_SEC * fs_hz)))
    rows: list[dict[str, Any]] = []

    for bundle in validation_bundles:
        filename = bundle["filename"]
        csv_path = LR_INPUT_DIR / filename
        raw_df = load_openbci_csv(csv_path)
        processed = preprocess_session_signals(raw_df, "left_right", common_channels, fs_hz)
        block_lookup = {int(row["block_id"]): row for row in bundle["block_df"].to_dict(orient="records")}

        for event in bundle["event_df"].to_dict(orient="records"):
            block_id = int(event["assigned_block_id"])
            block = block_lookup[block_id]
            peak_sample = int(round(float(event["event_time_sec"]) * fs_hz))
            block_start_sample = int(round(float(block["start_time_sec"]) * fs_hz))
            block_end_sample = int(round(float(block["end_time_sec"]) * fs_hz))
            window_start, window_end = _window_bounds(
                peak_sample=peak_sample,
                block_start_sample=block_start_sample,
                block_end_sample=block_end_sample,
                window_samples=window_samples,
            )
            window = processed.iloc[window_start : window_end + 1].to_numpy(dtype=float)

            row = {
                "file": filename,
                "event_id": int(event["event_id"]),
                "event_time_sec": float(event["event_time_sec"]),
                "event_time_relative_sec": float(event["event_time_relative_sec"]),
                "block_id": block_id,
                "label": str(event["assigned_side"]),
                "block_start_time_sec": float(block["start_time_sec"]),
                "block_end_time_sec": float(block["end_time_sec"]),
                "peak_value": float(event["peak_value"]),
                "peak_prominence": float(event["peak_prominence"]),
                "window_start_time_sec": float(window_start / fs_hz),
                "window_end_time_sec": float(window_end / fs_hz),
                "window_sec": float((window_end - window_start + 1) / fs_hz),
            }
            row.update(_event_features(window, common_channels))
            rows.append(row)

    event_features_df = pd.DataFrame(rows).sort_values(["file", "event_time_sec"]).reset_index(drop=True)
    metadata_columns = {
        "file",
        "event_id",
        "event_time_sec",
        "event_time_relative_sec",
        "block_id",
        "label",
        "block_start_time_sec",
        "block_end_time_sec",
        "peak_value",
        "peak_prominence",
        "window_start_time_sec",
        "window_end_time_sec",
        "window_sec",
    }
    feature_columns = [column for column in event_features_df.columns if column not in metadata_columns]
    return event_features_df, feature_columns


def _fit_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train = train_df[feature_columns].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy()
    X_test = test_df[feature_columns].to_numpy(dtype=float)
    y_test = test_df["label"].to_numpy()

    train_files = sorted({str(value) for value in train_df["file"].tolist()})
    train_file_str = train_files[0] if len(train_files) == 1 else ", ".join(train_files)
    train_slug = train_files[0].replace(".csv", "").replace("(", "").replace(")", "").replace(",", "")
    test_file = str(test_df["file"].iloc[0])
    test_slug = test_file.replace(".csv", "").replace("(", "").replace(")", "").replace(",", "")

    rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for model_name, model in _model_bank().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=LABELS)
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{label}" for label in LABELS],
            columns=[f"pred_{label}" for label in LABELS],
        )
        confusion_path = output_dir / f"confusion_{train_slug}_to_{test_slug}_{model_name}.csv"
        cm_df.to_csv(confusion_path)
        rows.append(
            {
                "model_name": model_name,
                "train_file": train_file_str,
                "test_file": test_file,
                "train_samples": int(len(train_df)),
                "test_samples": int(len(test_df)),
                "train_class_counts": json.dumps(train_df["label"].value_counts().sort_index().to_dict()),
                "test_class_counts": json.dumps(test_df["label"].value_counts().sort_index().to_dict()),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
                "confusion_matrix_csv": str(confusion_path),
            }
        )
        for test_row, pred_label in zip(test_df.to_dict(orient="records"), y_pred):
            prediction_rows.append(
                {
                    "model_name": model_name,
                    "train_file": train_file_str,
                    "test_file": test_file,
                    "file": test_row["file"],
                    "event_id": int(test_row["event_id"]),
                    "event_time_sec": float(test_row["event_time_sec"]),
                    "block_id": int(test_row["block_id"]),
                    "true_label": str(test_row["label"]),
                    "pred_label": str(pred_label),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(prediction_rows)


def _pooled_results(
    prediction_df: pd.DataFrame,
    event_features_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    pooled_rows: list[dict[str, Any]] = []
    for model_name, model_predictions in prediction_df.groupby("model_name"):
        y_true = model_predictions["true_label"].to_numpy()
        y_pred = model_predictions["pred_label"].to_numpy()
        cm = confusion_matrix(y_true, y_pred, labels=LABELS)
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{label}" for label in LABELS],
            columns=[f"pred_{label}" for label in LABELS],
        )
        confusion_path = output_dir / f"confusion_pooled_{model_name}.csv"
        cm_df.to_csv(confusion_path)
        pooled_rows.append(
            {
                "model_name": model_name,
                "train_file": "__pooled_cross_session__",
                "test_file": "__pooled_cross_session__",
                "train_samples": int(len(event_features_df)),
                "test_samples": int(len(model_predictions)),
                "train_class_counts": json.dumps(event_features_df["label"].value_counts().sort_index().to_dict()),
                "test_class_counts": json.dumps(model_predictions["true_label"].value_counts().sort_index().to_dict()),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
                "confusion_matrix_csv": str(confusion_path),
            }
        )
    return pd.DataFrame(pooled_rows)


def run_classifier(
    output_dir: Path,
    validation_dir: Path = DEFAULT_VALIDATION_DIR,
    fs_hz: float = DEFAULT_FS_HZ,
    include_files: list[str] | None = None,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    resolved_files = _resolve_include_files(include_files)
    _ensure_validation_outputs(validation_dir, fs_hz, resolved_files)

    file_paths = [LR_INPUT_DIR / filename for filename in resolved_files]
    common_channels = _compute_common_channels(file_paths)
    validation_bundles = [_load_validation_bundle(validation_dir, filename) for filename in resolved_files]
    event_features_df, feature_columns = _build_event_feature_table(validation_bundles, common_channels, fs_hz)
    event_features_df.to_csv(output_dir / "event_feature_table.csv", index=False)

    fold_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for test_file in resolved_files:
        train_df = event_features_df[event_features_df["file"] != test_file].reset_index(drop=True)
        test_df = event_features_df[event_features_df["file"] == test_file].reset_index(drop=True)
        fold_results_df, fold_predictions_df = _fit_fold(train_df, test_df, feature_columns, output_dir)
        fold_frames.append(fold_results_df)
        prediction_frames.append(fold_predictions_df)

    results_df = pd.concat(fold_frames, ignore_index=True)
    prediction_df = pd.concat(prediction_frames, ignore_index=True)
    pooled_df = _pooled_results(prediction_df, event_features_df, output_dir)
    results_df = pd.concat([results_df, pooled_df], ignore_index=True)
    results_df.to_csv(output_dir / "classifier_results.csv", index=False)
    prediction_df.to_csv(output_dir / "event_predictions.csv", index=False)

    best_row = pooled_df.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]
    summary = {
        "files_used": resolved_files,
        "selected_channels": common_channels,
        "sample_counts": {
            "total": int(len(event_features_df)),
            LEFT: int((event_features_df["label"] == LEFT).sum()),
            RIGHT: int((event_features_df["label"] == RIGHT).sum()),
        },
        "best_model": {
            "name": str(best_row["model_name"]),
            "accuracy": float(best_row["accuracy"]),
            "macro_f1": float(best_row["macro_f1"]),
            "confusion_matrix_csv": str(best_row["confusion_matrix_csv"]),
        },
        "pooled_metrics": pooled_df[["model_name", "accuracy", "macro_f1"]].to_dict(orient="records"),
    }
    write_json(output_dir / "summary.json", summary)

    pooled_preview = pooled_df.loc[:, ["model_name", "accuracy", "macro_f1"]].copy()
    pooled_preview["accuracy"] = pooled_preview["accuracy"].map(lambda value: f"{float(value):.3f}")
    pooled_preview["macro_f1"] = pooled_preview["macro_f1"].map(lambda value: f"{float(value):.3f}")
    lines = [
        "# Event-Level LEFT vs RIGHT Baseline",
        "",
        f"- Files used: `{', '.join(resolved_files)}`",
        f"- Selected channels: `{', '.join(common_channels)}`",
        f"- Labeled events: `{len(event_features_df)}`",
        "",
        "## Pooled Results",
        dataframe_to_markdown(pooled_preview),
        "",
        "## Recommendation",
        (
            f"- Best pooled model: `{summary['best_model']['name']}` "
            f"with accuracy `{summary['best_model']['accuracy']:.3f}` "
            f"and macro-F1 `{summary['best_model']['macro_f1']:.3f}`."
        ),
    ]
    write_markdown(output_dir / "summary.md", "\n".join(lines))
    return {
        "results_df": results_df,
        "prediction_df": prediction_df,
        "summary": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the event-level LEFT vs RIGHT baseline on the trusted Yaniv LR sessions."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "summary",
        help="Directory for classifier outputs.",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=DEFAULT_VALIDATION_DIR,
        help="Directory containing LR validation outputs.",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=DEFAULT_FS_HZ,
        help="Authoritative sample rate.",
    )
    parser.add_argument(
        "--include-session",
        action="append",
        default=None,
        help="Restrict the evaluation to the listed LR session filename(s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_classifier(
        output_dir=args.output_dir,
        validation_dir=args.validation_dir,
        fs_hz=float(args.fs),
        include_files=args.include_session,
    )


if __name__ == "__main__":
    main()
