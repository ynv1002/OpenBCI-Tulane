from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.lr_event_validation.review_yaniv_lr import DEFAULT_INPUT_DIR as LR_INPUT_DIR
from analysis.lr_event_validation.review_yaniv_lr import run_validation
from analysis.utils import (
    compute_channel_quality,
    dataframe_to_markdown,
    ensure_output_dir,
    load_openbci_csv,
    preprocess_session_signals,
    write_json,
    write_markdown,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_VALIDATION_DIR = REPO_ROOT / "analysis" / "lr_event_validation" / "outputs"
DEFAULT_FS_HZ = 250.0
EVENT_WINDOW_SEC = 0.50
LEFT = "LEFT"
RIGHT = "RIGHT"
LABELS = [LEFT, RIGHT]
HIGH_TRUST_FILES = [
    "LR-2-27-26-(01).csv",
    "LR-3-15-26-(04).csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a first event-level LEFT vs RIGHT classifier from Yaniv high-trust EEG_LR runs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for event-level classifier outputs.",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=DEFAULT_VALIDATION_DIR,
        help="Directory containing Yaniv LR validation outputs.",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=DEFAULT_FS_HZ,
        help="Authoritative sample rate for event timing and feature windows.",
    )
    return parser.parse_args()


def _fail(message: str) -> None:
    raise SystemExit(message)


def _target_count_channels(channel_quality_df: pd.DataFrame) -> list[str]:
    ordered = channel_quality_df.copy()
    ordered["channel_index"] = ordered["channel"].str.extract(r"(\d+)").astype(int)
    usable = ordered[ordered["status"] != "unsafe"].sort_values("channel_index")
    channels = usable["channel"].astype(str).tolist()
    if len(channels) < 2:
        _fail("Fewer than two target channels survived the unsafe-channel screen.")
    return channels


def _slug(name: str) -> str:
    return name.replace(".csv", "").replace("(", "").replace(")", "").replace(",", "").replace(" ", "_")


def _ensure_validation_outputs(validation_dir: Path, fs_hz: float) -> None:
    missing = []
    for filename in HIGH_TRUST_FILES:
        stem = Path(filename).stem
        if not (validation_dir / stem / "event_table.csv").exists():
            missing.append(filename)
    if missing:
        run_validation(
            input_dir=LR_INPUT_DIR,
            output_dir=validation_dir,
            fs_hz=fs_hz,
            make_plot=False,
        )


def _load_validation_bundle(validation_dir: Path, filename: str) -> dict[str, Any]:
    stem = Path(filename).stem
    file_dir = validation_dir / stem
    event_table_path = file_dir / "event_table.csv"
    block_summary_path = file_dir / "block_summary.csv"
    run_notes_path = file_dir / "run_notes.json"
    if not event_table_path.exists() or not block_summary_path.exists() or not run_notes_path.exists():
        _fail(f"Missing validation outputs for {filename} in {file_dir}")

    event_df = pd.read_csv(event_table_path)
    block_df = pd.read_csv(block_summary_path)
    run_notes = json.loads(run_notes_path.read_text(encoding="utf-8"))
    kept_df = event_df[
        (event_df["kept_for_count"] == True)
        & (event_df["inside_marker_block"] == "yes")
        & (event_df["assigned_side"].isin(LABELS))
    ].copy()
    kept_df["file"] = filename
    block_df["file"] = filename
    return {
        "filename": filename,
        "file_dir": file_dir,
        "event_df": kept_df.reset_index(drop=True),
        "block_df": block_df.reset_index(drop=True),
        "run_notes": run_notes,
    }


def _block_lookup(block_df: pd.DataFrame) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    for row in block_df.to_dict(orient="records"):
        lookup[int(row["block_id"])] = row
    return lookup


def _compute_common_channels(file_paths: list[Path]) -> list[str]:
    channel_sets: list[set[str]] = []
    for path in file_paths:
        raw_df = load_openbci_csv(path)
        quality_df = compute_channel_quality(raw_df)
        channel_sets.append(set(_target_count_channels(quality_df)))
    common = sorted(set.intersection(*channel_sets), key=lambda name: int(name.split("_")[1]))
    if len(common) < 2:
        _fail("Fewer than two common non-unsafe channels survived across the high-trust sessions.")
    return common


def _window_bounds(
    peak_sample: int,
    block_start_sample: int,
    block_end_sample: int,
    window_samples: int,
) -> tuple[int, int]:
    half_left = window_samples // 2
    half_right = window_samples - half_left - 1
    start = peak_sample - half_left
    end = peak_sample + half_right

    if start < block_start_sample:
        shift = block_start_sample - start
        start += shift
        end += shift
    if end > block_end_sample:
        shift = end - block_end_sample
        start -= shift
        end -= shift

    start = max(block_start_sample, start)
    end = min(block_end_sample, end)

    current_len = end - start + 1
    if current_len < window_samples:
        deficit = window_samples - current_len
        grow_left = min(deficit // 2 + deficit % 2, start - block_start_sample)
        grow_right = min(deficit // 2, block_end_sample - end)
        start -= grow_left
        end += grow_right
        remaining = window_samples - (end - start + 1)
        if remaining > 0:
            extra_left = min(remaining, start - block_start_sample)
            start -= extra_left
            remaining -= extra_left
        if remaining > 0:
            extra_right = min(remaining, block_end_sample - end)
            end += extra_right

    return int(start), int(end)


def _per_channel_features(values: np.ndarray) -> dict[str, float]:
    return {
        "rms": float(np.sqrt(np.mean(values ** 2))),
        "mav": float(np.mean(np.abs(values))),
        "variance": float(np.var(values)),
        "peak_abs": float(np.max(np.abs(values))),
        "waveform_length": float(np.sum(np.abs(np.diff(values)))) if len(values) > 1 else 0.0,
    }


def _aggregate_rms(window: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(window ** 2, axis=1))


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
        block_lookup = _block_lookup(bundle["block_df"])

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
            features = _event_features(window, common_channels)

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
            row.update(features)
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


def _model_bank() -> dict[str, Any]:
    return {
        "LDA": Pipeline([("scaler", StandardScaler()), ("model", LinearDiscriminantAnalysis())]),
        "LogisticRegression": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=3000, class_weight="balanced"))]
        ),
    }


def _fit_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    X_train = train_df[feature_columns].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy()
    X_test = test_df[feature_columns].to_numpy(dtype=float)
    y_test = test_df["label"].to_numpy()
    train_file = str(train_df["file"].iloc[0])
    test_file = str(test_df["file"].iloc[0])

    for model_name, model in _model_bank().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=LABELS)
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{label}" for label in LABELS],
            columns=[f"pred_{label}" for label in LABELS],
        )
        confusion_path = output_dir / f"confusion_{_slug(train_file)}_to_{_slug(test_file)}_{model_name}.csv"
        cm_df.to_csv(confusion_path)
        rows.append(
            {
                "model_name": model_name,
                "train_file": train_file,
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
                    "train_file": train_file,
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
    rows: list[dict[str, Any]] = []
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
        rows.append(
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
    return pd.DataFrame(rows)


def _build_summary(
    event_features_df: pd.DataFrame,
    results_df: pd.DataFrame,
    common_channels: list[str],
    feature_columns: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    pooled = results_df[results_df["train_file"] == "__pooled_cross_session__"].copy()
    best_row = pooled.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]
    best_confusion = pd.read_csv(best_row["confusion_matrix_csv"], index_col=0)
    best_model = str(best_row["model_name"])
    left_events = int((event_features_df["label"] == LEFT).sum())
    right_events = int((event_features_df["label"] == RIGHT).sum())

    summary = {
        "files_used": HIGH_TRUST_FILES,
        "excluded_files": [
            "LR-3-8-26-(02).csv",
            "LR-3-8-26-(03).csv",
        ],
        "selected_channels": common_channels,
        "event_window_sec": EVENT_WINDOW_SEC,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "sample_counts": {
            "total": int(len(event_features_df)),
            LEFT: left_events,
            RIGHT: right_events,
        },
        "train_test_strategy": "Two-fold leave-one-session-out cross-session evaluation across the two high-trust Yaniv runs.",
        "best_model": {
            "name": best_model,
            "accuracy": float(best_row["accuracy"]),
            "macro_f1": float(best_row["macro_f1"]),
            "confusion_matrix_csv": str(best_row["confusion_matrix_csv"]),
            "confusion_matrix": best_confusion.to_dict(),
        },
    }
    write_json(output_dir / "summary.json", summary)

    preview = pooled.loc[:, ["model_name", "accuracy", "macro_f1"]].copy()
    preview["accuracy"] = preview["accuracy"].map(lambda value: f"{float(value):.3f}")
    preview["macro_f1"] = preview["macro_f1"].map(lambda value: f"{float(value):.3f}")
    confusion_markdown = dataframe_to_markdown(best_confusion.reset_index().rename(columns={"index": "label"}))

    lines = [
        "# Event-Level LEFT vs RIGHT Baseline",
        "",
        "## Setup",
        f"- Files used: `{', '.join(HIGH_TRUST_FILES)}`",
        "- Excluded for this first pass: `LR-3-8-26-(02).csv`, `LR-3-8-26-(03).csv`",
        f"- Selected channels: `{', '.join(common_channels)}`",
        f"- Event window: `{EVENT_WINDOW_SEC:.2f} s` centered on each kept in-block event",
        "- Features: conservative time-domain event features only",
        "  RMS, MAV, variance, peak absolute amplitude, waveform length, aggregate RMS summaries, and simple channel-vs-mean asymmetry",
        "- Spectral mu/beta features: not included in this first baseline",
        "- CSP: intentionally out of scope for this first pass",
        "",
        "## Dataset",
        f"- Total labeled events: `{len(event_features_df)}`",
        f"- LEFT events: `{left_events}`",
        f"- RIGHT events: `{right_events}`",
        f"- Feature count: `{len(feature_columns)}`",
        "",
        "## Cross-Session Results",
        dataframe_to_markdown(preview),
        "",
        f"- Best pooled model: `{best_model}` with accuracy `{float(best_row['accuracy']):.3f}` and macro-F1 `{float(best_row['macro_f1']):.3f}`",
        "",
        "## Best Confusion Matrix",
        confusion_markdown,
        "",
        "## Interpretation",
    ]
    if float(best_row["macro_f1"]) >= 0.80:
        lines.append(
            "- Simple event-level LEFT vs RIGHT separation looks promising on the two high-trust Yaniv runs."
        )
    elif float(best_row["macro_f1"]) >= 0.65:
        lines.append(
            "- Event-level LEFT vs RIGHT separation is learnable enough to justify a next iteration, but it is not yet strong enough to treat as production-ready."
        )
    else:
        lines.append(
            "- Event-level LEFT vs RIGHT separation is only weakly learnable with this first conservative baseline."
        )
    lines.append(
        "- The next step should be refining event-centered features or calibration, not jumping to a larger model stack immediately."
    )
    write_markdown(output_dir / "summary.md", "\n".join(lines))
    return summary


def run_classifier(
    output_dir: Path,
    validation_dir: Path,
    fs_hz: float,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    validation_dir = ensure_output_dir(validation_dir)
    _ensure_validation_outputs(validation_dir, fs_hz)

    file_paths = [LR_INPUT_DIR / filename for filename in HIGH_TRUST_FILES]
    common_channels = _compute_common_channels(file_paths)
    validation_bundles = [_load_validation_bundle(validation_dir, filename) for filename in HIGH_TRUST_FILES]
    event_features_df, feature_columns = _build_event_feature_table(validation_bundles, common_channels, fs_hz)
    event_features_path = output_dir / "event_feature_table.csv"
    event_features_df.to_csv(event_features_path, index=False)

    fold_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for test_file in HIGH_TRUST_FILES:
        train_df = event_features_df[event_features_df["file"] != test_file].reset_index(drop=True)
        test_df = event_features_df[event_features_df["file"] == test_file].reset_index(drop=True)
        fold_results_df, fold_predictions_df = _fit_fold(
            train_df=train_df,
            test_df=test_df,
            feature_columns=feature_columns,
            output_dir=output_dir,
        )
        fold_frames.append(fold_results_df)
        prediction_frames.append(fold_predictions_df)

    classifier_results_df = pd.concat(fold_frames, ignore_index=True)
    prediction_df = pd.concat(prediction_frames, ignore_index=True)
    pooled_df = _pooled_results(prediction_df, event_features_df, output_dir)
    classifier_results_df = pd.concat([classifier_results_df, pooled_df], ignore_index=True)

    classifier_results_path = output_dir / "classifier_results.csv"
    prediction_path = output_dir / "event_predictions.csv"
    classifier_results_df.to_csv(classifier_results_path, index=False)
    prediction_df.to_csv(prediction_path, index=False)

    summary = _build_summary(
        event_features_df=event_features_df,
        results_df=classifier_results_df,
        common_channels=common_channels,
        feature_columns=feature_columns,
        output_dir=output_dir,
    )

    pooled_preview = classifier_results_df[classifier_results_df["train_file"] == "__pooled_cross_session__"][
        ["model_name", "accuracy", "macro_f1"]
    ].copy()
    pooled_preview["accuracy"] = pooled_preview["accuracy"].map(lambda value: f"{float(value):.3f}")
    pooled_preview["macro_f1"] = pooled_preview["macro_f1"].map(lambda value: f"{float(value):.3f}")
    print("Event-level LEFT vs RIGHT baseline")
    print(f"  Output directory: {output_dir}")
    print(f"  Files used: {', '.join(HIGH_TRUST_FILES)}")
    print(f"  Selected channels: {', '.join(common_channels)}")
    print(f"  Labeled events: {len(event_features_df)} ({LEFT}={summary['sample_counts'][LEFT]}, {RIGHT}={summary['sample_counts'][RIGHT]})")
    print(f"  Train/test strategy: {summary['train_test_strategy']}")
    print("")
    print("Pooled cross-session results")
    print(dataframe_to_markdown(pooled_preview))
    print("")
    print(
        f"Best baseline: {summary['best_model']['name']} "
        f"(accuracy={summary['best_model']['accuracy']:.3f}, macro_f1={summary['best_model']['macro_f1']:.3f})"
    )

    return {
        "event_features_path": event_features_path,
        "classifier_results_path": classifier_results_path,
        "prediction_path": prediction_path,
        "summary": summary,
    }


def main() -> None:
    args = parse_args()
    run_classifier(
        output_dir=args.output_dir,
        validation_dir=args.validation_dir,
        fs_hz=float(args.fs),
    )


if __name__ == "__main__":
    main()
