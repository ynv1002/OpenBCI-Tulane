from __future__ import annotations

import argparse
import pickle
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
    from analysis.run_eeg_direction_base_up_calibrated import (
        CALIBRATION_SEC,
        compute_calibration_stats,
        normalize_with_calibration,
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
        compute_calibration_stats,
        normalize_with_calibration,
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


CLEAN_SESSION_FILENAMES = [
    "LR-2-27-26-(01).csv",
    "LR-3-15-26-(04).csv",
]
WINDOW_SEC = 1.0
OVERLAP = 0.5
FEATURE_MODE = "combined"
WITH_ASYMMETRY = True
LABELS = ["LEFT", "RIGHT"]
EXPECTED_SHARED_CHANNELS = [
    "Channel_1",
    "Channel_3",
    "Channel_4",
    "Channel_6",
    "Channel_7",
    "Channel_8",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain and evaluate a clean cross-session LEFT vs RIGHT EEG model on the two high-trust Yaniv LR sessions."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for clean cross-session direction outputs.",
    )
    return parser.parse_args()


def model_bank() -> Dict[str, object]:
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        ),
        "LDA": Pipeline([("scaler", StandardScaler()), ("model", LinearDiscriminantAnalysis())]),
        "LogisticRegression": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=3000, class_weight="balanced"))]
        ),
    }


def resolve_clean_audits() -> List[Dict[str, object]]:
    audits = [audit for audit in audit_all_sessions() if audit["family"] == "left_right"]
    selected = [audit for audit in audits if audit["filename"] in CLEAN_SESSION_FILENAMES]
    selected.sort(key=lambda row: CLEAN_SESSION_FILENAMES.index(row["filename"]))
    found = [audit["filename"] for audit in selected]
    missing = [filename for filename in CLEAN_SESSION_FILENAMES if filename not in found]
    if missing:
        raise ValueError(f"Missing clean LR session(s): {missing}")
    return selected


def resolve_shared_channels(clean_audits: Sequence[Dict[str, object]]) -> Tuple[List[str], Dict[str, str]]:
    selected_channels, excluded_channels = select_model_channels(clean_audits)
    if selected_channels != EXPECTED_SHARED_CHANNELS:
        raise ValueError(
            "Shared clean channel set changed unexpectedly. "
            f"Expected {EXPECTED_SHARED_CHANNELS}, got {selected_channels}."
        )
    return selected_channels, excluded_channels


def build_normalized_session_windows(
    audit: Dict[str, object],
    selected_channels: Sequence[str],
    split_role: str,
) -> Dict[str, object]:
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
    left_right_only = remaining.loc[remaining["label"].isin(LABELS)].copy().reset_index(drop=True)
    if left_right_only.empty:
        raise ValueError(f"No post-calibration LEFT/RIGHT windows remained for {audit['filename']}.")

    return {
        "frame": left_right_only,
        "feature_columns": feature_columns,
        "calibration_windows": calibration_windows,
    }


def fit_and_score(
    model: object,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> Dict[str, object]:
    X_train = train_df.loc[:, list(feature_columns)].to_numpy(dtype=float)
    X_test = test_df.loc[:, list(feature_columns)].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    confusion = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=LABELS),
        index=[f"true_{label}" for label in LABELS],
        columns=[f"pred_{label}" for label in LABELS],
    )
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "confusion_matrix": confusion,
        "y_true": list(y_test),
        "y_pred": list(y_pred),
        "model": model,
    }


def pooled_confusion_frame(y_true: Sequence[str], y_pred: Sequence[str]) -> pd.DataFrame:
    confusion = confusion_matrix(y_true, y_pred, labels=LABELS)
    return pd.DataFrame(
        confusion,
        index=[f"true_{label}" for label in LABELS],
        columns=[f"pred_{label}" for label in LABELS],
    )


def reuse_note(pooled_macro_f1: float) -> str:
    if pooled_macro_f1 >= 0.75:
        return "strong enough to reuse directly for the hybrid direction stage"
    if pooled_macro_f1 >= 0.65:
        return "promising enough to reuse as the first hybrid direction candidate, but validate in-pipeline"
    return "not strong enough to reuse directly; keep as a clean benchmark and retrain again after hybrid integration changes"


def save_artifact(
    output_path: Path,
    trained_model: object,
    selected_channels: Sequence[str],
    feature_columns: Sequence[str],
    winning_model_name: str,
) -> None:
    payload = {
        "artifact_type": "clean_left_right_window_model",
        "model_family": winning_model_name,
        "model": trained_model,
        "selected_channels": list(selected_channels),
        "feature_columns": list(feature_columns),
        "calibration_sec": CALIBRATION_SEC,
        "window_sec": WINDOW_SEC,
        "overlap": OVERLAP,
        "feature_mode": FEATURE_MODE,
        "with_asymmetry": WITH_ASYMMETRY,
        "source_sessions": list(CLEAN_SESSION_FILENAMES),
        "labels": list(LABELS),
    }
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle)


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    clean_audits = resolve_clean_audits()
    selected_channels, excluded_channels = resolve_shared_channels(clean_audits)
    session_bundles = {
        audit["filename"]: build_normalized_session_windows(audit, selected_channels, split_role="clean_cross_session")
        for audit in clean_audits
    }

    fold_pairs = [
        (CLEAN_SESSION_FILENAMES[0], CLEAN_SESSION_FILENAMES[1]),
        (CLEAN_SESSION_FILENAMES[1], CLEAN_SESSION_FILENAMES[0]),
    ]

    rows: List[Dict[str, object]] = []
    pooled_predictions: Dict[str, Dict[str, List[str]]] = {
        model_name: {"y_true": [], "y_pred": []} for model_name in model_bank()
    }

    for train_session, test_session in fold_pairs:
        train_bundle = session_bundles[train_session]
        test_bundle = session_bundles[test_session]
        if train_bundle["feature_columns"] != test_bundle["feature_columns"]:
            raise ValueError("Feature columns differ between clean sessions after applying the shared channel set.")

        for model_name, model in model_bank().items():
            metrics = fit_and_score(
                model=model,
                train_df=train_bundle["frame"],
                test_df=test_bundle["frame"],
                feature_columns=train_bundle["feature_columns"],
            )
            confusion_csv = output_dir / (
                f"confusion_eeg_direction_clean_cross_session_"
                f"{model_name}_{train_session.replace('.', '_').replace('(', '').replace(')', '')}"
                f"_to_{test_session.replace('.', '_').replace('(', '').replace(')', '')}.csv"
            )
            metrics["confusion_matrix"].to_csv(confusion_csv)

            pooled_predictions[model_name]["y_true"].extend(metrics["y_true"])
            pooled_predictions[model_name]["y_pred"].extend(metrics["y_pred"])

            rows.append(
                {
                    "model_name": model_name,
                    "train_session": train_session,
                    "test_session": test_session,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "selected_channels": ", ".join(selected_channels),
                    "feature_count": len(train_bundle["feature_columns"]),
                    "window_sec": WINDOW_SEC,
                    "overlap": OVERLAP,
                    "calibration_sec": CALIBRATION_SEC,
                    "feature_mode": FEATURE_MODE,
                    "with_asymmetry": WITH_ASYMMETRY,
                    "train_window_count": len(train_bundle["frame"]),
                    "test_window_count": len(test_bundle["frame"]),
                    "train_class_counts": train_bundle["frame"]["label"].value_counts().sort_index().to_dict(),
                    "test_class_counts": test_bundle["frame"]["label"].value_counts().sort_index().to_dict(),
                    "train_calibration_windows": train_bundle["calibration_windows"],
                    "test_calibration_windows": test_bundle["calibration_windows"],
                    "confusion_matrix_csv": str(confusion_csv),
                }
            )

    pooled_rows: List[Dict[str, object]] = []
    for model_name, prediction_bundle in pooled_predictions.items():
        pooled_confusion = pooled_confusion_frame(prediction_bundle["y_true"], prediction_bundle["y_pred"])
        pooled_accuracy = float(accuracy_score(prediction_bundle["y_true"], prediction_bundle["y_pred"]))
        pooled_macro_f1 = float(
            f1_score(prediction_bundle["y_true"], prediction_bundle["y_pred"], labels=LABELS, average="macro", zero_division=0)
        )
        confusion_csv = output_dir / f"confusion_eeg_direction_clean_cross_session_{model_name}_pooled.csv"
        pooled_confusion.to_csv(confusion_csv)
        pooled_rows.append(
            {
                "model_name": model_name,
                "train_session": "pooled_loso",
                "test_session": "pooled_loso",
                "accuracy": pooled_accuracy,
                "macro_f1": pooled_macro_f1,
                "selected_channels": ", ".join(selected_channels),
                "feature_count": len(session_bundles[CLEAN_SESSION_FILENAMES[0]]["feature_columns"]),
                "window_sec": WINDOW_SEC,
                "overlap": OVERLAP,
                "calibration_sec": CALIBRATION_SEC,
                "feature_mode": FEATURE_MODE,
                "with_asymmetry": WITH_ASYMMETRY,
                "train_window_count": sum(len(bundle["frame"]) for bundle in session_bundles.values()),
                "test_window_count": sum(len(bundle["frame"]) for bundle in session_bundles.values()),
                "train_class_counts": "pooled",
                "test_class_counts": "pooled",
                "train_calibration_windows": "pooled",
                "test_calibration_windows": "pooled",
                "confusion_matrix_csv": str(confusion_csv),
            }
        )

    results_df = pd.DataFrame(rows + pooled_rows).sort_values(
        ["test_session", "macro_f1", "accuracy", "model_name"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)

    pooled_df = pd.DataFrame(pooled_rows).sort_values(
        ["macro_f1", "accuracy", "model_name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    winning_model_name = str(pooled_df.iloc[0]["model_name"])

    pooled_training_frame = pd.concat(
        [session_bundles[filename]["frame"] for filename in CLEAN_SESSION_FILENAMES],
        ignore_index=True,
    )
    feature_columns = session_bundles[CLEAN_SESSION_FILENAMES[0]]["feature_columns"]
    winning_model = model_bank()[winning_model_name]
    winning_model.fit(
        pooled_training_frame.loc[:, feature_columns].to_numpy(dtype=float),
        pooled_training_frame["label"].to_numpy(),
    )

    results_csv = output_dir / "left_right_windowed_results.csv"
    summary_md = output_dir / "left_right_windowed_results.md"
    artifact_path = output_dir / "clean_left_right_window_model.pkl"

    results_df.to_csv(results_csv, index=False)
    save_artifact(artifact_path, winning_model, selected_channels, feature_columns, winning_model_name)

    per_fold_df = results_df[results_df["train_session"] != "pooled_loso"].copy()
    summary_lines = [
        "# Clean Cross-Session LEFT vs RIGHT EEG Retrain",
        "",
        f"- Sessions: `{', '.join(CLEAN_SESSION_FILENAMES)}`",
        f"- Shared selected channels: `{', '.join(selected_channels)}`",
        f"- Excluded channels: `{' | '.join(f'{key}: {value}' for key, value in excluded_channels.items())}`",
        f"- Windowing: `{WINDOW_SEC:.1f}` s with overlap `{OVERLAP:.1f}`",
        f"- Calibration: first `{CALIBRATION_SEC:.0f}` seconds per session using baseline/rest windows",
        f"- Features: `{FEATURE_MODE}` with asymmetry `{WITH_ASYMMETRY}`",
        "",
        "## Per-Fold Results",
        "",
        dataframe_to_markdown(
            per_fold_df.loc[
                :,
                [
                    "model_name",
                    "train_session",
                    "test_session",
                    "accuracy",
                    "macro_f1",
                    "feature_count",
                    "confusion_matrix_csv",
                ],
            ]
        ),
        "",
        "## Pooled LOSO Results",
        "",
        dataframe_to_markdown(
            pooled_df.loc[
                :,
                [
                    "model_name",
                    "accuracy",
                    "macro_f1",
                    "feature_count",
                    "confusion_matrix_csv",
                ],
            ]
        ),
        "",
        "## Recommendation",
        "",
        f"- Winning model: `{winning_model_name}`",
        f"- Winning pooled macro-F1: `{float(pooled_df.iloc[0]['macro_f1']):.3f}`",
        f"- Winning pooled accuracy: `{float(pooled_df.iloc[0]['accuracy']):.3f}`",
        f"- Hybrid reuse readout: {reuse_note(float(pooled_df.iloc[0]['macro_f1']))}",
        f"- Saved artifact: `{artifact_path}`",
    ]
    write_markdown(summary_md, "\n".join(summary_lines))

    print("Per-fold results")
    print(
        per_fold_df.loc[:, ["model_name", "train_session", "test_session", "accuracy", "macro_f1"]].to_string(
            index=False
        )
    )
    print("\nPooled LOSO results")
    print(pooled_df.loc[:, ["model_name", "accuracy", "macro_f1"]].to_string(index=False))
    print(f"\nWinning model: {winning_model_name}")
    print(f"Hybrid reuse readout: {reuse_note(float(pooled_df.iloc[0]['macro_f1']))}")
    print(f"\nWrote {results_csv}")
    print(f"Wrote {summary_md}")
    print(f"Wrote {artifact_path}")


if __name__ == "__main__":
    main()
