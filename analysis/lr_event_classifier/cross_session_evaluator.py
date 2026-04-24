from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def _slug(name: str) -> str:
    return name.replace(".csv", "").replace("(", "").replace(")", "").replace(",", "").replace(" ", "_")


def evaluate_models(
    event_matrix_df: pd.DataFrame,
    feature_columns: list[str],
    model_bank: dict[str, Any],
    labels: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    """
    Executes a Leave-One-Session-Out (LOSO) cross-validation scheme over a fully materialized
    event matrix, generating standardized precision matrices and scorecard rows.
    
    Args:
        event_matrix_df: DataFrame containing an event matrix (must include 'file' and 'label' columns)
        feature_columns: Feature keys to use as X
        model_bank: Instantiated sklearn-compatible models
        labels: Class labels
        output_dir: Guaranteed destination for model predictions and precision CSVs
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    unique_files = sorted(event_matrix_df["file"].unique().tolist())
    fold_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    
    for test_file in unique_files:
        train_df = event_matrix_df[event_matrix_df["file"] != test_file].reset_index(drop=True)
        test_df = event_matrix_df[event_matrix_df["file"] == test_file].reset_index(drop=True)
        
        if len(train_df) == 0:
            continue
            
        X_train = train_df[feature_columns].to_numpy(dtype=float)
        y_train = train_df["label"].to_numpy()
        X_test = test_df[feature_columns].to_numpy(dtype=float)
        y_test = test_df["label"].to_numpy()
        
        rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []
        train_files = sorted({str(value) for value in train_df["file"].tolist()})
        train_file_str = train_files[0] if len(train_files) == 1 else ", ".join(train_files)
        train_slug = _slug(train_files[0]) if len(train_files) == 1 else f"{len(train_files)}_train_sessions"
        test_slug = _slug(test_file)
        
        for model_name, model in model_bank.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            cm_df = pd.DataFrame(
                cm,
                index=[f"true_{label}" for label in labels],
                columns=[f"pred_{label}" for label in labels],
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
        fold_frames.append(pd.DataFrame(rows))
        prediction_frames.append(pd.DataFrame(prediction_rows))
        
    classifier_results_df = pd.concat(fold_frames, ignore_index=True)
    prediction_df = pd.concat(prediction_frames, ignore_index=True)

    pooled_rows: list[dict[str, Any]] = []
    for model_name, model_predictions in prediction_df.groupby("model_name"):
        y_true = model_predictions["true_label"].to_numpy()
        y_pred = model_predictions["pred_label"].to_numpy()
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{label}" for label in labels],
            columns=[f"pred_{label}" for label in labels],
        )
        confusion_path = output_dir / f"confusion_pooled_{model_name}.csv"
        cm_df.to_csv(confusion_path)
        pooled_rows.append(
            {
                "model_name": model_name,
                "train_file": "__pooled_cross_session__",
                "test_file": "__pooled_cross_session__",
                "train_samples": int(len(event_matrix_df)),
                "test_samples": int(len(model_predictions)),
                "train_class_counts": json.dumps(event_matrix_df["label"].value_counts().sort_index().to_dict()),
                "test_class_counts": json.dumps(model_predictions["true_label"].value_counts().sort_index().to_dict()),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
                "confusion_matrix_csv": str(confusion_path),
            }
        )
    pooled_df = pd.DataFrame(pooled_rows)
    classifier_results_df = pd.concat([classifier_results_df, pooled_df], ignore_index=True)

    classifier_results_path = output_dir / "classifier_results.csv"
    prediction_path = output_dir / "event_predictions.csv"
    classifier_results_df.to_csv(classifier_results_path, index=False)
    prediction_df.to_csv(prediction_path, index=False)
    
    best_row = pooled_df.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]
    best_confusion = pd.read_csv(best_row["confusion_matrix_csv"], index_col=0)
    
    summary = {
        "best_model": {
            "name": str(best_row["model_name"]),
            "accuracy": float(best_row["accuracy"]),
            "macro_f1": float(best_row["macro_f1"]),
            "confusion_matrix_csv": str(best_row["confusion_matrix_csv"]),
            "confusion_matrix": best_confusion.to_dict(),
        },
        "pooled_metrics": pooled_df[["model_name", "accuracy", "macro_f1"]].to_dict(orient="records"),
    }
    
    return {
        "classifier_results_df": classifier_results_df,
        "prediction_df": prediction_df,
        "summary": summary,
    }


def evaluate_tensor_models(
    X_tensor: np.ndarray,
    metadata_df: pd.DataFrame,
    model_bank: dict[str, Any],
    labels: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    """
    Executes a Leave-One-Session-Out (LOSO) cross-validation scheme over a 3D tensor
    of epochs/trials, generating standardized precision matrices and scorecard rows.
    
    Args:
        X_tensor: np.ndarray of shape (n_trials, n_channels, n_times)
        metadata_df: DataFrame matching n_trials with 'file' and 'label' columns
        model_bank: Instantiated sklearn-compatible pipelines
        labels: Class labels
        output_dir: Guaranteed destination for model predictions and precision CSVs
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    unique_files = sorted(metadata_df["file"].unique().tolist())
    fold_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    
    for test_file in unique_files:
        train_mask = (metadata_df["file"] != test_file).to_numpy()
        test_mask = (metadata_df["file"] == test_file).to_numpy()
        
        train_df = metadata_df[train_mask].reset_index(drop=True)
        test_df = metadata_df[test_mask].reset_index(drop=True)
        
        if len(train_df) == 0:
            continue
            
        X_train = X_tensor[train_mask]
        y_train = train_df["label"].to_numpy()
        X_test = X_tensor[test_mask]
        y_test = test_df["label"].to_numpy()
        
        rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []
        train_files = sorted({str(value) for value in train_df["file"].tolist()})
        train_file_str = train_files[0] if len(train_files) == 1 else ", ".join(train_files)
        train_slug = _slug(train_files[0]) if len(train_files) == 1 else f"{len(train_files)}_train_sessions"
        test_slug = _slug(test_file)
        
        for model_name, model in model_bank.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            cm_df = pd.DataFrame(
                cm,
                index=[f"true_{label}" for label in labels],
                columns=[f"pred_{label}" for label in labels],
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
        fold_frames.append(pd.DataFrame(rows))
        prediction_frames.append(pd.DataFrame(prediction_rows))
        
    classifier_results_df = pd.concat(fold_frames, ignore_index=True)
    prediction_df = pd.concat(prediction_frames, ignore_index=True)

    pooled_rows: list[dict[str, Any]] = []
    for model_name, model_predictions in prediction_df.groupby("model_name"):
        y_true = model_predictions["true_label"].to_numpy()
        y_pred = model_predictions["pred_label"].to_numpy()
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{label}" for label in labels],
            columns=[f"pred_{label}" for label in labels],
        )
        confusion_path = output_dir / f"confusion_pooled_{model_name}.csv"
        cm_df.to_csv(confusion_path)
        pooled_rows.append(
            {
                "model_name": model_name,
                "train_file": "__pooled_cross_session__",
                "test_file": "__pooled_cross_session__",
                "train_samples": int(len(metadata_df)),
                "test_samples": int(len(model_predictions)),
                "train_class_counts": json.dumps(metadata_df["label"].value_counts().sort_index().to_dict()),
                "test_class_counts": json.dumps(model_predictions["true_label"].value_counts().sort_index().to_dict()),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
                "confusion_matrix_csv": str(confusion_path),
            }
        )
    pooled_df = pd.DataFrame(pooled_rows)
    classifier_results_df = pd.concat([classifier_results_df, pooled_df], ignore_index=True)

    classifier_results_path = output_dir / "classifier_results.csv"
    prediction_path = output_dir / "event_predictions.csv"
    classifier_results_df.to_csv(classifier_results_path, index=False)
    prediction_df.to_csv(prediction_path, index=False)
    
    best_row = pooled_df.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]
    best_confusion = pd.read_csv(best_row["confusion_matrix_csv"], index_col=0)
    
    summary = {
        "best_model": {
            "name": str(best_row["model_name"]),
            "accuracy": float(best_row["accuracy"]),
            "macro_f1": float(best_row["macro_f1"]),
            "confusion_matrix_csv": str(best_row["confusion_matrix_csv"]),
            "confusion_matrix": best_confusion.to_dict(),
        },
        "pooled_metrics": pooled_df[["model_name", "accuracy", "macro_f1"]].to_dict(orient="records"),
    }
    
    return {
        "classifier_results_df": classifier_results_df,
        "prediction_df": prediction_df,
        "summary": summary,
    }
