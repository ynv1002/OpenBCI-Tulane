from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from config import ExperimentConfig


FEATURE_SUFFIXES = (
    "left_aggregate",
    "right_aggregate",
    "x_raw",
    "x_smooth",
    "total_activation",
    "abs_direction",
)


def label_counts(df: pd.DataFrame) -> Dict[str, int]:
    counts = df["label_clean"].value_counts(dropna=False).sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def per_label_summary(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    rows = []
    for method in config.aggregation_methods:
        cols = [f"{method}_{suffix}" for suffix in FEATURE_SUFFIXES]
        grouped = df.groupby("label_clean")[cols].agg(["mean", "std"])
        for label, values in grouped.iterrows():
            row = {"method": method, "label": label}
            for column in cols:
                clean_name = column.replace(f"{method}_", "", 1)
                for stat in ("mean", "std"):
                    row[f"{clean_name}_{stat}"] = float(values[(column, stat)])
            rows.append(row)
    return pd.DataFrame(rows)


def cohens_d(left_values: np.ndarray, right_values: np.ndarray) -> float:
    left_values = np.asarray(left_values, dtype=float)
    right_values = np.asarray(right_values, dtype=float)
    left_var = left_values.var(ddof=1)
    right_var = right_values.var(ddof=1)
    pooled = np.sqrt(
        ((len(left_values) - 1) * left_var + (len(right_values) - 1) * right_var)
        / max(len(left_values) + len(right_values) - 2, 1)
    )
    if pooled == 0:
        return 0.0
    return float((right_values.mean() - left_values.mean()) / pooled)


def empirical_overlap(a: np.ndarray, b: np.ndarray, bins: int = 100) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    min_edge = float(min(a.min(), b.min()))
    max_edge = float(max(a.max(), b.max()))
    if min_edge == max_edge:
        return 1.0
    hist_a, edges = np.histogram(a, bins=bins, range=(min_edge, max_edge), density=True)
    hist_b, _ = np.histogram(b, bins=bins, range=(min_edge, max_edge), density=True)
    widths = np.diff(edges)
    return float(np.sum(np.minimum(hist_a, hist_b) * widths))


def left_right_separability(df: pd.DataFrame, method: str) -> Dict[str, float]:
    left = df.loc[df["label_clean"] == "Jleft", f"{method}_x_smooth"].to_numpy(dtype=float)
    right = df.loc[df["label_clean"] == "Jright", f"{method}_x_smooth"].to_numpy(dtype=float)
    y_true = np.concatenate([np.zeros(len(left), dtype=int), np.ones(len(right), dtype=int)])
    scores = np.concatenate([left, right])
    auc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "jleft_mean_x_smooth": float(left.mean()),
        "jright_mean_x_smooth": float(right.mean()),
        "cohens_d_x_smooth": cohens_d(left, right),
        "empirical_overlap_x_smooth": empirical_overlap(left, right),
        "roc_auc_x_smooth": auc,
    }


def scan_direction_thresholds(
    df: pd.DataFrame,
    method: str,
    thresholds: Iterable[float],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    subset = df.loc[df["label_clean"].isin(["Jleft", "Jright"])].copy()
    y_true = subset["label_clean"].to_numpy()
    x_values = subset[f"{method}_x_smooth"].to_numpy(dtype=float)

    rows = []
    best_score = -np.inf
    best_threshold = None
    best_predictions = None
    labels = ["Jleft", "Jright", "Undecided"]

    for threshold in thresholds:
        predictions = np.where(
            x_values > threshold,
            "Jright",
            np.where(x_values < -threshold, "Jleft", "Undecided"),
        )
        coverage = float(np.mean(predictions != "Undecided"))
        accuracy_all = float(np.mean(predictions == y_true))
        decided_mask = predictions != "Undecided"
        accuracy_decided = (
            float(np.mean(predictions[decided_mask] == y_true[decided_mask]))
            if decided_mask.any()
            else 0.0
        )
        macro_f1 = float(
            f1_score(y_true, predictions, labels=["Jleft", "Jright"], average="macro", zero_division=0)
        )
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": coverage,
                "accuracy_all": accuracy_all,
                "accuracy_decided": accuracy_decided,
                "macro_f1": macro_f1,
            }
        )
        if macro_f1 > best_score:
            best_score = macro_f1
            best_threshold = float(threshold)
            best_predictions = predictions

    scan_df = pd.DataFrame(rows)
    best_row = scan_df.loc[scan_df["threshold"] == best_threshold].iloc[0]
    confusion = pd.DataFrame(
        confusion_matrix(y_true, best_predictions, labels=labels),
        index=pd.Index(labels, name="actual"),
        columns=pd.Index(labels, name="predicted"),
    )
    return scan_df, best_row, confusion


def scan_activation_thresholds(
    df: pd.DataFrame,
    method: str,
    thresholds: Iterable[float],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    subset = df.loc[df["label_clean"].isin(["norm", "Jaws", "Jleft", "Jright"])].copy()
    y_true = np.where(subset["label_clean"] == "norm", "norm", "active")
    scores = subset[f"{method}_total_activation"].to_numpy(dtype=float)

    rows = []
    best_score = -np.inf
    best_threshold = None
    best_predictions = None
    labels = ["norm", "active"]

    for threshold in thresholds:
        predictions = np.where(scores > threshold, "active", "norm")
        macro_f1 = float(
            f1_score(y_true, predictions, labels=labels, average="macro", zero_division=0)
        )
        accuracy = float(np.mean(predictions == y_true))
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            }
        )
        if macro_f1 > best_score:
            best_score = macro_f1
            best_threshold = float(threshold)
            best_predictions = predictions

    scan_df = pd.DataFrame(rows)
    best_row = scan_df.loc[scan_df["threshold"] == best_threshold].iloc[0]
    confusion = pd.DataFrame(
        confusion_matrix(y_true, best_predictions, labels=labels),
        index=pd.Index(labels, name="actual"),
        columns=pd.Index(labels, name="predicted"),
    )
    return scan_df, best_row, confusion


def summarize_jaws(df: pd.DataFrame, method: str) -> Dict[str, float]:
    jaws = df.loc[df["label_clean"] == "Jaws"]
    norm = df.loc[df["label_clean"] == "norm"]
    lateral = df.loc[df["label_clean"].isin(["Jleft", "Jright"])]

    jaws_total = jaws[f"{method}_total_activation"].to_numpy(dtype=float)
    norm_total = norm[f"{method}_total_activation"].to_numpy(dtype=float)
    lateral_total = lateral[f"{method}_total_activation"].to_numpy(dtype=float)
    jaws_abs_dir = jaws[f"{method}_abs_direction"].to_numpy(dtype=float)
    lateral_abs_dir = lateral[f"{method}_abs_direction"].to_numpy(dtype=float)

    return {
        "jaws_mean_total_activation": float(jaws_total.mean()),
        "norm_mean_total_activation": float(norm_total.mean()),
        "lateral_mean_total_activation": float(lateral_total.mean()),
        "jaws_mean_abs_direction": float(jaws_abs_dir.mean()),
        "lateral_mean_abs_direction": float(lateral_abs_dir.mean()),
        "jaws_total_vs_norm_ratio": float(jaws_total.mean() / max(norm_total.mean(), 1e-9)),
        "jaws_abs_direction_vs_lateral_ratio": float(
            jaws_abs_dir.mean() / max(lateral_abs_dir.mean(), 1e-9)
        ),
    }
