from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from config import AuditConfig


FEATURE_SUFFIXES = (
    "left_aggregate",
    "right_aggregate",
    "x_raw",
    "x_smooth",
    "total_activation",
    "abs_direction",
)


def _safe_skew(series: pd.Series) -> float:
    try:
        return float(series.skew())
    except Exception:
        return float("nan")


def compute_channel_diagnostics(
    raw_df: pd.DataFrame,
    mode_signals: Dict[str, pd.DataFrame],
    config: AuditConfig,
) -> pd.DataFrame:
    rows = []
    raw_signals = raw_df[list(config.all_channels)].astype(float)
    for mode, signal_df in mode_signals.items():
        for channel in config.all_channels:
            series = signal_df[channel].astype(float)
            q25, q75 = np.percentile(series, [25, 75])
            mad = float(np.median(np.abs(series - np.median(series))))
            first = series.iloc[: max(10, len(series) // 10)]
            last = series.iloc[-max(10, len(series) // 10) :]
            x = raw_df["time_sec"].to_numpy(dtype=float)
            trend_slope = float(np.polyfit(x, series.to_numpy(dtype=float), 1)[0]) if len(series) > 1 else 0.0
            raw_series = raw_signals[channel]
            fraction_near_rail = float(
                np.mean(np.abs(np.abs(raw_series) - config.raw_rail_nominal) < 500.0)
            )
            rows.append(
                {
                    "mode": mode,
                    "channel": channel,
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "iqr": float(q75 - q25),
                    "mad": mad,
                    "abs_p95": float(series.abs().quantile(0.95)),
                    "abs_p99": float(series.abs().quantile(0.99)),
                    "skew": _safe_skew(series),
                    "trend_slope_per_sec": trend_slope,
                    "median_first_decile": float(first.median()),
                    "median_last_decile": float(last.median()),
                    "median_drift": float(last.median() - first.median()),
                    "raw_fraction_near_nominal_rail": fraction_near_rail,
                    "flag_suspicious_saturation": bool(
                        raw_series.abs().max() >= config.suspicious_raw_rail
                        or fraction_near_rail > 0.01
                    ),
                }
            )

    diagnostics = pd.DataFrame(rows)
    for mode in diagnostics["mode"].unique():
        mask = diagnostics["mode"] == mode
        mode_std_median = float(diagnostics.loc[mask, "std"].median())
        diagnostics.loc[mask, "flag_near_constant"] = (
            diagnostics.loc[mask, "std"] < max(1e-6, mode_std_median * 0.05)
        )
        diagnostics.loc[mask, "flag_unusually_noisy"] = (
            diagnostics.loc[mask, "std"] > max(1e-6, mode_std_median * 2.5)
        )
        diagnostics.loc[mask, "flag_extreme_skew"] = diagnostics.loc[mask, "skew"].abs() > 2.0
        diagnostics.loc[mask, "flag_large_drift"] = (
            diagnostics.loc[mask, "median_drift"].abs()
            > np.maximum(diagnostics.loc[mask, "std"] * 0.5, 1e-6)
        )
    return diagnostics


def per_label_summary(processed_df: pd.DataFrame, config: AuditConfig) -> pd.DataFrame:
    subset = processed_df.loc[processed_df["label_clean"].isin(config.supported_labels)].copy()
    rows = []
    for method in config.aggregation_methods:
        cols = [f"{method}_{suffix}" for suffix in FEATURE_SUFFIXES]
        grouped = subset.groupby("label_clean")[cols].agg(["mean", "std"])
        for label, values in grouped.iterrows():
            row = {"mode": processed_df["mode"].iloc[0], "method": method, "label": label}
            for column in cols:
                clean_name = column.replace(f"{method}_", "", 1)
                for stat in ("mean", "std"):
                    row[f"{clean_name}_{stat}"] = float(values[(column, stat)])
            rows.append(row)
    return pd.DataFrame(rows)


def processor_behavior_summary(processed_df: pd.DataFrame, config: AuditConfig) -> Dict[str, float]:
    clip_fraction = []
    low_fraction = []
    high_fraction = []
    threshold_gap = []
    for channel in config.tracked_channels:
        avg_uv = processed_df[f"{channel}_average_uv"]
        output_norm = processed_df[f"{channel}_output_normalized"]
        gap = processed_df[f"{channel}_upper_threshold"] - processed_df[f"{channel}_lower_threshold"]
        clip_fraction.append(float(np.mean(avg_uv >= config.openbci.uv_limit - 1e-9)))
        low_fraction.append(float(np.mean(output_norm <= config.near_zero_normalized_threshold)))
        high_fraction.append(float(np.mean(output_norm >= config.near_one_normalized_threshold)))
        threshold_gap.append(float(gap.median()))

    mid_fraction = [1.0 - low - high for low, high in zip(low_fraction, high_fraction)]
    return {
        "average_uv_clip_fraction_mean": float(np.mean(clip_fraction)),
        "normalized_low_fraction_mean": float(np.mean(low_fraction)),
        "normalized_high_fraction_mean": float(np.mean(high_fraction)),
        "normalized_mid_fraction_mean": float(np.mean(mid_fraction)),
        "median_threshold_gap_uv": float(np.median(threshold_gap)),
    }


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


def left_right_separability(processed_df: pd.DataFrame, method: str) -> Dict[str, float]:
    left = processed_df.loc[processed_df["label_clean"] == "Jleft", f"{method}_x_smooth"].to_numpy(dtype=float)
    right = processed_df.loc[processed_df["label_clean"] == "Jright", f"{method}_x_smooth"].to_numpy(dtype=float)
    y_true = np.concatenate([np.zeros(len(left), dtype=int), np.ones(len(right), dtype=int)])
    scores = np.concatenate([left, right])
    auc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "jleft_mean_x_smooth": float(left.mean()),
        "jright_mean_x_smooth": float(right.mean()),
        "cohens_d_x_smooth": cohens_d(left, right),
        "empirical_overlap_x_smooth": empirical_overlap(left, right),
        "roc_auc_x_smooth": auc,
        "expected_direction_order": bool(float(right.mean()) > float(left.mean())),
    }


def scan_direction_thresholds(
    processed_df: pd.DataFrame,
    method: str,
    thresholds: Iterable[float],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    subset = processed_df.loc[processed_df["label_clean"].isin(["Jleft", "Jright"])].copy()
    y_true = subset["label_clean"].to_numpy()
    x_values = subset[f"{method}_x_smooth"].to_numpy(dtype=float)
    rows = []
    best_score = -np.inf
    best_predictions = None
    best_threshold = None
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
            best_predictions = predictions
            best_threshold = float(threshold)

    scan_df = pd.DataFrame(rows)
    best_row = scan_df.loc[scan_df["threshold"] == best_threshold].iloc[0]
    confusion = pd.DataFrame(
        confusion_matrix(y_true, best_predictions, labels=labels),
        index=pd.Index(labels, name="actual"),
        columns=pd.Index(labels, name="predicted"),
    )
    return scan_df, best_row, confusion


def scan_activation_thresholds(
    processed_df: pd.DataFrame,
    method: str,
    thresholds: Iterable[float],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    subset = processed_df.loc[processed_df["label_clean"].isin(["norm", "Jaws", "Jleft", "Jright"])].copy()
    y_true = np.where(subset["label_clean"] == "norm", "norm", "active")
    scores = subset[f"{method}_total_activation"].to_numpy(dtype=float)
    rows = []
    best_score = -np.inf
    best_predictions = None
    best_threshold = None
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
            best_predictions = predictions
            best_threshold = float(threshold)

    scan_df = pd.DataFrame(rows)
    best_row = scan_df.loc[scan_df["threshold"] == best_threshold].iloc[0]
    confusion = pd.DataFrame(
        confusion_matrix(y_true, best_predictions, labels=labels),
        index=pd.Index(labels, name="actual"),
        columns=pd.Index(labels, name="predicted"),
    )
    return scan_df, best_row, confusion


def summarize_jaws(processed_df: pd.DataFrame, method: str) -> Dict[str, float]:
    jaws = processed_df.loc[processed_df["label_clean"] == "Jaws"]
    norm = processed_df.loc[processed_df["label_clean"] == "norm"]
    lateral = processed_df.loc[processed_df["label_clean"].isin(["Jleft", "Jright"])]
    jaws_total = jaws[f"{method}_total_activation"].to_numpy(dtype=float)
    norm_total = norm[f"{method}_total_activation"].to_numpy(dtype=float)
    lateral_abs_dir = lateral[f"{method}_abs_direction"].to_numpy(dtype=float)
    jaws_abs_dir = jaws[f"{method}_abs_direction"].to_numpy(dtype=float)
    return {
        "jaws_mean_total_activation": float(jaws_total.mean()),
        "norm_mean_total_activation": float(norm_total.mean()),
        "jaws_total_vs_norm_ratio": float(jaws_total.mean() / max(norm_total.mean(), 1e-9)),
        "jaws_mean_abs_direction": float(jaws_abs_dir.mean()),
        "lateral_mean_abs_direction": float(lateral_abs_dir.mean()),
        "jaws_abs_direction_vs_lateral_ratio": float(
            jaws_abs_dir.mean() / max(lateral_abs_dir.mean(), 1e-3)
        ),
        "jaws_center_score": float(
            (jaws_total.mean() / max(norm_total.mean(), 1e-9))
            - (jaws_abs_dir.mean() / max(lateral_abs_dir.mean(), 1e-3))
        ),
    }


def build_mode_comparison(
    processed_by_mode: Dict[str, pd.DataFrame],
    config: AuditConfig,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    rows = []
    payload: Dict[str, object] = {}
    best_direction = None
    best_sensible = None
    best_jaws = None

    for mode, processed_df in processed_by_mode.items():
        mode_payload = {
            "processor_behavior": processor_behavior_summary(processed_df, config),
            "methods": {},
        }
        for method in config.aggregation_methods:
            separability = left_right_separability(processed_df, method)
            direction_scan, direction_best, direction_confusion = scan_direction_thresholds(
                processed_df, method, config.direction_thresholds
            )
            activation_scan, activation_best, activation_confusion = scan_activation_thresholds(
                processed_df, method, config.activation_thresholds
            )
            jaws = summarize_jaws(processed_df, method)
            row = {
                "mode": mode,
                "method": method,
                **mode_payload["processor_behavior"],
                **separability,
                "direction_threshold": float(direction_best["threshold"]),
                "direction_macro_f1": float(direction_best["macro_f1"]),
                "direction_accuracy_all": float(direction_best["accuracy_all"]),
                "direction_coverage": float(direction_best["coverage"]),
                "activation_threshold": float(activation_best["threshold"]),
                "activation_macro_f1": float(activation_best["macro_f1"]),
                "activation_accuracy": float(activation_best["accuracy"]),
                **jaws,
            }
            rows.append(row)
            mode_payload["methods"][method] = {
                "separability": separability,
                "direction_scan": direction_scan,
                "direction_best": direction_best.to_dict(),
                "direction_confusion": direction_confusion,
                "activation_scan": activation_scan,
                "activation_best": activation_best.to_dict(),
                "activation_confusion": activation_confusion,
                "jaws_summary": jaws,
            }
        payload[mode] = mode_payload

    comparison = pd.DataFrame(rows)
    best_direction = comparison.sort_values(
        ["direction_macro_f1", "roc_auc_x_smooth"], ascending=[False, False]
    ).iloc[0]
    best_sensible = comparison.sort_values(
        [
            "average_uv_clip_fraction_mean",
            "normalized_mid_fraction_mean",
            "direction_macro_f1",
        ],
        ascending=[True, False, False],
    ).iloc[0]
    jaws_candidates = comparison.loc[
        (comparison["average_uv_clip_fraction_mean"] < 0.5)
        & (comparison["normalized_mid_fraction_mean"] > 0.1)
    ]
    if jaws_candidates.empty:
        jaws_candidates = comparison
    best_jaws = jaws_candidates.sort_values(
        ["jaws_center_score", "jaws_total_vs_norm_ratio"], ascending=[False, False]
    ).iloc[0]

    summary = {
        "per_mode": payload,
        "best_direction_mode_method": best_direction.to_dict(),
        "best_processor_sensibility_mode_method": best_sensible.to_dict(),
        "best_jaws_center_mode_method": best_jaws.to_dict(),
    }
    return comparison, summary
