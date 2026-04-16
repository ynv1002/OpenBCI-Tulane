from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from config import ExperimentConfig


LABEL_MAP = {
    "norm": "norm",
    "jaws": "Jaws",
    "jleft": "Jleft",
    "jright": "Jright",
    "rest": "norm",
    "left": "Jleft",
    "right": "Jright",
}


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_label(value: object) -> str:
    if pd.isna(value):
        return "missing"
    raw = str(value).strip()
    if not raw or raw.lower() == "nan":
        return "missing"
    return LABEL_MAP.get(raw.lower(), raw)


def load_emg_csv(csv_path: Path, config: ExperimentConfig) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if config.label_column not in df.columns:
        raise ValueError(
            f"Expected label column '{config.label_column}' in {csv_path}, found {list(df.columns)}"
        )

    missing_channels = [c for c in config.tracked_channels if c not in df.columns]
    if missing_channels:
        raise ValueError(f"Missing tracked channels: {missing_channels}")

    out = df.copy()
    out["label_original"] = out[config.label_column]
    out["label_clean"] = out[config.label_column].map(normalize_label)
    out["sample_index"] = np.arange(len(out), dtype=int)
    out["time_sec"] = out["sample_index"] / float(config.fs_hz)
    return out


def _channel_offset(signal: pd.Series, strategy: str) -> float:
    if strategy == "median":
        return float(signal.median())
    if strategy == "mean":
        return float(signal.mean())
    raise ValueError(f"Unsupported centering strategy: {strategy}")


def infer_scale_and_transform(
    df: pd.DataFrame,
    config: ExperimentConfig,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    tracked = list(config.tracked_channels)
    raw = df[tracked].astype(float)

    per_channel_stats = {}
    raw_abs_medians = []
    raw_abs_maxima = []
    for channel in tracked:
        series = raw[channel]
        per_channel_stats[channel] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "median": float(series.median()),
        }
        raw_abs_medians.append(abs(float(series.median())))
        raw_abs_maxima.append(float(series.abs().max()))

    raw_abs_median = float(np.median(raw_abs_medians))
    raw_abs_max = float(np.max(raw_abs_maxima))
    detected_mode = "raw_uv"
    detection_reasons: List[str] = []

    if config.scaling.mode == "auto":
        if raw_abs_median > config.scaling.counts_abs_median_trigger:
            detected_mode = "openbci_counts_centered"
            detection_reasons.append(
                f"median absolute raw channel level {raw_abs_median:.2f} exceeds "
                f"{config.scaling.counts_abs_median_trigger:.2f}"
            )
        if raw_abs_max > config.scaling.counts_abs_max_trigger:
            detected_mode = "openbci_counts_centered"
            detection_reasons.append(
                f"max absolute raw channel value {raw_abs_max:.2f} exceeds "
                f"{config.scaling.counts_abs_max_trigger:.2f}"
            )
    else:
        detected_mode = config.scaling.mode
        detection_reasons.append(f"scale mode forced to '{config.scaling.mode}'")

    transformed = raw.copy()
    channel_offsets: Dict[str, float] = {}
    scale_factor = 1.0

    if detected_mode == "openbci_counts_centered":
        scale_factor = config.scaling.openbci_count_to_uv
        for channel in tracked:
            offset = _channel_offset(raw[channel], config.scaling.center_strategy)
            channel_offsets[channel] = offset
            transformed[channel] = (raw[channel] - offset) * scale_factor
    elif detected_mode == "raw_uv":
        for channel in tracked:
            channel_offsets[channel] = 0.0
    else:
        raise ValueError(f"Unsupported scale mode: {detected_mode}")

    transformed_stats = {}
    for channel in tracked:
        series = transformed[channel]
        transformed_stats[channel] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "median": float(series.median()),
            "abs_p95": float(series.abs().quantile(0.95)),
            "abs_p99": float(series.abs().quantile(0.99)),
        }

    report = {
        "mode_requested": config.scaling.mode,
        "mode_applied": detected_mode,
        "detection_reasons": detection_reasons,
        "scale_factor_to_uv": scale_factor,
        "centering_strategy": config.scaling.center_strategy if detected_mode != "raw_uv" else "none",
        "channel_offsets": channel_offsets,
        "raw_channel_stats": per_channel_stats,
        "transformed_channel_stats": transformed_stats,
        "openbci_defaults_uv_limit": config.openbci.uv_limit,
        "data_scale_comment": build_scale_comment(detected_mode, transformed_stats, config),
    }

    out = df.copy()
    for channel in tracked:
        out[f"{channel}_signal_uv"] = transformed[channel]
    return out, report


def build_scale_comment(
    detected_mode: str,
    transformed_stats: Dict[str, Dict[str, float]],
    config: ExperimentConfig,
) -> str:
    p95_values = [stats["abs_p95"] for stats in transformed_stats.values()]
    p99_values = [stats["abs_p99"] for stats in transformed_stats.values()]
    median_p95 = float(np.median(p95_values))
    median_p99 = float(np.median(p99_values))
    uv_limit = config.openbci.uv_limit

    prefix = (
        "Tracked channels were centered and converted from assumed OpenBCI counts to microvolts"
        if detected_mode == "openbci_counts_centered"
        else "Tracked channels were treated as already being in microvolts"
    )
    compatibility = (
        "broadly compatible with the OpenBCI defaults"
        if median_p99 <= uv_limit * 1.25
        else "still somewhat larger than the OpenBCI defaults"
    )
    return (
        f"{prefix}; median abs p95={median_p95:.2f} uV and median abs p99={median_p99:.2f} uV, "
        f"which is {compatibility} (uvLimit={uv_limit:.2f} uV)."
    )


def choose_plot_window(
    df: pd.DataFrame,
    config: ExperimentConfig,
) -> Tuple[int, int]:
    window_samples = max(1, int(round(config.plots.inspection_window_seconds * config.fs_hz)))
    labels = df["label_clean"].to_numpy()
    active_indices = np.flatnonzero(np.isin(labels, list(config.active_labels)))
    center_index = int(active_indices[0]) if len(active_indices) else 0
    start = max(0, center_index - window_samples // 4)
    end = min(len(df), start + window_samples)
    return start, end


def compute_label_spans(labels: Sequence[str]) -> List[Tuple[int, int, str]]:
    if not labels:
        return []
    spans: List[Tuple[int, int, str]] = []
    start = 0
    current = labels[0]
    for idx in range(1, len(labels)):
        if labels[idx] != current:
            spans.append((start, idx - 1, current))
            start = idx
            current = labels[idx]
    spans.append((start, len(labels) - 1, current))
    return spans


def dataframe_to_markdown(df: pd.DataFrame, include_index: bool = True) -> str:
    render = df.copy()
    if not include_index:
        render = render.reset_index(drop=True)

    if include_index:
        headers = [render.index.name or "index"] + render.columns.tolist()
        rows = [[idx] + row.tolist() for idx, row in render.iterrows()]
    else:
        headers = render.columns.tolist()
        rows = render.values.tolist()

    table = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        table.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(table)


def to_serializable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    return value


def write_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2, sort_keys=True)

