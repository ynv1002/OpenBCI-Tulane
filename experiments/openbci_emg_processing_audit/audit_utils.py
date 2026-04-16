from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from config import AuditConfig


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


def load_emg_csv(csv_path: Path, config: AuditConfig) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [channel for channel in config.all_channels if channel not in df.columns]
    if missing:
        raise ValueError(f"Missing expected channel columns: {missing}")
    if config.label_column not in df.columns:
        raise ValueError(f"Missing label column '{config.label_column}'")

    out = df.copy()
    out["label_original"] = out[config.label_column]
    out["label_clean"] = out[config.label_column].map(normalize_label)
    out["sample_index"] = np.arange(len(out), dtype=int)
    out["time_sec"] = out["sample_index"] / float(config.fs_hz)
    return out


def choose_plot_window(df: pd.DataFrame, config: AuditConfig) -> Tuple[int, int]:
    window_samples = max(1, int(round(config.plots.inspection_window_seconds * config.fs_hz)))
    labels = df["label_clean"].to_numpy()
    active_indices = np.flatnonzero(np.isin(labels, list(config.active_labels)))
    center_index = int(active_indices[0]) if len(active_indices) else 0
    start = max(0, center_index - window_samples // 4)
    end = min(len(df), start + window_samples)
    return start, end


def choose_sanity_window(df: pd.DataFrame, config: AuditConfig) -> Tuple[int, int]:
    labels = df["label_clean"].to_numpy()
    active_indices = np.flatnonzero(np.isin(labels, list(config.active_labels)))
    start = int(active_indices[0]) if len(active_indices) else 0
    end = min(len(df), start + config.sanity_window_samples)
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


def assess_raw_signal_scale(df: pd.DataFrame, config: AuditConfig) -> Dict[str, object]:
    raw = df[list(config.all_channels)].astype(float)
    per_channel = {}
    medians = []
    maxima = []
    rail_hits = []
    for channel in config.all_channels:
        series = raw[channel]
        median = float(series.median())
        max_abs = float(series.abs().max())
        near_nominal_rail = float(np.mean(np.abs(np.abs(series) - config.raw_rail_nominal) < 500.0))
        per_channel[channel] = {
            "median": median,
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "max_abs": max_abs,
            "fraction_near_nominal_rail": near_nominal_rail,
        }
        medians.append(abs(median))
        maxima.append(max_abs)
        if max_abs >= config.suspicious_raw_rail or near_nominal_rail > 0.01:
            rail_hits.append(channel)

    centered_scaled = raw.sub(raw.median(axis=0), axis=1) * config.count_to_uv
    tracked_abs_p99 = centered_scaled[list(config.tracked_channels)].abs().quantile(0.99)

    likely_kind = "unknown"
    reasons = []
    if float(np.median(medians)) > 1000.0 or float(np.max(maxima)) > 10000.0:
        likely_kind = "raw_ads1299_like_counts"
        reasons.append("large absolute offsets and magnitudes are inconsistent with ready-to-use microvolts")
    if rail_hits:
        reasons.append(f"channels near the nominal OpenBCI rail: {', '.join(rail_hits)}")

    plausible_count_to_uv = bool(float(tracked_abs_p99.median()) <= config.openbci.uv_limit * 1.25)
    plausibility_comment = (
        "centered+scaled tracked channels land in a range that is broadly compatible with the OpenBCI uvLimit"
        if plausible_count_to_uv
        else "centered+scaled tracked channels still look too large for the OpenBCI defaults"
    )

    return {
        "likely_signal_kind": likely_kind,
        "reasons": reasons,
        "count_to_uv_assumption": config.count_to_uv,
        "count_to_uv_plausible": plausible_count_to_uv,
        "count_to_uv_comment": plausibility_comment,
        "tracked_centered_scaled_abs_p99_uv": {k: float(v) for k, v in tracked_abs_p99.items()},
        "raw_channel_overview": per_channel,
        "channels_near_rail": rail_hits,
    }


def build_sanity_window(
    raw_df: pd.DataFrame,
    mode_signals: Dict[str, pd.DataFrame],
    processed_by_mode: Dict[str, pd.DataFrame],
    config: AuditConfig,
) -> pd.DataFrame:
    start, end = choose_sanity_window(raw_df, config)
    channel = config.sanity_channel
    base = raw_df.iloc[start:end][["sample_index", "time_sec", "label_clean"]].copy()
    base[f"{channel}_raw_value"] = mode_signals["raw"].iloc[start:end][channel].to_numpy()
    base[f"{channel}_centered_value"] = mode_signals["centered"].iloc[start:end][channel].to_numpy()
    base[f"{channel}_scaled_value"] = mode_signals["scaled"].iloc[start:end][channel].to_numpy()
    base[f"{channel}_filtered_value"] = mode_signals["filtered"].iloc[start:end][channel].to_numpy()

    for mode, processed_df in processed_by_mode.items():
        base[f"{mode}_{channel}_average_uv"] = processed_df.iloc[start:end][f"{channel}_average_uv"].to_numpy()
        base[f"{mode}_{channel}_lower_threshold"] = processed_df.iloc[start:end][
            f"{channel}_lower_threshold"
        ].to_numpy()
        base[f"{mode}_{channel}_upper_threshold"] = processed_df.iloc[start:end][
            f"{channel}_upper_threshold"
        ].to_numpy()
        base[f"{mode}_{channel}_output_normalized"] = processed_df.iloc[start:end][
            f"{channel}_output_normalized"
        ].to_numpy()
    return base


def to_serializable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    return value


def write_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2, sort_keys=True)

