from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from config import AuditConfig

try:
    from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


@dataclass
class ChannelState:
    average_uv: float
    lower_threshold: float
    upper_threshold: float
    output_normalized: float


class AdaptiveMAVChannel:
    def __init__(self, channel_name: str, config: AuditConfig) -> None:
        self.channel_name = channel_name
        self.config = config
        self.window_samples = max(1, int(round(config.openbci.window_seconds * config.fs_hz)))
        self.samples: Deque[float] = deque(maxlen=self.window_samples)
        self.lower_threshold = float(config.openbci.initial_lower_threshold)
        self.upper_threshold = float(config.openbci.initial_upper_threshold)

    def update(self, sample_uv: float) -> ChannelState:
        clipped_abs_sample = min(abs(float(sample_uv)), self.config.openbci.uv_limit)
        self.samples.append(clipped_abs_sample)
        average_uv = float(np.mean(self.samples))

        if average_uv >= self.upper_threshold and average_uv <= self.config.openbci.uv_limit:
            self.upper_threshold = average_uv
        if average_uv <= self.lower_threshold:
            self.lower_threshold = average_uv
        if average_uv + self.config.openbci.minimum_delta_uv <= self.upper_threshold:
            self.upper_threshold *= self.config.openbci.creep_increasing
        if self.lower_threshold <= 1:
            self.lower_threshold = 1.0
        if self.lower_threshold <= average_uv:
            self.lower_threshold *= 1.0 / self.config.openbci.creep_decreasing
        if self.lower_threshold < self.config.openbci.lower_threshold_minimum:
            self.lower_threshold = self.config.openbci.lower_threshold_minimum
        if self.upper_threshold <= self.lower_threshold + self.config.openbci.minimum_delta_uv:
            self.upper_threshold = self.lower_threshold + self.config.openbci.minimum_delta_uv

        output_normalized = (average_uv - self.lower_threshold) / (
            self.upper_threshold - self.lower_threshold
        )
        output_normalized = max(0.0, float(output_normalized))

        return ChannelState(
            average_uv=average_uv,
            lower_threshold=float(self.lower_threshold),
            upper_threshold=float(self.upper_threshold),
            output_normalized=output_normalized,
        )


class OpenBCIEMGJoystick1DProcessor:
    def __init__(self, config: AuditConfig) -> None:
        self.config = config
        self.channel_processors = {
            channel: AdaptiveMAVChannel(channel, config) for channel in config.tracked_channels
        }
        self.prev_x = {method: 0.0 for method in config.aggregation_methods}

    def _aggregate(self, values: Iterable[float], method: str) -> float:
        data = np.asarray(list(values), dtype=float)
        if method == "mean":
            return float(np.mean(data))
        if method == "max":
            return float(np.max(data))
        raise ValueError(f"Unsupported aggregation method: {method}")

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        signal_cols = [f"{channel}_signal_uv" for channel in self.config.tracked_channels]
        signal_matrix = df[signal_cols].to_numpy(dtype=float)

        records: List[Dict[str, float]] = []
        for row in signal_matrix:
            channel_states: Dict[str, ChannelState] = {}
            for channel, value in zip(self.config.tracked_channels, row):
                channel_states[channel] = self.channel_processors[channel].update(value)

            record: Dict[str, float] = {}
            for channel, state in channel_states.items():
                record[f"{channel}_average_uv"] = state.average_uv
                record[f"{channel}_lower_threshold"] = state.lower_threshold
                record[f"{channel}_upper_threshold"] = state.upper_threshold
                record[f"{channel}_output_normalized"] = state.output_normalized

            for method in self.config.aggregation_methods:
                left_values = [
                    channel_states[channel].output_normalized
                    for channel in self.config.channel_map["left"]
                ]
                right_values = [
                    channel_states[channel].output_normalized
                    for channel in self.config.channel_map["right"]
                ]
                left_aggregate = self._aggregate(left_values, method)
                right_aggregate = self._aggregate(right_values, method)
                x_raw = right_aggregate - left_aggregate
                prev_x = self.prev_x[method]
                x_smooth = prev_x + (1.0 - self.config.openbci.smoothing) * (x_raw - prev_x)
                self.prev_x[method] = x_smooth

                record[f"{method}_left_aggregate"] = left_aggregate
                record[f"{method}_right_aggregate"] = right_aggregate
                record[f"{method}_x_raw"] = x_raw
                record[f"{method}_x_smooth"] = x_smooth
                record[f"{method}_total_activation"] = left_aggregate + right_aggregate
                record[f"{method}_abs_direction"] = abs(x_smooth)

            records.append(record)
        return pd.DataFrame.from_records(records, index=df.index)


def _apply_filter_frame(df: pd.DataFrame, config: AuditConfig) -> Tuple[pd.DataFrame, Dict[str, object]]:
    signal_df = df.copy()
    if SCIPY_AVAILABLE:
        nyq = 0.5 * config.fs_hz
        low = config.filtering.highpass_hz / nyq
        high = config.filtering.lowpass_hz / nyq
        sos = butter(config.filtering.order, [low, high], btype="bandpass", output="sos")
        b_notch, a_notch = iirnotch(
            config.filtering.notch_hz, config.filtering.notch_q, fs=config.fs_hz
        )
        filtered = signal_df.copy()
        for channel in signal_df.columns:
            x = signal_df[channel].to_numpy(dtype=float)
            x = filtfilt(b_notch, a_notch, x)
            x = sosfiltfilt(sos, x)
            filtered[channel] = x
        return filtered, {
            "filter_backend": "scipy",
            "pipeline": [
                f"notch {config.filtering.notch_hz:.1f} Hz (Q={config.filtering.notch_q:.1f})",
                (
                    f"butter bandpass {config.filtering.highpass_hz:.1f}-"
                    f"{config.filtering.lowpass_hz:.1f} Hz order {config.filtering.order}"
                ),
            ],
        }

    detrend_window = max(3, int(round(config.filtering.fallback_detrend_seconds * config.fs_hz)))
    filtered = signal_df.copy()
    for channel in signal_df.columns:
        series = signal_df[channel]
        baseline = series.rolling(window=detrend_window, center=True, min_periods=1).mean()
        highpassed = series - baseline
        smoothed = highpassed.rolling(
            window=config.filtering.fallback_smoothing_samples, center=True, min_periods=1
        ).mean()
        filtered[channel] = smoothed
    return filtered, {
        "filter_backend": "fallback",
        "pipeline": [
            f"rolling mean detrend window {detrend_window} samples",
            f"rolling mean smoothing window {config.filtering.fallback_smoothing_samples} samples",
        ],
    }


def build_mode_signals(
    raw_df: pd.DataFrame,
    config: AuditConfig,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, object]]]:
    raw_signals = raw_df[list(config.all_channels)].astype(float).copy()
    centered = raw_signals.sub(raw_signals.median(axis=0), axis=1)
    scaled = centered * config.count_to_uv
    filtered, filter_meta = _apply_filter_frame(scaled, config)

    mode_signals = {
        "raw": raw_signals,
        "centered": centered,
        "scaled": scaled,
        "filtered": filtered,
    }
    metadata = {
        "raw": {
            "description": config.mode_labels["raw"],
            "units": "raw_input_units",
            "centering": "none",
            "scaling": "none",
            "filtering": "none",
        },
        "centered": {
            "description": config.mode_labels["centered"],
            "units": "raw_input_units_centered",
            "centering": "per-channel median subtraction",
            "scaling": "none",
            "filtering": "none",
        },
        "scaled": {
            "description": config.mode_labels["scaled"],
            "units": "uV_assumed",
            "centering": "per-channel median subtraction",
            "scaling": f"{config.count_to_uv:.5f} uV/count",
            "filtering": "none",
        },
        "filtered": {
            "description": config.mode_labels["filtered"],
            "units": "uV_assumed_filtered",
            "centering": "per-channel median subtraction",
            "scaling": f"{config.count_to_uv:.5f} uV/count",
            "filtering": filter_meta,
        },
    }
    return mode_signals, metadata

