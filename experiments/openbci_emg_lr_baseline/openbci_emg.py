from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List

import numpy as np
import pandas as pd

from config import ExperimentConfig


@dataclass
class ChannelState:
    average_uv: float
    lower_threshold: float
    upper_threshold: float
    output_normalized: float


class AdaptiveMAVChannel:
    def __init__(self, channel_name: str, config: ExperimentConfig) -> None:
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
    def __init__(self, config: ExperimentConfig) -> None:
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
        for sample_idx in range(signal_matrix.shape[0]):
            row = signal_matrix[sample_idx]
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

