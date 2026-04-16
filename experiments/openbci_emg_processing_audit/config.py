from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = Path(
    "/Users/yanivnaggar/Desktop/Fall 2025/Independent Study/CogniSync-main/Clench_1/Y_EMG.csv"
)
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "outputs"


@dataclass(frozen=True)
class OpenBCIConfig:
    window_seconds: float = 1.0
    uv_limit: float = 200.0
    creep_increasing: float = 0.9
    creep_decreasing: float = 0.99999
    minimum_delta_uv: float = 10.0
    lower_threshold_minimum: float = 6.0
    smoothing: float = 0.9
    initial_lower_threshold: float = 6.0
    initial_upper_threshold: float = 16.0


@dataclass(frozen=True)
class FilterConfig:
    highpass_hz: float = 20.0
    lowpass_hz: float = 100.0
    order: int = 4
    notch_hz: float = 60.0
    notch_q: float = 30.0
    fallback_detrend_seconds: float = 0.5
    fallback_smoothing_samples: int = 3


@dataclass(frozen=True)
class PlotConfig:
    inspection_window_seconds: float = 20.0
    histogram_bins: int = 80
    scatter_sample_limit: int = 12000


@dataclass(frozen=True)
class AuditConfig:
    csv_path: Path = DEFAULT_CSV_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    fs_hz: float = 250.0
    label_column: str = "label"
    all_channels: Tuple[str, ...] = tuple(f"ch{i}" for i in range(1, 9))
    tracked_channels: Tuple[str, ...] = ("ch1", "ch4", "ch3", "ch6")
    channel_map: Dict[str, Tuple[str, ...]] = field(
        default_factory=lambda: {
            "left": ("ch1", "ch4"),
            "right": ("ch3", "ch6"),
        }
    )
    aggregation_methods: Tuple[str, ...] = ("mean", "max")
    modes: Tuple[str, ...] = ("raw", "centered", "scaled", "filtered")
    mode_labels: Dict[str, str] = field(
        default_factory=lambda: {
            "raw": "Mode A: raw input",
            "centered": "Mode B: centered only",
            "scaled": "Mode C: centered + scaled",
            "filtered": "Mode D: centered + scaled + EMG filtered",
        }
    )
    openbci: OpenBCIConfig = field(default_factory=OpenBCIConfig)
    filtering: FilterConfig = field(default_factory=FilterConfig)
    plots: PlotConfig = field(default_factory=PlotConfig)
    count_to_uv: float = 0.02235
    sample_rate_note: str = (
        "Defaulted to 250 Hz because the existing repo uses 250.0 Hz throughout "
        "the OpenBCI processing scripts."
    )
    supported_labels: Tuple[str, ...] = ("norm", "Jaws", "Jleft", "Jright")
    active_labels: Tuple[str, ...] = ("Jaws", "Jleft", "Jright")
    direction_thresholds: Tuple[float, ...] = field(
        default_factory=lambda: tuple(round(i / 100.0, 2) for i in range(0, 101))
    )
    activation_thresholds: Tuple[float, ...] = field(
        default_factory=lambda: tuple(round(i / 100.0, 2) for i in range(0, 201))
    )
    sanity_channel: str = "ch1"
    sanity_window_samples: int = 40
    suspicious_raw_rail: float = 187000.0
    raw_rail_nominal: float = 187500.0
    near_zero_normalized_threshold: float = 0.05
    near_one_normalized_threshold: float = 0.95

