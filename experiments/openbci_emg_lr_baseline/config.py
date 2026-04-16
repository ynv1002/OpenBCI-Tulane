from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
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
class ScaleConfig:
    mode: str = "auto"
    openbci_count_to_uv: float = 0.02235
    counts_abs_median_trigger: float = 1000.0
    counts_abs_max_trigger: float = 10000.0
    center_strategy: str = "median"


@dataclass(frozen=True)
class PlotConfig:
    inspection_window_seconds: float = 20.0
    histogram_bins: int = 80
    scatter_sample_limit: int = 12000


@dataclass(frozen=True)
class ExperimentConfig:
    csv_path: Path = DEFAULT_CSV_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    fs_hz: float = 250.0
    sample_rate_note: str = (
        "Defaulted to 250 Hz because the existing repo uses 250.0 Hz throughout "
        "the OpenBCI processing scripts."
    )
    label_column: str = "label"
    tracked_channels: Tuple[str, ...] = ("ch1", "ch4", "ch3", "ch6")
    channel_map: Dict[str, Tuple[str, ...]] = field(
        default_factory=lambda: {
            "left": ("ch1", "ch4"),
            "right": ("ch3", "ch6"),
        }
    )
    aggregation_methods: Tuple[str, ...] = ("mean", "max")
    primary_aggregation: str = "mean"
    openbci: OpenBCIConfig = field(default_factory=OpenBCIConfig)
    scaling: ScaleConfig = field(default_factory=ScaleConfig)
    plots: PlotConfig = field(default_factory=PlotConfig)
    direction_thresholds: Tuple[float, ...] = field(
        default_factory=lambda: tuple(round(i / 100.0, 2) for i in range(0, 101))
    )
    activation_thresholds: Tuple[float, ...] = field(
        default_factory=lambda: tuple(round(i / 100.0, 2) for i in range(0, 201))
    )
    keep_aux_columns: bool = True
    supported_labels: Tuple[str, ...] = ("norm", "Jaws", "Jleft", "Jright")
    active_labels: Tuple[str, ...] = ("Jaws", "Jleft", "Jright")

