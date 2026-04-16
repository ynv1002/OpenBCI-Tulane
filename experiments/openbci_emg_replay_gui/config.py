from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = Path(
    "/Users/yanivnaggar/Desktop/Fall 2025/Independent Study/CogniSync-main/Clench_1/Y_EMG.csv"
)
AUDIT_ROOT = EXPERIMENT_ROOT.parent / "openbci_emg_processing_audit"


@dataclass(frozen=True)
class ReplayGUIConfig:
    csv_path: Path = DEFAULT_CSV_PATH
    fs_hz: float = 250.0
    default_mode: str = "scaled"
    default_speed: float = 1.0
    trace_window_seconds: float = 8.0
    timer_interval_ms: int = 40
    title: str = "OpenBCI EMG Replay"
    primary_aggregation: str = "mean"
    modes: Tuple[str, ...] = ("raw", "centered", "scaled", "filtered")
    mode_labels: Dict[str, str] = field(
        default_factory=lambda: {
            "raw": "Raw",
            "centered": "Centered",
            "scaled": "Scaled",
            "filtered": "Filtered",
        }
    )
    speed_options: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    tracked_channels: Tuple[str, ...] = ("ch1", "ch4", "ch3", "ch6")
    label_colors: Dict[str, str] = field(
        default_factory=lambda: {
            "norm": "#d9d9d9",
            "Jaws": "#f4d35e",
            "Jleft": "#7db7e8",
            "Jright": "#ee6c4d",
            "missing": "#bdbdbd",
        }
    )
    active_labels: Tuple[str, ...] = ("Jaws", "Jleft", "Jright")
    x_smooth_display_limit: float = 1.0
    clip_warning_threshold: float = 0.5
    low_mid_warning_threshold: float = 0.1
    collapsed_mid_threshold: float = 0.01
