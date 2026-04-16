from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import AUDIT_ROOT, ReplayGUIConfig
else:
    from .config import AUDIT_ROOT, ReplayGUIConfig


@dataclass
class ReplayModeData:
    mode: str
    label: str
    df: pd.DataFrame
    warning_text: str
    activation_scale: float
    activation_limit: float
    x_trace_limit: float


@dataclass
class ReplayDataset:
    modes: Dict[str, ReplayModeData]
    mode_order: Tuple[str, ...]
    fs_hz: float
    default_mode: str
    primary_aggregation: str
    summary_by_mode: Dict[str, Dict[str, float]]
    raw_df: pd.DataFrame

    @property
    def max_index(self) -> int:
        return len(self.raw_df) - 1


_AUDIT_MODULE_CACHE: Tuple[object, object, object] | None = None


def _load_module(alias: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {alias} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def _load_audit_modules() -> Tuple[object, object, object]:
    global _AUDIT_MODULE_CACHE
    if _AUDIT_MODULE_CACHE is not None:
        return _AUDIT_MODULE_CACHE

    previous_config = sys.modules.get("config")
    audit_config = _load_module("_replay_audit_config", AUDIT_ROOT / "config.py")
    try:
        sys.modules["config"] = audit_config
        audit_utils = _load_module("_replay_audit_utils", AUDIT_ROOT / "audit_utils.py")
        signal_stages = _load_module("_replay_signal_stages", AUDIT_ROOT / "signal_stages.py")
    finally:
        if previous_config is not None:
            sys.modules["config"] = previous_config
        else:
            sys.modules.pop("config", None)

    _AUDIT_MODULE_CACHE = (audit_config, audit_utils, signal_stages)
    return _AUDIT_MODULE_CACHE


def _build_audit_config(gui_config: ReplayGUIConfig) -> object:
    audit_config_module, _, _ = _load_audit_modules()
    return audit_config_module.AuditConfig(csv_path=gui_config.csv_path, fs_hz=gui_config.fs_hz)


def _build_processor_input(
    raw_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    mode: str,
    signal_units: str,
    tracked_channels: Tuple[str, ...],
) -> pd.DataFrame:
    processor_input = raw_df[["sample_index", "time_sec", "label_original", "label_clean"]].copy()
    processor_input["mode"] = mode
    processor_input["signal_units"] = signal_units
    for channel in signal_df.columns:
        processor_input[f"signal_{channel}"] = signal_df[channel].to_numpy()
    for channel in tracked_channels:
        processor_input[f"{channel}_signal_uv"] = signal_df[channel].to_numpy()
    return processor_input


def _mode_warning(summary: Dict[str, float], config: ReplayGUIConfig) -> str:
    warnings: List[str] = []
    if summary["clip_fraction"] >= config.clip_warning_threshold:
        warnings.append("averageuV is clipped most of the time")
    if summary["mid_fraction"] <= config.collapsed_mid_threshold:
        warnings.append("normalized outputs are effectively collapsed")
    elif summary["mid_fraction"] <= config.low_mid_warning_threshold:
        warnings.append("normalized outputs spend little time in the mid-range")
    return " | ".join(warnings)


def _mode_summary(processed_df: pd.DataFrame, config: ReplayGUIConfig) -> Dict[str, float]:
    clip_fraction = []
    mid_fraction = []
    for channel in config.tracked_channels:
        avg_uv = processed_df[f"{channel}_average_uv"].to_numpy(dtype=float)
        normalized = processed_df[f"{channel}_output_normalized"].to_numpy(dtype=float)
        clip_fraction.append(float(np.mean(avg_uv >= 200.0 - 1e-9)))
        low = float(np.mean(normalized <= 0.05))
        high = float(np.mean(normalized >= 0.95))
        mid_fraction.append(max(0.0, 1.0 - low - high))

    x_limit = float(np.nanquantile(np.abs(processed_df[f"{config.primary_aggregation}_x_smooth"]), 0.995))
    x_limit = max(config.x_smooth_display_limit, x_limit * 1.05)
    activation_scale = float(
        np.nanquantile(processed_df[f"{config.primary_aggregation}_total_activation"], 0.95)
    )
    activation_scale = max(activation_scale, 1e-6)
    activation_limit = max(activation_scale * 1.1, 1.0)
    return {
        "clip_fraction": float(np.mean(clip_fraction)),
        "mid_fraction": float(np.mean(mid_fraction)),
        "x_trace_limit": x_limit,
        "activation_scale": activation_scale,
        "activation_limit": activation_limit,
    }


def build_replay_dataset(config: ReplayGUIConfig) -> ReplayDataset:
    audit_config, audit_utils, signal_stages = _load_audit_modules()
    audit_cfg = _build_audit_config(config)
    raw_df = audit_utils.load_emg_csv(config.csv_path, audit_cfg)
    mode_signals, mode_metadata = signal_stages.build_mode_signals(raw_df, audit_cfg)

    modes: Dict[str, ReplayModeData] = {}
    summary_by_mode: Dict[str, Dict[str, float]] = {}

    for mode in config.modes:
        signal_df = mode_signals[mode]
        processor_input = _build_processor_input(
            raw_df,
            signal_df,
            mode=mode,
            signal_units=mode_metadata[mode]["units"],
            tracked_channels=config.tracked_channels,
        )
        processor = signal_stages.OpenBCIEMGJoystick1DProcessor(audit_cfg)
        processor_outputs = processor.process(processor_input)
        processed_df = pd.concat([processor_input, processor_outputs], axis=1)

        summary = _mode_summary(processed_df, config)
        summary_by_mode[mode] = summary
        modes[mode] = ReplayModeData(
            mode=mode,
            label=config.mode_labels[mode],
            df=processed_df,
            warning_text=_mode_warning(summary, config),
            activation_scale=summary["activation_scale"],
            activation_limit=summary["activation_limit"],
            x_trace_limit=summary["x_trace_limit"],
        )

    return ReplayDataset(
        modes=modes,
        mode_order=config.modes,
        fs_hz=config.fs_hz,
        default_mode=config.default_mode,
        primary_aggregation=config.primary_aggregation,
        summary_by_mode=summary_by_mode,
        raw_df=raw_df,
    )
