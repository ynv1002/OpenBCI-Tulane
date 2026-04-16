from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_utils import choose_plot_window, compute_label_spans
from config import AuditConfig


LABEL_COLORS = {
    "norm": "#d9d9d9",
    "Jaws": "#f4d35e",
    "Jleft": "#7db7e8",
    "Jright": "#ee6c4d",
    "missing": "#bdbdbd",
}


def _add_label_spans(ax: plt.Axes, df_window: pd.DataFrame) -> None:
    times = df_window["time_sec"].to_numpy(dtype=float)
    spans = compute_label_spans(df_window["label_clean"].tolist())
    for start, end, label in spans:
        ax.axvspan(
            times[start],
            times[end],
            color=LABEL_COLORS.get(label, "#cccccc"),
            alpha=0.12,
            linewidth=0.0,
        )


def _finalize(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_tracked_signals_by_mode(
    raw_df: pd.DataFrame,
    mode_signals: Dict[str, pd.DataFrame],
    config: AuditConfig,
    output_dir: Path,
) -> List[Path]:
    start, end = choose_plot_window(raw_df, config)
    raw_window = raw_df.iloc[start:end]
    paths = []
    for mode, signal_df in mode_signals.items():
        window = signal_df.iloc[start:end]
        fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
        for ax, channel in zip(axes, config.tracked_channels):
            ax.plot(raw_window["time_sec"], window[channel], linewidth=1.0, color="#1d3557")
            ax.set_ylabel(channel)
            _add_label_spans(ax, raw_window)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"{config.mode_labels[mode]}: tracked channels")
        path = output_dir / f"tracked_channels_{mode}.png"
        _finalize(fig, path)
        paths.append(path)
    return paths


def plot_all_channel_overview(raw_df: pd.DataFrame, config: AuditConfig, output_dir: Path) -> Path:
    start, end = choose_plot_window(raw_df, config)
    window = raw_df.iloc[start:end]
    fig, axes = plt.subplots(8, 1, figsize=(14, 14), sharex=True)
    for ax, channel in zip(axes, config.all_channels):
        ax.plot(window["time_sec"], window[channel], linewidth=0.9, color="#4c956c")
        ax.set_ylabel(channel)
        _add_label_spans(ax, window)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("All-channel raw overview")
    path = output_dir / "all_channel_raw_overview.png"
    _finalize(fig, path)
    return path


def plot_channel_quality_overview(
    diagnostics_df: pd.DataFrame,
    config: AuditConfig,
    output_dir: Path,
) -> Path:
    raw_diag = diagnostics_df.loc[diagnostics_df["mode"] == "raw"].copy()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].bar(raw_diag["channel"], raw_diag["std"], color="#457b9d")
    axes[0].set_ylabel("raw std")
    axes[1].bar(
        raw_diag["channel"],
        raw_diag["raw_fraction_near_nominal_rail"],
        color=["#d62828" if flag else "#6c757d" for flag in raw_diag["flag_suspicious_saturation"]],
    )
    axes[1].set_ylabel("fraction near rail")
    axes[1].set_xlabel("channel")
    fig.suptitle("Raw-channel quality overview")
    path = output_dir / "channel_quality_overview.png"
    _finalize(fig, path)
    return path


def plot_threshold_dynamics_by_mode(
    processed_by_mode: Dict[str, pd.DataFrame],
    config: AuditConfig,
    output_dir: Path,
) -> List[Path]:
    paths = []
    for mode, processed_df in processed_by_mode.items():
        start, end = choose_plot_window(processed_df, config)
        window = processed_df.iloc[start:end]
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        for ax, channel in zip(axes, config.tracked_channels):
            ax.plot(window["time_sec"], window[f"{channel}_average_uv"], label="averageuV", linewidth=1.2)
            ax.plot(window["time_sec"], window[f"{channel}_lower_threshold"], label="lowerThreshold", linewidth=1.0)
            ax.plot(window["time_sec"], window[f"{channel}_upper_threshold"], label="upperThreshold", linewidth=1.0)
            ax.set_ylabel(channel)
            _add_label_spans(ax, window)
        axes[0].legend(loc="upper right", ncol=3)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"{config.mode_labels[mode]}: threshold dynamics")
        path = output_dir / f"threshold_dynamics_{mode}.png"
        _finalize(fig, path)
        paths.append(path)
    return paths


def plot_normalized_outputs_by_mode(
    processed_by_mode: Dict[str, pd.DataFrame],
    config: AuditConfig,
    output_dir: Path,
) -> List[Path]:
    paths = []
    for mode, processed_df in processed_by_mode.items():
        start, end = choose_plot_window(processed_df, config)
        window = processed_df.iloc[start:end]
        fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
        for ax, channel in zip(axes, config.tracked_channels):
            ax.plot(window["time_sec"], window[f"{channel}_output_normalized"], linewidth=1.0, color="#198754")
            ax.set_ylabel(channel)
            ax.set_ylim(0.0, max(1.05, float(window[f"{channel}_output_normalized"].max()) * 1.05))
            _add_label_spans(ax, window)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"{config.mode_labels[mode]}: normalized outputs")
        path = output_dir / f"normalized_outputs_{mode}.png"
        _finalize(fig, path)
        paths.append(path)
    return paths


def plot_x_traces_by_mode(
    processed_by_mode: Dict[str, pd.DataFrame],
    config: AuditConfig,
    output_dir: Path,
) -> List[Path]:
    paths = []
    for mode, processed_df in processed_by_mode.items():
        start, end = choose_plot_window(processed_df, config)
        window = processed_df.iloc[start:end]
        fig, axes = plt.subplots(len(config.aggregation_methods), 1, figsize=(14, 7), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, method in zip(axes, config.aggregation_methods):
            ax.plot(window["time_sec"], window[f"{method}_x_raw"], label=f"{method} x_raw", linewidth=1.0)
            ax.plot(window["time_sec"], window[f"{method}_x_smooth"], label=f"{method} x_smooth", linewidth=1.2)
            ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
            ax.set_ylabel(method)
            ax.legend(loc="upper right")
            _add_label_spans(ax, window)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"{config.mode_labels[mode]}: x_raw and x_smooth")
        path = output_dir / f"x_traces_{mode}.png"
        _finalize(fig, path)
        paths.append(path)
    return paths


def plot_x_smooth_histograms(
    processed_by_mode: Dict[str, pd.DataFrame],
    config: AuditConfig,
    output_dir: Path,
) -> List[Path]:
    labels = ["norm", "Jaws", "Jleft", "Jright"]
    paths = []
    for method in config.aggregation_methods:
        fig, axes = plt.subplots(len(config.modes), 1, figsize=(12, 12), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, mode in zip(axes, config.modes):
            processed_df = processed_by_mode[mode]
            for label in labels:
                subset = processed_df.loc[processed_df["label_clean"] == label, f"{method}_x_smooth"]
                color = LABEL_COLORS.get(label)
                if subset.nunique(dropna=True) <= 1:
                    value = float(subset.iloc[0]) if len(subset) else 0.0
                    ax.axvline(value, color=color, alpha=0.8, linewidth=1.2, label=label)
                else:
                    ax.hist(
                        subset,
                        bins=config.plots.histogram_bins,
                        density=True,
                        alpha=0.35,
                        label=label,
                        color=color,
                    )
            ax.set_ylabel(mode)
            ax.legend(loc="upper right", ncol=4)
        axes[-1].set_xlabel("x_smooth")
        fig.suptitle(f"x_smooth distributions by label and mode ({method})")
        path = output_dir / f"x_smooth_histograms_{method}.png"
        _finalize(fig, path)
        paths.append(path)
    return paths


def plot_activation_scatter(
    processed_by_mode: Dict[str, pd.DataFrame],
    config: AuditConfig,
    output_dir: Path,
) -> List[Path]:
    paths = []
    for method in config.aggregation_methods:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False, sharey=False)
        axes = axes.ravel()
        for ax, mode in zip(axes, config.modes):
            processed_df = processed_by_mode[mode]
            if len(processed_df) > config.plots.scatter_sample_limit:
                sample = processed_df.sample(config.plots.scatter_sample_limit, random_state=42)
            else:
                sample = processed_df
            for label, subset in sample.groupby("label_clean"):
                ax.scatter(
                    subset[f"{method}_total_activation"],
                    subset[f"{method}_x_smooth"],
                    s=5,
                    alpha=0.25,
                    color=LABEL_COLORS.get(label, "#666666"),
                    label=label,
                )
            ax.set_title(mode)
            ax.set_xlabel("total_activation")
            ax.set_ylabel("x_smooth")
        axes[0].legend(loc="upper left", ncol=4, markerscale=2)
        fig.suptitle(f"total_activation vs x_smooth by mode ({method})")
        path = output_dir / f"activation_scatter_{method}.png"
        _finalize(fig, path)
        paths.append(path)
    return paths


def save_all_plots(
    raw_df: pd.DataFrame,
    mode_signals: Dict[str, pd.DataFrame],
    processed_by_mode: Dict[str, pd.DataFrame],
    diagnostics_df: pd.DataFrame,
    config: AuditConfig,
    output_dir: Path,
) -> List[Path]:
    paths: List[Path] = []
    paths.extend(plot_tracked_signals_by_mode(raw_df, mode_signals, config, output_dir))
    paths.append(plot_all_channel_overview(raw_df, config, output_dir))
    paths.append(plot_channel_quality_overview(diagnostics_df, config, output_dir))
    paths.extend(plot_threshold_dynamics_by_mode(processed_by_mode, config, output_dir))
    paths.extend(plot_normalized_outputs_by_mode(processed_by_mode, config, output_dir))
    paths.extend(plot_x_traces_by_mode(processed_by_mode, config, output_dir))
    paths.extend(plot_x_smooth_histograms(processed_by_mode, config, output_dir))
    paths.extend(plot_activation_scatter(processed_by_mode, config, output_dir))
    return paths
