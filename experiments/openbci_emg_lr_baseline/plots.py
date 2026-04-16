from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ExperimentConfig
from utils import choose_plot_window, compute_label_spans


LABEL_COLORS = {
    "norm": "#d9d9d9",
    "Jaws": "#f4d35e",
    "Jleft": "#7db7e8",
    "Jright": "#ee6c4d",
    "missing": "#bbbbbb",
}


def _add_label_spans(ax: plt.Axes, df_window: pd.DataFrame) -> None:
    spans = compute_label_spans(df_window["label_clean"].tolist())
    times = df_window["time_sec"].to_numpy(dtype=float)
    if len(times) == 0:
        return
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


def plot_raw_channels(processed_df: pd.DataFrame, config: ExperimentConfig, output_dir: Path) -> Path:
    start, end = choose_plot_window(processed_df, config)
    window = processed_df.iloc[start:end]
    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
    for ax, channel in zip(axes, config.tracked_channels):
        ax.plot(window["time_sec"], window[channel], linewidth=1.0, color="#2a2a72")
        ax.set_ylabel(channel)
        _add_label_spans(ax, window)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Raw tracked channels (counts) over a small window")
    path = output_dir / "raw_tracked_channels_window.png"
    _finalize(fig, path)
    return path


def plot_normalized_channels(processed_df: pd.DataFrame, config: ExperimentConfig, output_dir: Path) -> Path:
    start, end = choose_plot_window(processed_df, config)
    window = processed_df.iloc[start:end]
    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
    for ax, channel in zip(axes, config.tracked_channels):
        ax.plot(
            window["time_sec"],
            window[f"{channel}_output_normalized"],
            linewidth=1.0,
            color="#198754",
        )
        ax.set_ylabel(channel)
        ax.set_ylim(0, max(1.05, float(window[f"{channel}_output_normalized"].max()) * 1.05))
        _add_label_spans(ax, window)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Normalized OpenBCI-style outputs over a small window")
    path = output_dir / "normalized_tracked_channels_window.png"
    _finalize(fig, path)
    return path


def plot_thresholds(processed_df: pd.DataFrame, config: ExperimentConfig, output_dir: Path) -> Path:
    start, end = choose_plot_window(processed_df, config)
    window = processed_df.iloc[start:end]
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for ax, channel in zip(axes, config.tracked_channels):
        ax.plot(window["time_sec"], window[f"{channel}_average_uv"], label="averageuV", linewidth=1.2)
        ax.plot(
            window["time_sec"],
            window[f"{channel}_lower_threshold"],
            label="lowerThreshold",
            linewidth=1.0,
        )
        ax.plot(
            window["time_sec"],
            window[f"{channel}_upper_threshold"],
            label="upperThreshold",
            linewidth=1.0,
        )
        ax.set_ylabel(channel)
        _add_label_spans(ax, window)
    axes[0].legend(loc="upper right", ncol=3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Adaptive thresholds and averageuV over a small window")
    path = output_dir / "tracked_channel_thresholds_window.png"
    _finalize(fig, path)
    return path


def plot_x_traces(processed_df: pd.DataFrame, config: ExperimentConfig, output_dir: Path) -> Path:
    start, end = choose_plot_window(processed_df, config)
    window = processed_df.iloc[start:end]
    fig, axes = plt.subplots(len(config.aggregation_methods), 1, figsize=(14, 7), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, method in zip(axes, config.aggregation_methods):
        ax.plot(window["time_sec"], window[f"{method}_x_raw"], label=f"{method} x_raw", linewidth=1.0)
        ax.plot(
            window["time_sec"],
            window[f"{method}_x_smooth"],
            label=f"{method} x_smooth",
            linewidth=1.2,
        )
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylabel(method)
        ax.legend(loc="upper right")
        _add_label_spans(ax, window)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Joystick X traces over a small window")
    path = output_dir / "x_raw_x_smooth_window.png"
    _finalize(fig, path)
    return path


def plot_x_histograms(processed_df: pd.DataFrame, config: ExperimentConfig, output_dir: Path) -> Path:
    labels = ["norm", "Jaws", "Jleft", "Jright"]
    fig, axes = plt.subplots(len(config.aggregation_methods), 1, figsize=(12, 8), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, method in zip(axes, config.aggregation_methods):
        for label in labels:
            subset = processed_df.loc[processed_df["label_clean"] == label, f"{method}_x_smooth"]
            ax.hist(
                subset,
                bins=config.plots.histogram_bins,
                density=True,
                alpha=0.4,
                label=label,
                color=LABEL_COLORS.get(label, None),
            )
        ax.set_ylabel(method)
        ax.legend(loc="upper right", ncol=4)
    axes[-1].set_xlabel("x_smooth")
    fig.suptitle("x_smooth distributions by label")
    path = output_dir / "x_smooth_histograms.png"
    _finalize(fig, path)
    return path


def plot_activation_scatter(processed_df: pd.DataFrame, config: ExperimentConfig, output_dir: Path) -> Path:
    fig, axes = plt.subplots(len(config.aggregation_methods), 2, figsize=(14, 8), sharex=False, sharey=False)
    axes = np.atleast_2d(axes)

    if len(processed_df) > config.plots.scatter_sample_limit:
        sample = processed_df.sample(config.plots.scatter_sample_limit, random_state=42)
    else:
        sample = processed_df

    for row_idx, method in enumerate(config.aggregation_methods):
        ax_left = axes[row_idx, 0]
        ax_right = axes[row_idx, 1]
        for label, subset in sample.groupby("label_clean"):
            color = LABEL_COLORS.get(label, "#666666")
            ax_left.scatter(
                subset[f"{method}_total_activation"],
                subset[f"{method}_x_smooth"],
                s=5,
                alpha=0.25,
                color=color,
                label=label,
            )
            ax_right.scatter(
                subset[f"{method}_total_activation"],
                subset[f"{method}_abs_direction"],
                s=5,
                alpha=0.25,
                color=color,
                label=label,
            )
        ax_left.set_title(f"{method}: total_activation vs x_smooth")
        ax_right.set_title(f"{method}: total_activation vs abs_direction")
        ax_left.set_xlabel("total_activation")
        ax_right.set_xlabel("total_activation")
        ax_left.set_ylabel("x_smooth")
        ax_right.set_ylabel("abs_direction")
    axes[0, 0].legend(loc="upper left", ncol=4, markerscale=2)
    fig.suptitle("Activation and directional scatter diagnostics")
    path = output_dir / "activation_direction_scatter.png"
    _finalize(fig, path)
    return path


def plot_threshold_scan(
    direction_scans: Dict[str, pd.DataFrame],
    activation_scans: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="col")
    methods = list(direction_scans.keys())
    for col_idx, method in enumerate(methods):
        direction_df = direction_scans[method]
        activation_df = activation_scans[method]
        axes[0, col_idx].plot(direction_df["threshold"], direction_df["macro_f1"], label="macro_f1")
        axes[0, col_idx].plot(direction_df["threshold"], direction_df["coverage"], label="coverage")
        axes[0, col_idx].set_title(f"{method}: Jleft vs Jright")
        axes[0, col_idx].legend(loc="best")
        axes[1, col_idx].plot(activation_df["threshold"], activation_df["macro_f1"], label="macro_f1")
        axes[1, col_idx].plot(activation_df["threshold"], activation_df["accuracy"], label="accuracy")
        axes[1, col_idx].set_title(f"{method}: norm vs active")
        axes[1, col_idx].legend(loc="best")
        axes[1, col_idx].set_xlabel("threshold")
    path = output_dir / "threshold_scans.png"
    _finalize(fig, path)
    return path


def save_all_plots(
    processed_df: pd.DataFrame,
    config: ExperimentConfig,
    output_dir: Path,
    direction_scans: Dict[str, pd.DataFrame],
    activation_scans: Dict[str, pd.DataFrame],
) -> List[Path]:
    paths = [
        plot_raw_channels(processed_df, config, output_dir),
        plot_normalized_channels(processed_df, config, output_dir),
        plot_thresholds(processed_df, config, output_dir),
        plot_x_traces(processed_df, config, output_dir),
        plot_x_histograms(processed_df, config, output_dir),
        plot_activation_scatter(processed_df, config, output_dir),
        plot_threshold_scan(direction_scans, activation_scans, output_dir),
    ]
    return paths

