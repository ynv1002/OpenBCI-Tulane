from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, RadioButtons, Slider

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import ReplayGUIConfig
    from replay_data import ReplayDataset
else:
    from .config import ReplayGUIConfig
    from .replay_data import ReplayDataset


@dataclass
class _TraceArtists:
    x_line: object
    activation_line: object
    x_cursor: object
    activation_cursor: object


class ReplayViewer:
    def __init__(self, dataset: ReplayDataset, config: ReplayGUIConfig) -> None:
        self.dataset = dataset
        self.config = config
        self.current_mode = config.default_mode
        self.current_index = 0
        self.playing = False
        self.playback_speed = config.default_speed
        self._slider_event_enabled = True

        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title(config.title)
        self._build_axes()
        self._build_widgets()
        self._build_static_plot_objects()
        self._connect_events()

        self.timer = self.fig.canvas.new_timer(interval=self.config.timer_interval_ms)
        self.timer.add_callback(self._on_timer)
        self.timer.start()
        self._set_mode(self.current_mode, redraw_overview=True)

    def _build_axes(self) -> None:
        self.ax_status = self.fig.add_axes([0.05, 0.905, 0.73, 0.075])
        self.ax_ticker = self.fig.add_axes([0.05, 0.69, 0.58, 0.16])
        self.ax_activation = self.fig.add_axes([0.68, 0.60, 0.10, 0.25])
        self.ax_trace = self.fig.add_axes([0.05, 0.31, 0.73, 0.27])
        self.ax_trace_activation = self.ax_trace.twinx()
        self.ax_overview = self.fig.add_axes([0.05, 0.12, 0.73, 0.13])
        self.ax_slider = self.fig.add_axes([0.05, 0.05, 0.73, 0.035])

        self.ax_play = self.fig.add_axes([0.82, 0.12, 0.05, 0.05])
        self.ax_back = self.fig.add_axes([0.88, 0.12, 0.05, 0.05])
        self.ax_forward = self.fig.add_axes([0.94, 0.12, 0.05, 0.05])
        self.ax_mode = self.fig.add_axes([0.82, 0.52, 0.17, 0.22])
        self.ax_speed = self.fig.add_axes([0.82, 0.29, 0.17, 0.17])

    def _build_widgets(self) -> None:
        self.time_slider = Slider(
            self.ax_slider,
            "Sample",
            0,
            self.dataset.max_index,
            valinit=0,
            valstep=1,
        )
        self.play_button = Button(self.ax_play, "Play")
        self.back_button = Button(self.ax_back, "<")
        self.forward_button = Button(self.ax_forward, ">")
        self.mode_radio = RadioButtons(
            self.ax_mode,
            [self.config.mode_labels[mode] for mode in self.dataset.mode_order],
            active=self.dataset.mode_order.index(self.current_mode),
        )
        speed_labels = [f"{speed:g}x" for speed in self.config.speed_options]
        self.speed_radio = RadioButtons(
            self.ax_speed,
            speed_labels,
            active=self.config.speed_options.index(self.playback_speed),
        )

    def _build_static_plot_objects(self) -> None:
        self.ax_status.axis("off")
        self.status_text = self.ax_status.text(0.00, 0.70, "", fontsize=14, fontweight="bold")
        self.info_text = self.ax_status.text(0.00, 0.28, "", fontsize=11)
        self.warning_text = self.ax_status.text(0.60, 0.28, "", fontsize=11, color="#b22222")

        self.ax_ticker.set_xlim(-self.config.x_smooth_display_limit, self.config.x_smooth_display_limit)
        self.ax_ticker.set_ylim(0.0, 1.0)
        self.ax_ticker.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        self.ax_ticker.set_yticks([])
        self.ax_ticker.set_title("Directional Ticker (mean_x_smooth)")
        self.ax_ticker.text(-0.98, 0.88, "Left", fontsize=11, ha="left")
        self.ax_ticker.text(0.98, 0.88, "Right", fontsize=11, ha="right")
        self.ticker_bar = Rectangle((0.0, 0.20), 0.0, 0.60, color="#6c757d", alpha=0.85)
        self.ax_ticker.add_patch(self.ticker_bar)
        self.ticker_value_text = self.ax_ticker.text(0.0, 0.5, "", ha="center", va="center", fontsize=12)

        self.ax_activation.set_xlim(0.0, 1.0)
        self.ax_activation.set_ylim(0.0, 1.0)
        self.ax_activation.set_xticks([])
        self.ax_activation.set_ylabel("Activation")
        self.ax_activation.set_title("Activation Gauge")
        self.activation_bar = Rectangle((0.15, 0.0), 0.70, 0.0, color="#2a9d8f", alpha=0.85)
        self.ax_activation.add_patch(self.activation_bar)
        self.activation_value_text = self.ax_activation.text(
            0.5, 1.02, "", ha="center", va="bottom", transform=self.ax_activation.transAxes, fontsize=11
        )

        self.ax_trace.set_title("Recent Trace Window")
        self.ax_trace.set_ylabel("mean_x_smooth")
        self.ax_trace_activation.set_ylabel("mean_total_activation")
        (x_line,) = self.ax_trace.plot([], [], color="#1d3557", linewidth=1.8, label="mean_x_smooth")
        (activation_line,) = self.ax_trace_activation.plot(
            [], [], color="#2a9d8f", linewidth=1.2, label="mean_total_activation"
        )
        x_cursor = self.ax_trace.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        activation_cursor = self.ax_trace_activation.axvline(
            0.0, color="black", linestyle="--", linewidth=1.0
        )
        self.trace_artists = _TraceArtists(x_line, activation_line, x_cursor, activation_cursor)

        self.ax_overview.set_title("Full Session Overview (mean_x_smooth)")
        self.ax_overview.set_ylabel("x_smooth")
        (self.overview_line,) = self.ax_overview.plot([], [], color="#6d597a", linewidth=1.0)
        self.overview_cursor = self.ax_overview.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        self._overview_spans: List[object] = []
        self._trace_spans: List[object] = []

    def _connect_events(self) -> None:
        self.time_slider.on_changed(self._on_slider_changed)
        self.play_button.on_clicked(self._on_toggle_play)
        self.back_button.on_clicked(self._on_step_back)
        self.forward_button.on_clicked(self._on_step_forward)
        self.mode_radio.on_clicked(self._on_mode_selected)
        self.speed_radio.on_clicked(self._on_speed_selected)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _current_mode_data(self):
        return self.dataset.modes[self.current_mode]

    def _current_df(self):
        return self._current_mode_data().df

    def _set_mode(self, mode: str, redraw_overview: bool = False) -> None:
        self.current_mode = mode
        mode_data = self._current_mode_data()
        if redraw_overview:
            self._draw_overview(mode_data.df, mode_data.x_trace_limit)
        else:
            self._draw_overview(mode_data.df, mode_data.x_trace_limit)
        self._refresh()

    def _set_index(self, index: int, from_slider: bool = False) -> None:
        self.current_index = int(np.clip(index, 0, self.dataset.max_index))
        if not from_slider:
            self._slider_event_enabled = False
            self.time_slider.set_val(self.current_index)
            self._slider_event_enabled = True
        self._refresh()

    def _format_speed_label(self) -> str:
        return f"{self.playback_speed:g}x"

    def _current_time(self) -> float:
        return float(self._current_df().iloc[self.current_index]["time_sec"])

    def _draw_overview(self, df, x_limit: float) -> None:
        self.overview_line.set_data(df["time_sec"], df[f"{self.config.primary_aggregation}_x_smooth"])
        self.ax_overview.set_xlim(float(df["time_sec"].iloc[0]), float(df["time_sec"].iloc[-1]))
        self.ax_overview.set_ylim(-x_limit, x_limit)
        for patch in self._overview_spans:
            patch.remove()
        self._overview_spans.clear()

        labels = df["label_clean"].to_numpy()
        times = df["time_sec"].to_numpy(dtype=float)
        if len(times):
            start = 0
            current = labels[0]
            for idx in range(1, len(labels)):
                if labels[idx] != current:
                    if current in self.config.label_colors:
                        self._overview_spans.append(
                            self.ax_overview.axvspan(
                                times[start],
                                times[idx - 1],
                                color=self.config.label_colors[current],
                                alpha=0.12,
                                linewidth=0.0,
                            )
                        )
                    start = idx
                    current = labels[idx]
            if current in self.config.label_colors:
                self._overview_spans.append(
                    self.ax_overview.axvspan(
                        times[start],
                        times[-1],
                        color=self.config.label_colors[current],
                        alpha=0.12,
                        linewidth=0.0,
                    )
                )

    def _draw_trace_spans(self, time_values, label_values) -> None:
        for patch in self._trace_spans:
            patch.remove()
        self._trace_spans.clear()
        if len(time_values) == 0:
            return
        start = 0
        current = label_values[0]
        for idx in range(1, len(label_values)):
            if label_values[idx] != current:
                if current in self.config.label_colors:
                    self._trace_spans.append(
                        self.ax_trace.axvspan(
                            time_values[start],
                            time_values[idx - 1],
                            color=self.config.label_colors[current],
                            alpha=0.12,
                            linewidth=0.0,
                        )
                    )
                start = idx
                current = label_values[idx]
        if current in self.config.label_colors:
            self._trace_spans.append(
                self.ax_trace.axvspan(
                    time_values[start],
                    time_values[-1],
                    color=self.config.label_colors[current],
                    alpha=0.12,
                    linewidth=0.0,
                )
            )

    def _refresh(self) -> None:
        mode_data = self._current_mode_data()
        df = mode_data.df
        row = df.iloc[self.current_index]
        current_label = str(row["label_clean"])
        label_color = self.config.label_colors.get(current_label, "#333333")

        self.status_text.set_text(
            f"{self.config.title} | {mode_data.label} mode | Label: {current_label}"
        )
        self.status_text.set_color(label_color)
        self.info_text.set_text(
            f"time={float(row['time_sec']):.3f}s | sample={int(row['sample_index'])} | "
            f"x_smooth={float(row[f'{self.config.primary_aggregation}_x_smooth']):.3f} | "
            f"activation={float(row[f'{self.config.primary_aggregation}_total_activation']):.3f} | "
            f"speed={self._format_speed_label()}"
        )
        self.warning_text.set_text(mode_data.warning_text)

        x_value = float(np.clip(row[f"{self.config.primary_aggregation}_x_smooth"], -1.0, 1.0))
        left = min(0.0, x_value)
        width = abs(x_value)
        self.ticker_bar.set_x(left)
        self.ticker_bar.set_width(width)
        self.ticker_bar.set_color("#ee6c4d" if x_value >= 0 else "#457b9d")
        self.ticker_value_text.set_text(f"{x_value:+.3f}")

        activation_value = float(row[f"{self.config.primary_aggregation}_total_activation"])
        activation_norm = float(np.clip(activation_value / mode_data.activation_scale, 0.0, 1.0))
        self.activation_bar.set_height(activation_norm)
        self.activation_value_text.set_text(
            f"{activation_value:.3f} ({activation_norm * 100:.0f}% of p95)"
        )

        window_samples = max(1, int(round(self.config.trace_window_seconds * self.dataset.fs_hz)))
        start_index = max(0, self.current_index - window_samples + 1)
        window = df.iloc[start_index : self.current_index + 1]
        times = window["time_sec"].to_numpy(dtype=float)
        x_trace = window[f"{self.config.primary_aggregation}_x_smooth"].to_numpy(dtype=float)
        activation_trace = window[f"{self.config.primary_aggregation}_total_activation"].to_numpy(dtype=float)

        self.trace_artists.x_line.set_data(times, x_trace)
        self.trace_artists.activation_line.set_data(times, activation_trace)
        current_time = float(row["time_sec"])
        self.trace_artists.x_cursor.set_xdata([current_time, current_time])
        self.trace_artists.activation_cursor.set_xdata([current_time, current_time])
        if len(times) <= 1 or float(times[0]) == float(times[-1]):
            self.ax_trace.set_xlim(current_time - 0.5, current_time + 0.5)
        else:
            self.ax_trace.set_xlim(float(times[0]), float(times[-1]))
        self.ax_trace.set_ylim(-mode_data.x_trace_limit, mode_data.x_trace_limit)
        self.ax_trace_activation.set_ylim(0.0, mode_data.activation_limit)
        self._draw_trace_spans(times, window["label_clean"].tolist())

        self.overview_cursor.set_xdata([current_time, current_time])
        self.fig.canvas.draw_idle()

    def _timer_step_samples(self) -> int:
        seconds_per_tick = self.config.timer_interval_ms / 1000.0
        return max(1, int(round(self.dataset.fs_hz * self.playback_speed * seconds_per_tick)))

    def _on_slider_changed(self, value) -> None:
        if not self._slider_event_enabled:
            return
        self._set_index(int(value), from_slider=True)

    def _on_toggle_play(self, _event) -> None:
        self.playing = not self.playing
        self.play_button.label.set_text("Pause" if self.playing else "Play")
        self.fig.canvas.draw_idle()

    def _on_step_back(self, _event) -> None:
        self.playing = False
        self.play_button.label.set_text("Play")
        self._set_index(self.current_index - self._timer_step_samples())

    def _on_step_forward(self, _event) -> None:
        self.playing = False
        self.play_button.label.set_text("Play")
        self._set_index(self.current_index + self._timer_step_samples())

    def _on_mode_selected(self, label: str) -> None:
        inverse_mode_map = {value: key for key, value in self.config.mode_labels.items()}
        self._set_mode(inverse_mode_map[label], redraw_overview=True)

    def _on_speed_selected(self, label: str) -> None:
        self.playback_speed = float(label.rstrip("x"))
        self._refresh()

    def _on_timer(self) -> None:
        if not self.playing:
            return
        next_index = self.current_index + self._timer_step_samples()
        if next_index >= self.dataset.max_index:
            self.playing = False
            self.play_button.label.set_text("Play")
            next_index = self.dataset.max_index
        self._set_index(next_index)

    def _on_close(self, _event) -> None:
        if self.timer is not None:
            self.timer.stop()


def launch_viewer(dataset: ReplayDataset, config: ReplayGUIConfig, show: bool = True) -> ReplayViewer:
    viewer = ReplayViewer(dataset, config)
    if show:
        plt.show()
    return viewer
