from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import ReplayGUIConfig
    from replay_data import build_replay_dataset
    from viewer import launch_viewer
else:
    from .config import ReplayGUIConfig
    from .replay_data import build_replay_dataset
    from .viewer import launch_viewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a Matplotlib replay GUI for the OpenBCI-style EMG pipeline."
    )
    parser.add_argument("--csv", type=Path, default=ReplayGUIConfig().csv_path, help="Input CSV path.")
    parser.add_argument("--fs", type=float, default=ReplayGUIConfig().fs_hz, help="Sampling rate in Hz.")
    parser.add_argument(
        "--mode",
        choices=ReplayGUIConfig().modes,
        default=ReplayGUIConfig().default_mode,
        help="Initial preprocessing mode shown in the GUI.",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=ReplayGUIConfig().trace_window_seconds,
        help="Trailing trace window length in seconds.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=ReplayGUIConfig().default_speed,
        help="Initial playback speed multiplier.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build the replay dataset and GUI objects without opening the interactive window.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ReplayGUIConfig:
    base = ReplayGUIConfig()
    return ReplayGUIConfig(
        csv_path=args.csv,
        fs_hz=args.fs,
        default_mode=args.mode,
        default_speed=args.speed,
        trace_window_seconds=args.window_sec,
        timer_interval_ms=base.timer_interval_ms,
        title=base.title,
        primary_aggregation=base.primary_aggregation,
        modes=base.modes,
        mode_labels=base.mode_labels,
        speed_options=base.speed_options,
        tracked_channels=base.tracked_channels,
        label_colors=base.label_colors,
        active_labels=base.active_labels,
        x_smooth_display_limit=base.x_smooth_display_limit,
        clip_warning_threshold=base.clip_warning_threshold,
        low_mid_warning_threshold=base.low_mid_warning_threshold,
        collapsed_mid_threshold=base.collapsed_mid_threshold,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)
    dataset = build_replay_dataset(config)
    launch_viewer(dataset, config, show=not args.no_show)


if __name__ == "__main__":
    main()
