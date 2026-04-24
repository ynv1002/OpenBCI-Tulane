from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.bci_game_runtime import BCITrackingGameConfig, LIVE, REPLAY, probe_live_board_connection
    from analysis.bci_tracking_game import BCITrackingGameApp, run_headless_replay_smoke
else:
    from .bci_game_runtime import BCITrackingGameConfig, LIVE, REPLAY, probe_live_board_connection
    from .bci_tracking_game import BCITrackingGameApp, run_headless_replay_smoke


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the closed-loop live/replay BCI rhythm game using the current jaw and hand runtime decoders."
    )
    parser.add_argument("--mode", choices=[LIVE, REPLAY], default=REPLAY, help="Initial mode.")
    parser.add_argument("--replay-csv", type=Path, default=BCITrackingGameConfig().replay_files[0], help="Replay CSV to load in replay mode.")
    parser.add_argument("--replay-speed", type=float, default=1.0, help="Initial replay speed multiplier.")
    parser.add_argument("--jaw-artifact", type=Path, default=BCITrackingGameConfig().jaw_artifact_path)
    parser.add_argument("--direction-artifact", type=Path, default=BCITrackingGameConfig().direction_artifact_path)
    parser.add_argument("--output-root", type=Path, default=BCITrackingGameConfig().output_root)
    parser.add_argument("--calibration-sec", type=float, default=BCITrackingGameConfig().calibration_sec)
    parser.add_argument("--board", type=str, default="cyton", choices=("cyton", "synthetic", "playback"), help="BrainFlow board preset for live mode.")
    parser.add_argument("--serial-port", type=str, default="", help="Cyton serial port for live mode.")
    parser.add_argument("--playback-file", type=Path, help="Playback file for BrainFlow playback board mode.")
    parser.add_argument("--max-live-sec", type=float, help="Optional maximum live runtime in seconds.")
    parser.add_argument("--autostart", action="store_true", help="Start the selected mode as soon as the window opens.")
    parser.add_argument("--autoplay-replay", action="store_true", help="If replay mode is selected, start playback immediately after loading.")
    parser.add_argument("--auto-close-sec", type=float, help="Close the window automatically after this many wall-clock seconds. Useful for smoke tests.")
    parser.add_argument("--headless-smoke", action="store_true", help="Run a replay-mode smoke test without opening Tk. This is intended for verification only.")
    parser.add_argument("--probe-live", action="store_true", help="Probe the selected live board connection and exit without launching Tk.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = BCITrackingGameConfig()
    config = BCITrackingGameConfig(
        jaw_artifact_path=args.jaw_artifact.resolve(),
        direction_artifact_path=args.direction_artifact.resolve(),
        output_root=args.output_root.resolve(),
        replay_files=tuple(path.resolve() for path in base_config.replay_files),
        calibration_sec=float(args.calibration_sec),
        timer_interval_ms=base_config.timer_interval_ms,
        replay_step_sec=base_config.replay_step_sec,
        hand_event_confirmation_sec=base_config.hand_event_confirmation_sec,
        hand_event_lookback_sec=base_config.hand_event_lookback_sec,
        hand_action_latch_sec=base_config.hand_action_latch_sec,
        hand_switch_cooldown_sec=base_config.hand_switch_cooldown_sec,
        jaw_hold_probability_threshold=base_config.jaw_hold_probability_threshold,
        jaw_hold_onset_sec=base_config.jaw_hold_onset_sec,
        jaw_hold_release_sec=base_config.jaw_hold_release_sec,
        stale_stream_warning_sec=base_config.stale_stream_warning_sec,
    )
    resolved_playback_file = args.playback_file.resolve() if args.playback_file else None
    if args.probe_live:
        probe_summary = probe_live_board_connection(
            board=str(args.board),
            serial_port=str(args.serial_port),
            playback_file=resolved_playback_file,
        )
        print("Live board probe ok.")
        print(f"Board: {probe_summary['board']} (resolved id {probe_summary['board_id']})")
        print(f"Sampling rate: {probe_summary['sampling_rate_hz']} Hz")
        print(f"EEG channels: {probe_summary['eeg_channel_count']}")
        if probe_summary["serial_port"]:
            print(f"Serial port: {probe_summary['serial_port']}")
        if probe_summary["playback_file"]:
            print(f"Playback file: {probe_summary['playback_file']}")
        return
    if args.headless_smoke:
        if args.mode != REPLAY:
            raise SystemExit("--headless-smoke only supports replay mode.")
        summary = run_headless_replay_smoke(
            config=config,
            replay_csv=args.replay_csv.resolve(),
            replay_speed=float(args.replay_speed),
            calibration_sec=float(args.calibration_sec),
        )
        print(f"Saved smoke-test logs to {summary['paths']['metadata_path'].parent}")
        print(f"Final score: {summary['final_score']} | hits: {summary['final_hits']} | misses: {summary['final_misses']}")
        print(f"Final decoded state: {summary['final_state']}")
        return
    app = BCITrackingGameApp(
        config=config,
        initial_mode=args.mode,
        replay_csv=args.replay_csv.resolve(),
        replay_speed=float(args.replay_speed),
        board=str(args.board),
        serial_port=str(args.serial_port),
        playback_file=resolved_playback_file,
        max_live_sec=float(args.max_live_sec) if args.max_live_sec is not None else None,
        autostart=bool(args.autostart),
        autoplay_replay=bool(args.autoplay_replay),
        auto_close_sec=float(args.auto_close_sec) if args.auto_close_sec is not None else None,
    )
    app.run()


if __name__ == "__main__":
    main()
