from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.hybrid_bci_tester import (
        DEFAULT_REPLAY_FILES,
        LIVE,
        REPLAY,
        HybridTesterApp,
        HybridTesterConfig,
        run_headless_replay,
    )
else:
    from .hybrid_bci_tester import (
        DEFAULT_REPLAY_FILES,
        LIVE,
        REPLAY,
        HybridTesterApp,
        HybridTesterConfig,
        run_headless_replay,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the hybrid BCI tester GUI for LEFT/RIGHT/JAW/REST live and replay debugging."
    )
    parser.add_argument("--mode", choices=[LIVE, REPLAY], default=REPLAY, help="Initial mode.")
    parser.add_argument(
        "--replay-csv",
        type=Path,
        default=DEFAULT_REPLAY_FILES[0],
        help="Replay CSV path. Defaults to the first high-trust Yaniv LR session.",
    )
    parser.add_argument("--replay-speed", type=float, default=1.0, help="Initial replay speed multiplier.")
    parser.add_argument(
        "--sequence-type",
        choices=["random_balanced", "fixed"],
        default="random_balanced",
        help="Live prompt sequence type.",
    )
    parser.add_argument("--jaw-artifact", type=Path, default=HybridTesterConfig().jaw_artifact_path)
    parser.add_argument("--direction-artifact", type=Path, default=HybridTesterConfig().direction_artifact_path)
    parser.add_argument("--output-root", type=Path, default=HybridTesterConfig().output_root)
    parser.add_argument("--calibration-sec", type=float, default=HybridTesterConfig().calibration_sec)
    parser.add_argument("--prompt-timeout-sec", type=float, default=HybridTesterConfig().prompt_timeout_sec)
    parser.add_argument("--prompt-seed", type=int, default=HybridTesterConfig().prompt_seed)
    parser.add_argument("--sequence-repetitions", type=int, default=HybridTesterConfig().sequence_repetitions)
    parser.add_argument(
        "--board",
        type=str,
        default="cyton",
        choices=("cyton", "synthetic", "playback"),
        help="BrainFlow board preset for live mode.",
    )
    parser.add_argument("--serial-port", type=str, default="", help="Cyton serial port for live mode.")
    parser.add_argument("--playback-file", type=Path, help="Playback file for BrainFlow playback board.")
    parser.add_argument("--max-live-sec", type=float, help="Optional max runtime for live smoke tests.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run replay mode without opening Tk. This is intended for replay smoke tests only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = HybridTesterConfig(
        jaw_artifact_path=args.jaw_artifact.resolve(),
        direction_artifact_path=args.direction_artifact.resolve(),
        output_root=args.output_root.resolve(),
        replay_files=tuple(path.resolve() for path in DEFAULT_REPLAY_FILES),
        calibration_sec=float(args.calibration_sec),
        prompt_timeout_sec=float(args.prompt_timeout_sec),
        replay_step_sec=HybridTesterConfig().replay_step_sec,
        timer_interval_ms=HybridTesterConfig().timer_interval_ms,
        flash_sec=HybridTesterConfig().flash_sec,
        lr_event_confirmation_sec=HybridTesterConfig().lr_event_confirmation_sec,
        sequence_repetitions=int(args.sequence_repetitions),
        prompt_seed=int(args.prompt_seed),
    )

    if args.headless:
        if args.mode != REPLAY:
            raise SystemExit("--headless is only supported in replay mode.")
        paths = run_headless_replay(config, args.replay_csv.resolve(), speed=float(args.replay_speed))
        print(f"Saved replay logs to {paths['metadata_path'].parent}")
        print(f"  detections: {paths['detections_path']}")
        print(f"  confidence trace: {paths['trace_path']}")
        print(f"  ground truth segments: {paths['ground_truth_path']}")
        return

    app = HybridTesterApp(
        config=config,
        initial_mode=args.mode,
        replay_csv=args.replay_csv.resolve(),
        replay_speed=float(args.replay_speed),
        sequence_type=str(args.sequence_type),
        board=str(args.board),
        serial_port=str(args.serial_port),
        playback_file=args.playback_file.resolve() if args.playback_file else None,
        max_live_sec=float(args.max_live_sec) if args.max_live_sec is not None else None,
    )
    app.run()


if __name__ == "__main__":
    main()
