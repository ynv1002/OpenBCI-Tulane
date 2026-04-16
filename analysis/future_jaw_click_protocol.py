from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.utils import ensure_output_dir
else:
    from .utils import ensure_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a cleaner future jaw click collection plan.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for protocol outputs.",
    )
    return parser.parse_args()


def build_protocol_rows() -> list[dict]:
    rows = [
        {
            "block_order": 1,
            "task_name": "baseline_quiet",
            "marker_start": 0,
            "marker_end": 0,
            "repetitions": 1,
            "trial_duration_sec": 20.0,
            "rest_after_sec": 0.0,
            "experimenter_prompt": "Relax jaw and stay still.",
            "why_it_helps": "Improves inactive calibration and makes quiet false-trigger behavior measurable.",
        },
        {
            "block_order": 2,
            "task_name": "single_click",
            "marker_start": 11,
            "marker_end": 12,
            "repetitions": 20,
            "trial_duration_sec": 2.0,
            "rest_after_sec": 4.0,
            "experimenter_prompt": "One quick clench, then fully relax.",
            "why_it_helps": "Provides the cleanest onset and offset examples for tap-style control.",
        },
        {
            "block_order": 3,
            "task_name": "short_hold_release",
            "marker_start": 21,
            "marker_end": 22,
            "repetitions": 15,
            "trial_duration_sec": 1.5,
            "rest_after_sec": 4.0,
            "experimenter_prompt": "Clench briefly, hold about one second, then release.",
            "why_it_helps": "Strengthens active-hold labels while keeping onset and offset visually clean.",
        },
        {
            "block_order": 4,
            "task_name": "double_click",
            "marker_start": 31,
            "marker_end": 32,
            "repetitions": 15,
            "trial_duration_sec": 2.5,
            "rest_after_sec": 4.5,
            "experimenter_prompt": "Two quick clenches with a clear gap between them.",
            "why_it_helps": "Tests minimum separation and whether the detector splits consecutive taps correctly.",
        },
        {
            "block_order": 5,
            "task_name": "paced_burst_slow",
            "marker_start": 41,
            "marker_end": 42,
            "repetitions": 10,
            "trial_duration_sec": 4.0,
            "rest_after_sec": 5.0,
            "experimenter_prompt": "Repeat clicks at a slow steady tempo, about two clicks per second.",
            "why_it_helps": "Gives labeled burst structure without the ambiguity of unscripted repeated blocks.",
        },
        {
            "block_order": 6,
            "task_name": "paced_burst_fast",
            "marker_start": 51,
            "marker_end": 52,
            "repetitions": 10,
            "trial_duration_sec": 3.0,
            "rest_after_sec": 5.0,
            "experimenter_prompt": "Repeat clicks at a faster tempo, about three clicks per second.",
            "why_it_helps": "Stress-tests cooldown and burst handling for game-like rapid taps.",
        },
        {
            "block_order": 7,
            "task_name": "random_wait_single_click",
            "marker_start": 61,
            "marker_end": 62,
            "repetitions": 15,
            "trial_duration_sec": 2.0,
            "rest_after_sec": 3.0,
            "experimenter_prompt": "Wait for the cue, then make one click after a random delay.",
            "why_it_helps": "Reduces anticipation effects and improves onset realism for future live control.",
        },
    ]
    return rows


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    protocol_df = pd.DataFrame(build_protocol_rows())
    csv_path = output_dir / "future_jaw_click_protocol.csv"
    md_path = output_dir / "future_jaw_click_protocol.md"
    protocol_df.to_csv(csv_path, index=False)

    lines = [
        "# Future Jaw Click Collection Protocol",
        "",
        "This plan is aimed at live click detection rather than coarse hold-vs-repeated blocks.",
        "Marker edges still represent cue timing, not exact physiological onset truth, so the design favors isolated tasks and long inactive gaps.",
        "",
        "## Proposed Blocks",
        "",
        protocol_df.to_markdown(index=False),
        "",
        "## Practical Notes",
        "",
        "- Use a short audio cue or spoken cue at each trial start so the experimenter and participant share timing.",
        "- Keep long inactive gaps after each trial so replay evaluation can distinguish missed clicks from detector chatter.",
        "- Prefer dedicated marker codes per task if the acquisition path supports them.",
        "- If the current marker stack only supports `1-4`, record each task in a separate file so the file identity carries task meaning.",
        "- Log cue timestamps separately if possible; even a simple text log will help compare cue timing to physiological onset later.",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("Future jaw click collection blocks")
    for row in protocol_df.to_dict(orient="records"):
        print(
            f"  {int(row['block_order'])}. {row['task_name']}: "
            f"markers {int(row['marker_start'])}/{int(row['marker_end'])}, "
            f"reps={int(row['repetitions'])}, "
            f"trial={float(row['trial_duration_sec']):.1f}s, rest={float(row['rest_after_sec']):.1f}s"
        )
    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
