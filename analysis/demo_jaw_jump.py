from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

DEFAULT_MPLCONFIGDIR = Path(__file__).resolve().parent / "outputs" / ".mplconfig"
DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.utils import ensure_output_dir
else:
    from .utils import ensure_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Flappy-style replay demo driven by jaw click events.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory containing replay outputs.",
    )
    parser.add_argument("--session", type=str, default="HR-3-15-26-(03).csv", help="Jaw session to replay.")
    parser.add_argument("--play", action="store_true", help="Print an ASCII replay in the terminal.")
    parser.add_argument("--speed", type=float, default=4.0, help="ASCII replay speed multiplier.")
    parser.add_argument("--jump-velocity", type=float, default=1.15, help="Velocity applied on each click.")
    parser.add_argument("--gravity", type=float, default=-1.85, help="Downward acceleration.")
    return parser.parse_args()


def _session_tag(session_name: str) -> str:
    return session_name.replace(".csv", "").replace("(", "").replace(")", "").replace("-", "_")


def _render_ascii(height: float, width: int = 30) -> str:
    position = int(round(np.clip(height, 0.0, 1.0) * (width - 1)))
    cells = [" "] * width
    cells[position] = ">"
    return "|" + "".join(cells) + "|"


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    best_summary = json.loads((output_dir / "jaw_click_best_strategy.json").read_text(encoding="utf-8"))
    strategy_name = best_summary["best_strategy"]["strategy_name"]

    score_df = pd.read_csv(output_dir / "jaw_click_replay_scores.csv")
    click_df = pd.read_csv(output_dir / f"jaw_click_events_{strategy_name}.csv")
    session_scores = score_df[score_df["filename"] == args.session].sort_values("center_time_sec").reset_index(drop=True)
    session_clicks = click_df[click_df["filename"] == args.session].sort_values("center_time_sec").reset_index(drop=True)
    if session_scores.empty:
        raise SystemExit(f"No replay score rows found for {args.session}")

    click_times = set(np.round(session_clicks["center_time_sec"].to_numpy(dtype=float), 6))
    dt = float(np.median(np.diff(session_scores["center_time_sec"].to_numpy(dtype=float))))
    height = 0.50
    velocity = 0.00
    total_clicks = 0
    rows = []

    for _, row in session_scores.iterrows():
        current_time = float(row["center_time_sec"])
        did_click = np.round(current_time, 6) in click_times
        if did_click:
            velocity = float(args.jump_velocity)
            total_clicks += 1
        velocity += float(args.gravity) * dt
        height = float(np.clip(height + velocity * dt, 0.05, 0.95))
        if height <= 0.05 and velocity < 0.0:
            velocity = 0.0
        if height >= 0.95 and velocity > 0.0:
            velocity = 0.0
        rows.append(
            {
                "filename": args.session,
                "time_sec": current_time,
                "height": height,
                "velocity": velocity,
                "click_emitted": did_click,
                "clench_probability": float(row["prob_binary_CLENCH"]),
                "onset_probability": float(row["prob_4state_ONSET"]),
            }
        )
        if args.play:
            gauge = _render_ascii(height)
            marker = " CLICK" if did_click else ""
            print(
                f"\rt={current_time:7.2f}s {gauge} height={height:0.2f} vel={velocity:0.2f}{marker}",
                end="",
                flush=True,
            )
            time.sleep(max(0.0, dt / max(args.speed, 0.1)))

    if args.play:
        print()

    sim_df = pd.DataFrame(rows)
    session_tag = _session_tag(args.session)
    csv_path = output_dir / f"jaw_jump_demo_{strategy_name}_{session_tag}.csv"
    png_path = output_dir / f"jaw_jump_demo_{strategy_name}_{session_tag}.png"
    sim_df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax.plot(sim_df["time_sec"], sim_df["height"], color="#1f78b4", linewidth=1.6, label="bird_height")
    for _, row in sim_df[sim_df["click_emitted"]].iterrows():
        ax.axvline(float(row["time_sec"]), color="#ff7f00", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Jaw Jump Demo | {args.session} | {strategy_name}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Height")
    ax.legend(loc="upper right")
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    print(f"Strategy: {strategy_name}")
    print(f"Session: {args.session}")
    print(f"Clicks used: {total_clicks}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
