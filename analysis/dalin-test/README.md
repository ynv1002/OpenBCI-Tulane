# LRJ Recovery Debugging Toolkit

This folder is a one-off recovery and debugging workflow for the corrupted `Dalin-LRJ(1,2)-4:7.csv` recording only.

It reuses the existing offline jaw replay artifact at `analysis/outputs/realtime_clench_model.pkl` to help audit markers and suggest a conservative partial reconstruction. It is intentionally not a general framework, and `LR(6)` is out of scope here because it is already considered usable.

## Dataset Status

`Dalin-LRJ(1,2)-4:7.csv` should be kept as a debugging artifact, not treated as a clean LRJ training dataset or a core labeled result.

The scripts and outputs in this folder are for audit and review only. They are useful for understanding what went wrong and for spot-checking a few possible examples, but they should not be treated as a path to canonical ground-truth recovery. Commands such as `--part1` and `--pairs` remain inspection tools, not label-restoration tools.

## Why LRJ Is Low-Trust

- The recording continued well past the real task end. The file has 46 collapsed markers with even parity, the last marker is at `284.084 s`, and the replay trace continues to about `762.2 s`, leaving about `478.116 s` of post-last-marker tail that should not be treated as task-structured data.
- Headset or electrode contact likely changed during or after the intended run. That makes the later signal distribution less comparable to the earlier part of the recording and weakens confidence in model-guided interpretation later in the file.
- Marker values are unreliable even if marker timing still has some value. The most trustworthy remnants are marker timing, start/stop pairing, and rough early-run order, not the raw marker code itself.
- The current marker-pair review is already weak as evidence for a reusable LRJ dataset: only 2 of 18 duration-filtered trial windows are marked `likely jaw`.

## Assumptions

- Reconstruction timing uses a fixed `250 Hz` sample rate.
- Marker values are treated only as event timestamps.
- Any nonzero marker is treated as "a marker happened here"; the numeric code is preserved in the audit output but not trusted as ground truth.
- Jaw suggestions come from the reused model and trigger stack.
- Non-jaw suggestions are only made when timing and phase logic support them conservatively.

## Run

From the repo root:

```bash
python analysis/dalin-test/run_lrj_recovery.py --csv "/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Dalin/Dalin-LRJ(1,2)-4:7.csv"
```

If your shell only exposes `python3`, the same script works with `python3`.

Optional flags:

- `--artifact <path>` to override the saved jaw artifact.
- `--output-dir <path>` to choose a different output directory.
- `--fs 250.0` to keep the fixed reconstruction sample rate explicit.
- `--cluster-gap-sec 3.0` to tweak small-gap marker clustering.
- `--part1` to add a Part 1-only cluster interpretation through the last marker.
- `--pairs` to rebuild the LRJ review around consecutive collapsed marker pairs as trial windows.
- `--no-plot` to skip the debug PNG.

## Outputs

The script writes to `analysis/dalin-test/outputs/` by default:

- `marker_audit.csv`: one row per collapsed nonzero marker event.
- `score_trace.csv`: one row per offline replay window with model probabilities and trigger decisions.
- `jaw_events.csv`: one row per conservative jaw peak candidate derived from the clench-probability trace.
- `cluster_summary.csv`: one row per marker cluster with conservative reconstruction suggestions.
- `lrj_debug_plot.png`: simple marker and jaw-probability debug plot, when `matplotlib` is available.

When `--part1` is used, the script also writes:

- `part1_cluster_interpretation.csv`: one row per cluster through the last marker only, with conservative real-vs-noise, jaw-vs-nonjaw, clench-count, and optional LRJ sequence hints.
- `part1_cluster_review.png`: Part 1 review plot with marker timestamps, cluster spans, clench probability, and Part 1 jaw peaks.

Part 1 interpretation clips all evidence at the last marker time and ignores the long post-marker tail for cluster interpretation.

When `--pairs` is used, the script also writes:

- `trial_summary.csv`: one row per duration-filtered trial window built from consecutive collapsed marker pairs, with the raw pair type preserved as metadata (`1→2`, `2→3`, `1→1`, etc.), plus clench-probability peaks and a conservative jaw/non-jaw label.
- `trial_review.png`: simple trial-window plot with marker timestamps, clench probability, and detected trial peaks.

## How To Use This Folder Going Forward

- Keep LRJ as a debugging artifact for protocol mistakes and postmortem analysis.
- Use it to document why marker timing ended up more trustworthy than marker value in this run.
- Optionally hand-recover a few examples if a specific debugging question really needs them.
- Do not rely on LRJ as a main result, a primary training source, or a clean benchmark for LRJ performance.
- Put future effort into the cleaner left/right data, cleaner event structure, and live-control integration instead of spending more time trying to perfect LRJ reconstruction.

The original dataset is never modified.
