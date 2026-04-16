# Ben LRJ Review

This folder is a one-off audit and trial-count review workflow for Ben's `Ben-LRJ(1-6)-4:9.csv` recording.

This review path stays local to this folder, but the Ben file now also participates in the shared offline LRJ benchmark at `analysis/run_lrj_offline_benchmark.py`.

This wrapper itself is still review-oriented and should not be treated as a live artifact path.

## Dataset Assumptions

- Marker meanings are near-ground-truth but differ from the Yaniv left/right files.
- Markers are interpreted as same-code start/end pairs:
  - `1 = LEFT`
  - `2 = RIGHT`
  - `3 = JAW`
- Each code appears `12` times and reconstructs into `6` intervals.
- Expected trial order is:
  - `LEFT 1, 2, 3, 4, 5, 6`
  - `RIGHT 1, 2, 3, 4, 5, 6`
  - `JAW 1, 2, 3, 4, 5, 6`
- The expected count for each reconstructed interval is its within-label index.

## What This Review Does

- collapses nonzero markers
- pairs consecutive same-code markers into intervals
- validates the expected block structure and pulse counts
- audits channel quality and sampling notes
- computes a lightweight diagnostic peak count inside each interval
- writes a report plus CSV artifacts for manual review

The count overlay is informative only. It does not relabel trials automatically inside the review workflow.

## Run

From the repo root:

```bash
python3 analysis/ben-lrj-review/run_ben_lrj_review.py --csv "../OPENBCI_runs/Ben/Ben-LRJ(1-6)-4:9.csv"
```

Optional flags:

- `--output-dir <path>` to choose a different output directory.
- `--fs 250.0` to keep the fixed reporting rate explicit.
- `--no-plot` to skip the debug PNG.

## Outputs

The script writes to `analysis/ben-lrj-review/outputs/` by default:

- `marker_audit.csv`: one row per collapsed nonzero marker pulse with code, pair index, pair role, and timing.
- `trial_summary.csv`: one row per reconstructed interval with label, expected count, observed count, timing, and issue notes.
- `channel_quality.csv`: channel-level safe/questionable/unsafe summary.
- `summary.json`: machine-readable top-line verdict, issues, warnings, and output paths.
- `summary.md`: human-readable report with structural findings, count findings, and review tables.
- `trial_review.png`: timeline plot with reconstructed trial spans, marker lines, and kept diagnostic peaks.

## Notes

- This folder is Ben-specific on purpose.
- The shared LRJ mechanics now live in `analysis/lrj_review_core.py`, but this wrapper keeps the Ben command, docs, and outputs separate from Yaniv.
- The shared LRJ dataset and offline benchmark now live in `analysis/lrj_dataset.py` and `analysis/run_lrj_offline_benchmark.py`.
- Do not add this file to the shared `analysis/utils.py` dataset families from this workflow.
