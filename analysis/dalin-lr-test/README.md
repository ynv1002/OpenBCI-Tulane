# Dalin LR(6) Trial/Event Review

This folder is a one-off analysis path for the clean `Dalin-LR(6)-4:7.csv` recording only.

Unlike the low-trust LRJ file, this dataset has trusted marker meanings and a known trial structure:

- 12 total trials
- first 6 trials are `LEFT`
- next 6 trials are `RIGHT`
- expected click/clench counts per hand are `1, 2, 3, 4, 5, 6`
- marker meanings are trusted:
  - `1 = left start`
  - `2 = left stop`
  - `3 = right start`
  - `4 = right stop`

The goal here is to treat trusted marker pairs as ground-truth trial boundaries, estimate event counts inside each trial, and compare those counts with the known expected count ramp.

## Assumptions

- Reporting and event timing use a fixed `250 Hz` sample rate.
- Marker pairs are trusted and should define the 12 trial windows exactly.
- Primary event counting uses preprocessed left/right EEG plus aggregate RMS peak detection.
- Auxiliary left/right model outputs are diagnostic only and do not override marker-derived trial hand labels.

## Run

From the repo root:

```bash
python analysis/dalin-lr-test/run_lr6_review.py --csv "/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Dalin/Dalin-LR(6)-4:7.csv"
```

If your shell only exposes `python3`, the same script works with `python3`.

Optional flags:

- `--output-dir <path>` to choose a different output directory.
- `--fs 250.0` to keep the fixed reporting rate explicit.
- `--no-plot` to skip the debug PNG.

## Outputs

The script writes to `analysis/dalin-lr-test/outputs/` by default:

- `trial_summary.csv`: one row per trusted trial with hand, expected count, estimated count, match flag, detected event times, and auxiliary model diagnostics.
- `event_candidates.csv`: one row per detected event candidate, labeled as `LEFT_CLICK` or `RIGHT_CLICK` from the trusted trial hand, plus `kept_for_count` and `filter_reason` so dropped edge peaks remain auditable.
- `trial_review.png`: simple debug plot with full-run trial spans, smoothed aggregate RMS, and detected event peaks.

## Notes

- This folder is a clean-count benchmark, not a rescue/reconstruction workflow.
- The current cross-subject left/right model is included only as a comparison signal; it is expected to be much less trustworthy than the marker-derived trial hand labels on this Dalin file.
- The current conservative tuning adds only one heuristic: peaks earlier than `0.10 s` after trial start are ignored for counting. This is intended to suppress obvious onset-edge artifacts without broad retuning.
- With that small boundary rule, the current review lands at `9/12` exact matches. Trials `3`, `4`, and `8` remain under-counted and are intentionally left unresolved in this pass to avoid overfitting this one file.
- The auxiliary LR model transfer remains poor and unchanged by this tuning, so it stays diagnostic-only.
- The source CSV is never modified.
