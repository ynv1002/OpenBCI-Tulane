# Yaniv EEG_LR Event Validation

This folder applies the frozen `LR(6)` event counter to Yaniv's cleaner EEG left/right recordings as a transfer and validation pass.

The question here is deliberately narrow:

- do detected events mostly fall inside trusted marker-defined `LEFT` / `RIGHT` blocks?

This is not a side-classifier yet. Markers define the ground-truth block boundaries.

## Data Role

Input folder:

- `/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/EEG_LR`

Trusted marker mapping:

- `1 = LEFT start`
- `2 = LEFT stop`
- `3 = RIGHT start`
- `4 = RIGHT stop`

Practical trust split:

- strongest runs: `LR-2-27-26-(01).csv`, `LR-3-15-26-(04).csv`
- noisier robustness runs: `LR-3-8-26-(02).csv`, `LR-3-8-26-(03).csv`

## Frozen Counter Settings

These settings are intentionally copied from the tuned Dalin `LR(6)` review and should stay frozen unless a prompt explicitly asks to retune them:

- smoothing window: `75` samples
- minimum peak distance: `0.75 s`
- prominence rule: `max(0.6 * trial_smoothed_std, 0.02 uV)`
- early-boundary exclusion: peaks before `0.10 s`

## Current Result

Aggregate summary:

- `analysis/lr_event_validation/outputs/overall_summary.csv`

Current overall alignment:

- `708` kept events inside valid marker-defined blocks
- `94` kept events outside all valid blocks
- inside-block fraction `0.8828` (`88.3%`)

Per-file readout from the current summary:

- `LR-2-27-26-(01).csv`: `143 / 150` inside (`95.3%`)
- `LR-3-15-26-(04).csv`: `232 / 240` inside (`96.7%`)
- `LR-3-8-26-(02).csv`: `236 / 295` inside (`80.0%`)
- `LR-3-8-26-(03).csv`: `97 / 117` inside (`82.9%`)

Interpretation:

- the event structure is usable for later event-centered experiments
- Feb 27 and Mar 15 are the best source runs for clean follow-up work
- the March 8 runs stay useful, but mainly for robustness or later stress testing

## Run

From the repo root:

```bash
python3 analysis/lr_event_validation/run_yaniv_lr_validation.py --input-dir "/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/EEG_LR"
```

## Outputs

Per file under `analysis/lr_event_validation/outputs/<file-stem>/` when regenerated:

- `block_summary.csv`
- `event_table.csv`
- `<file-stem>_validation.png`
- `run_notes.json`

Aggregate outputs under `analysis/lr_event_validation/outputs/` when regenerated:

- `all_block_summary.csv`
- `all_event_table.csv`
- `overall_summary.csv`

Important event-table fields:

- whether the event was kept for counting
- whether it fell inside a valid marker-defined block
- assigned block id and side when applicable
- frozen-counter prominence metadata

## How This Feeds The Rest Of The Project

This folder is the bridge between marker-trusted Yaniv recordings and later modeling:

- `final_package/analysis_code/lr_event_classifier/` reuses these in-block kept events as labeled `LEFT` / `RIGHT` examples
- `final_package/gui_game_code/hybrid_bci_tester.py` reuses the same frozen event-counter logic for `LEFT` / `RIGHT` gating

If a future prompt is about event alignment or event extraction on Yaniv EEG_LR, start here before changing the classifier or the hybrid GUI.
