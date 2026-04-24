# Analysis Backend

This folder is the active backend for the final runtime and the reported analyses.

It is intentionally narrower than the older research worktree. The goal on this branch is that `analysis/` only contains code that either:

- powers the hybrid GUI/game runtime
- produces a result cited in `results/`
- or supports those two jobs directly

If you are opening the repo fresh, start at the repo root first. This folder is not the main entry point for the professor-facing story.

## Main Runtime Files

- `run_bci_tracking_game.py`
- `bci_tracking_game.py`
- `bci_game_runtime.py`
- `bci_session_flow.py`
- `hybrid_bci_tester.py`
- `jaw_trigger_rules.py`

These power the guided baseline, hybrid calibration flow, and gameplay.

New runs now default to:

- `../live_runs/all_runs/`

## Main Offline Result Paths

- `realtime_clench_detector.py`
  - jaw artifact training and replay/live jaw evaluation
- `run_eeg_direction_clean_cross_session.py`
  - windowed left/right benchmark
- `lr_event_validation/`
  - event-in-block validation for left/right EEG events
- `lr_event_classifier/`
  - event-level left/right classification
- `lrj_dataset.py`
- `lrj_benchmark.py`
- `run_lrj_offline_benchmark.py`
  - LRJ benchmark family
- `stepwise_protocol_registry.py`
- `stepwise_benchmark.py`
- `run_stepwise_benchmark.py`
  - broad offline benchmark spine
- `review_time_domain_baseline.py`
- `review_spectral_baseline.py`
- `review_csp_baseline.py`
- `review_eegnet_baseline.py`
  - comparison branches that support the failed-iteration story

## Dataset Families

- `LR`
  - block-based left/right EEG sessions
- `jaw`
  - jaw-focused sessions used for the stronger click branch
- `LRJ`
  - structured hybrid sessions with left, right, and jaw events

The `LRJ` family should be treated as related to `LR`, not identical to it. It uses the same general directional-control idea, but in the hybrid guided protocol with the extra jaw event included.

## Runtime Interpretation

The runtime does not do full live retraining.

It uses:

- saved artifacts from `../models/`
- baseline and guided-session calibration
- session-local threshold and decision tuning

That is why the professor-facing write-up describes the system as a hybrid calibrated runtime, not a single universal model.
