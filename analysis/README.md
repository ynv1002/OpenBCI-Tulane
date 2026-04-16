# EEG and Jaw Analysis Guide

This directory is the active center of the project.

It contains:

- offline audits and dataset reviews
- jaw click model training and replay/live testing
- EEG left/right window and event experiments
- the stage-gated stepwise jaw + EEG benchmark
- one-off Dalin recovery and benchmarking folders
- the current hybrid BCI tester GUI

If a new Codex session is working on the active BCI path, this is the first local README to read after the root README.

## Current Decisions You Should Preserve

- Use Yaniv `LR-2-27-26-(01).csv` and `LR-3-15-26-(04).csv` as the cleanest EEG left/right sessions.
- Treat Yaniv March 8 EEG_LR runs as lower-consistency robustness data, not first-pass clean training data.
- Treat Dalin `LR(6)` as a useful event-count benchmark.
- Treat Dalin LRJ as a frozen debugging artifact, not a clean labeled dataset.
- Treat Ben `Ben-LRJ(1-6)-4:9.csv` and Yaniv `Yaniv-LRJ(6)-4:7.csv` as the shared offline LRJ benchmark track.
- Treat the manual file-contract registry in `analysis/stepwise_protocol_registry.py` as the source of truth for the new stepwise benchmark.
- Reuse `analysis/lrj_dataset.py` as the shared LRJ source of truth for both the offline benchmark and the subject-specific review wrappers.
- Keep LRJ separate from `analysis.utils.FAMILY_SPECS`; it is a shared benchmark track, not a legacy family-spec dataset.
- Reuse the jaw artifact when possible; do not retrain it by default unless the task explicitly calls for retraining.
- Treat the clean cross-session window `LEFT` vs `RIGHT` artifact as a benchmark, not a strong production direction model.

## Dataset Families

### Yaniv EEG_LR

Folder:

- `/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/EEG_LR`

Marker semantics:

- `1 -> 2`: `LEFT`
- `3 -> 4`: `RIGHT`

High-trust sessions:

- `LR-2-27-26-(01).csv`
- `LR-3-15-26-(04).csv`

Lower-consistency sessions:

- `LR-3-8-26-(02).csv`
- `LR-3-8-26-(03).csv`

### Yaniv EMG_JvsN

Main family for jaw click work.

This supports:

- jaw replay evaluation
- the live/replay jaw detector
- the jaw component inside the hybrid tester

### Dalin one-offs

- `analysis/dalin-lr-test/`
  - clean `LR(6)` benchmark for event counting
- `analysis/dalin-test/`
  - low-trust LRJ debugging/recovery toolkit

### Ben and Yaniv LRJ one-offs

Files:

- `../OPENBCI_runs/Ben/Ben-LRJ(1-6)-4:9.csv`
- `../OPENBCI_runs/Yaniv/Yaniv-LRJ(6)-4:7.csv`

Semantics:

- `1 = LEFT`
- `2 = RIGHT`
- `3 = JAW`
- each label uses six same-code start/end pairs
- expected within-label counts are `1, 2, 3, 4, 5, 6`

Status:

- shared offline benchmark track
- not part of the shared `analysis.utils` dataset families
- not a default live artifact path

## Shared Signal Conventions

### Sampling

- use `250 Hz` as the authoritative reporting rate for the one-off OpenBCI analyses in this repo unless a script says otherwise

### Preprocessing

Windowed EEG work reuses the repo-native path in `analysis.utils`:

1. subtract per-session channel median
2. scale by `0.02235 uV/count`
3. apply family-specific filtering

Family filters:

- `left_right`: `60 Hz` notch, `1-40 Hz` bandpass
- `jaw`: `60 Hz` notch, `20-100 Hz` bandpass

### Calibration

Clean cross-session EEG direction retrains use:

- first `45 s` of each session
- baseline/rest windows only
- session-local normalization

### Channel Quality

Channel status remains intentionally simple:

- `safe`
- `questionable`
- `unsafe`

Training and counting paths generally exclude `unsafe` channels and keep the rest.

## Current Active Paths

### 1. Jaw Click Detection

Main script:

- `analysis/realtime_clench_detector.py`

Primary artifact:

- `analysis/outputs/realtime_clench_model.pkl`

Best current replay summary:

- `analysis/outputs/jaw_click_best_strategy.md`

Current best replay strategy:

- `binary_clench_threshold`
- test weighted event-F1 about `0.767`

Useful commands:

```bash
python3 -m analysis.realtime_clench_detector train
python3 -m analysis.realtime_clench_detector replay --csv <jaw_csv>
python3 -m analysis.realtime_clench_detector replay --visualize --csv <jaw_csv>
python3 -m analysis.realtime_clench_detector live --serial-port /dev/cu.usbserial-XXXX
```

### 1A. Stepwise Jaw + EEG Benchmark

Main files:

- `analysis/stepwise_protocol_registry.py`
- `analysis/stepwise_benchmark.py`
- `analysis/run_stepwise_benchmark.py`

Purpose:

- run a slow, stage-gated offline benchmark across explicit Ben, Yaniv, and diagnostic file contracts
- keep `jaw`, `EEG_LR`, and `LRJ` linked through shared event/count logic without collapsing them into one pooled dataset too early
- report per-stage gates, file-level diagnostics, and LRJ count-constrained decode summaries

Run:

```bash
python3 analysis/run_stepwise_benchmark.py
```

Key outputs:

- `analysis/outputs/stepwise_benchmark/summary.md`
- `analysis/outputs/stepwise_benchmark/protocol_registry.csv`
- `analysis/outputs/stepwise_benchmark/jaw_stage1/`
- `analysis/outputs/stepwise_benchmark/jaw_stage2/`
- `analysis/outputs/stepwise_benchmark/jaw_stage3_lrj/`
- `analysis/outputs/stepwise_benchmark/hand_lrj_decode/`
- `analysis/outputs/stepwise_benchmark/eeg_stage12_main/`
- `analysis/outputs/stepwise_benchmark/eeg_stage12_stress/`
- `analysis/outputs/stepwise_benchmark/eeg_stage3/`

### 2. Frozen EEG Event Counter

This came from the Dalin `LR(6)` review and is now reused as a stable event detector.

Frozen settings:

- smoothing window: `75` samples
- minimum peak distance: `0.75 s`
- prominence rule: `max(0.6 * trial_smoothed_std, 0.02 uV)`
- early-boundary exclusion: peaks before `0.10 s`

Primary reuse path:

- `analysis/lr_event_validation/`

### 3. Yaniv EEG_LR Event Validation

Folder:

- `analysis/lr_event_validation/`

Purpose:

- apply the frozen event counter to all Yaniv EEG_LR files
- measure whether detected events mostly stay inside trusted marker-defined blocks

Current summary:

- `analysis/lr_event_validation/outputs/overall_summary.csv`
- overall kept-event alignment is `708 / 802 = 88.3%` inside trusted blocks
- Feb 27 and Mar 15 are the strongest sessions for later modeling

Run:

```bash
python3 analysis/lr_event_validation/run_yaniv_lr_validation.py --input-dir "/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/EEG_LR"
```

### 4. Event-Level LEFT vs RIGHT Experiments

Folder:

- `analysis/lr_event_classifier/`

Purpose:

- turn kept in-block events from the two high-trust Yaniv runs into labeled `LEFT` / `RIGHT` examples
- test whether direction is learnable at the event level

Current results:

- baseline summary: `analysis/lr_event_classifier/outputs/summary.md`
- second pass summary: `analysis/lr_event_classifier/outputs/second_pass/feature_set_comparison_summary.md`
- best first-pass pooled result is Logistic Regression at about `0.525` accuracy / `0.524` macro-F1
- mu/beta plus asymmetry did not materially improve the baseline

This is still a diagnostic branch, not a strong deployable direction solution.

### 5. Clean Window-Based LEFT vs RIGHT Benchmark

Main script:

- `analysis/run_eeg_direction_clean_cross_session.py`

Artifact:

- `analysis/outputs/clean_left_right_window_model.pkl`

Summary:

- `analysis/outputs/eeg_direction_clean_cross_session.md`

Setup:

- only the two high-trust Yaniv runs
- `1.0 s` windows
- `0.5` overlap
- `combined` features with asymmetry
- session-local `45 s` calibration

Current outcome:

- winning model is Logistic Regression
- pooled accuracy about `0.484`
- pooled macro-F1 about `0.480`
- keep it as a clean benchmark artifact, not a strong direction stage

### 6. Hybrid BCI Tester

Main files:

- `analysis/hybrid_bci_tester.py`
- `analysis/run_hybrid_bci_tester.py`

Purpose:

- extend the earlier jaw visualizer into a multi-class testing GUI
- support `LEFT`, `RIGHT`, `JAW`, and `REST`
- run in live mode and replay mode
- log detections, prompts, confidence traces, and ground-truth segments

Current integration:

- jaw detector uses the existing jaw artifact
- `LEFT` / `RIGHT` require event detection plus direction-model agreement
- replay mode is intended for the two high-trust Yaniv sessions

Logs:

- `analysis/outputs/hybrid_bci_tester/<timestamp>_*`

Replay example:

```bash
python3 analysis/run_hybrid_bci_tester.py --mode REPLAY --replay-csv "/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/EEG_LR/LR-2-27-26-(01).csv"
```

Live example:

```bash
python3 analysis/run_hybrid_bci_tester.py --mode LIVE --board cyton --serial-port "<your-port>"
```

Important limitation:

- replay mode has been smoke-tested end to end
- live mode has only been smoke-tested here with BrainFlow synthetic input, not a real Cyton board in this environment

### 7. Older Direction and Gate Experiments

These scripts still matter for comparison, but they are not the current clean default path:

- `run_active_vs_rest_*`
- `run_eeg_direction_base_up_calibrated.py`
- `run_eeg_direction_cross_session.py`
- `run_left_right_spectral_experiments.py`
- `run_two_stage_lr_pipeline.py`
- `run_two_stage_lr_pipeline_1s_lr_model.py`
- `run_two_stage_lr_window_size_experiment.py`

Why they still matter:

- they contain the older window-based direction baselines
- they show prior two-stage `ACTIVE -> LEFT/RIGHT` behavior
- they are useful comparison points when deciding whether a new clean retrain is actually better

Why they are not the default:

- many of the older summaries mix cleaner and noisier sessions
- the stronger window results were often within-session or otherwise less reusable

### 8. Dalin One-Off Folders

- `analysis/dalin-lr-test/`
  - trusted-trial count benchmark
  - source of the frozen event-counter tuning
- `analysis/dalin-test/`
  - LRJ debugging artifact
  - keep for postmortem reasoning, not clean training

### 9. Shared LRJ Offline Benchmark

Shared benchmark files:

- `analysis/lrj_dataset.py`
- `analysis/lrj_benchmark.py`
- `analysis/run_lrj_offline_benchmark.py`

Purpose:

- merge Ben and Yaniv LRJ into one shared offline benchmark track
- preserve both movement label and expected count `1..6`
- report movement, count, and joint `(label, count)` transfer quality across the two files

Command:

```bash
python3 analysis/run_lrj_offline_benchmark.py
```

Key outputs:

- `analysis/outputs/lrj_interval_dataset.csv`
- `analysis/outputs/lrj_window_features.csv`
- `analysis/outputs/lrj_benchmark_results.csv`
- `analysis/outputs/lrj_benchmark_summary.md`

Important limitation:

- this benchmark track is offline-only in the current phase
- it does not change the saved jaw artifact, the saved EEG direction artifact, or the hybrid tester defaults

### 10. Subject LRJ Review Wrappers

Shared core:

- `analysis/lrj_review_core.py`

Wrappers:

- `analysis/ben-lrj-review/`
- `analysis/yaniv-lrj-review/`

Purpose:

- audit same-code LRJ marker structure
- reconstruct `LEFT`, `RIGHT`, and `JAW` intervals
- compare expected count ramps `1..6` with a diagnostic peak-count overlay

Commands:

```bash
python3 analysis/ben-lrj-review/run_ben_lrj_review.py --csv "../OPENBCI_runs/Ben/Ben-LRJ(1-6)-4:9.csv"
python3 analysis/yaniv-lrj-review/run_yaniv_lrj_review.py --csv "../OPENBCI_runs/Yaniv/Yaniv-LRJ(6)-4:7.csv"
```

Important limitation:

- these wrappers remain review paths even though the files also feed the shared LRJ benchmark
- count mismatches are diagnostic only and should not relabel trials automatically

## Key Output Areas

Main shared outputs live in `analysis/outputs/`.

Especially important files:

- `realtime_clench_model.pkl`
- `clean_left_right_window_model.pkl`
- `lrj_benchmark_summary.md`
- `lrj_interval_dataset.csv`
- `jaw_click_best_strategy.md`
- `eeg_direction_clean_cross_session.md`
- `active_vs_rest_summary_feb27_mar15.md`
- `two_stage_lr_pipeline_1s_lr_model_summary.md`

Subproject outputs stay local to their folders:

- `analysis/outputs/stepwise_benchmark/`
- `analysis/lr_event_validation/outputs/`
- `analysis/lr_event_classifier/outputs/`
- `analysis/dalin-lr-test/outputs/`
- `analysis/dalin-test/outputs/`
- `analysis/ben-lrj-review/outputs/`
- `analysis/yaniv-lrj-review/outputs/`

## Fresh-Session Advice

If a future prompt is ambiguous, start by answering these questions before editing code:

1. Is this task about jaw, EEG direction, event detection, or the hybrid tester?
2. Is the data high-trust Yaniv, lower-trust Yaniv, Ben LRJ, Yaniv LRJ, clean Dalin LR(6), or low-trust Dalin LRJ?
3. Is the user asking for a benchmark/diagnostic artifact or something intended for live reuse?

Those answers usually determine the correct script, dataset, and artifact family immediately.
