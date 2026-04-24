# BCI Project

This branch is the cleaned professor-facing version of the hybrid BCI project.

The project story is:

- we built a hybrid BCI system with `LEFT`, `RIGHT`, `JAW`, and `REST`
- the live flow runs a guided session with baseline, scripted collection, adaptation, and gameplay
- the jaw branch is the stronger branch
- the hand `LEFT/RIGHT` branch is weaker and still session-sensitive
- the current runtime uses saved artifacts plus session calibration/adaptation
- the current runtime does **not** retrain model weights during the GUI session

## Where To Start

If you are orienting to the active project, read in this order:

1. this root README
2. `analysis/README.md`
3. `final_package/README.md`

## Clean Branch Layout

- `final_package/`
  - professor-facing package with curated report, results, code, models, and live examples
- `analysis/`
  - active analysis, runtime, and benchmark code
- `bci_pipeline/`
  - small runtime helper module reused by the analysis-side runtime

## Main Runtime Deliverable

The current GUI/game deliverable is the tracking-game stack:

- `analysis/run_bci_tracking_game.py`
- `analysis/bci_tracking_game.py`
- `analysis/bci_game_runtime.py`
- `analysis/bci_session_flow.py`

This is the app that:

1. prompts for a participant/session
2. runs a `45 second` baseline
3. runs guided `LEFT`, `RIGHT`, `JAW tap`, and `HOLD` collection
4. applies session-local adaptation
5. transitions into the game

New run logs now land under:

- `final_package/live_run_examples/`

## Runtime Model Policy

This is important for interpreting the results correctly.

The current GUI/game runtime does **not** do full retraining during the live session.
Instead, it does:

- load the saved jaw artifact
- load the saved hand artifact
- collect baseline/calibration features
- tune thresholds and runtime decision settings from the guided session

So the honest description is:

- fixed saved models
- plus session calibration
- plus session-local runtime adaptation

not:

- one universal model trained on everything
- or live retraining during the GUI session

## Active Analysis Components

### 1. GUI/Game Runtime

- `analysis/run_bci_tracking_game.py`
- `analysis/bci_tracking_game.py`
- `analysis/bci_game_runtime.py`
- `analysis/bci_session_flow.py`

This is the main professor-facing code path.

### 2. Hybrid Tester Support Layer

- `analysis/hybrid_bci_tester.py`

This is an older support module that still provides runtime pieces reused by the game code.
It is not the main deliverable GUI.

### 3. Jaw Branch

- `analysis/realtime_clench_detector.py`

This is not a GUI.
It is the jaw modeling and runtime module used for:

- offline jaw training
- replay/live jaw testing
- the jaw branch inside the hybrid runtime

### 4. LR Event Validation

- `analysis/lr_event_validation/`

Offline validation path.
It checks whether detected EEG events land inside trusted `LEFT` / `RIGHT` marker blocks.

### 5. Event-Level LEFT/RIGHT Classification

- `analysis/lr_event_classifier/`

Offline modeling path.
It classifies detected events as `LEFT` or `RIGHT` after event extraction.

### 6. Window-Based LEFT/RIGHT Benchmark

- `analysis/run_eeg_direction_clean_cross_session.py`

Offline benchmark path.
It uses windowed EEG features with calibration and cross-session testing.

### 7. LRJ Benchmark

- `analysis/lrj_dataset.py`
- `analysis/lrj_benchmark.py`
- `analysis/run_lrj_offline_benchmark.py`

Offline benchmark path for the structured hybrid protocol family.

### 8. Stepwise Benchmark

- `analysis/stepwise_protocol_registry.py`
- `analysis/stepwise_benchmark.py`
- `analysis/run_stepwise_benchmark.py`

This is the broad offline benchmark spine used for the final results story.

## Dataset Families

The repo should be understood as several related protocol families, not one pooled dataset.

### LR

Yaniv `EEG_LR` sessions:

- `LR-2-27-26-(01).csv`
- `LR-3-15-26-(04).csv`
- `LR-3-8-26-(02).csv`
- `LR-3-8-26-(03).csv`

These are the main EEG `LEFT/RIGHT` block-based sessions.

### HR / EMG_JvsN

Yaniv jaw sessions used for jaw event modeling and replay/live jaw work.

### LRJ

Structured hybrid `LEFT` / `RIGHT` / `JAW` event-count protocol family:

- `Ben-LRJ(1-6)-4:9.csv`
- `Yaniv-LRJ(6)-4:7.csv`

These should **not** be described as random one-offs.
They are clean, structured hybrid protocol data with event/count labels for all three movement classes.
They are the closest current dataset family to the guided hybrid collection logic used by the GUI.

The right way to describe them is:

- a structured LRJ protocol family
- current shared offline benchmark data
- a strong candidate family for future retraining work

### Diagnostic Dalin Files

- `Dalin-LR(6)-4:7.csv`

These are not part of the professor-facing result story.
The only retained Dalin-side code on this branch is the frozen `LR(6)` counter helper that the runtime still reuses internally.
- `Dalin-LRJ(1,2)-4:7.csv`

These stay diagnostic-only and should not be promoted into the main benchmark story.

## Current Model Status

- Jaw is the strongest branch.
  - current best replay weighted event-F1 is about `0.767`
- Event-level `LEFT/RIGHT` remains weak but non-random.
  - expanded LR pooled accuracy / macro-F1 is about `0.613 / 0.613`
- Clean window-based `LEFT/RIGHT` remains weak.
  - pooled macro-F1 is about `0.480`
- LRJ hand exact-count decode is still weak.
- LRJ jaw exact-count decode is better than hand, but still not deployment-grade.

## Recommended Professor-Facing Links

The curated package lives under:

- `final_package/report/`
- `final_package/results/`
- `final_package/gui_game_code/`
- `final_package/live_run_examples/`
- `final_package/analysis_code/`
- `final_package/models/`

## Install

```bash
pip install -r requirements.txt
```

## Run The Main GUI/Game

Replay:

```bash
python3 analysis/run_bci_tracking_game.py --mode REPLAY
```

Live:

```bash
python3 analysis/run_bci_tracking_game.py --mode LIVE --board cyton --serial-port "<your-port>"
```
