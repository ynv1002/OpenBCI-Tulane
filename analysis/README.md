# EEG And Jaw Analysis Guide

This directory is the active center of the hybrid BCI project.

It contains:

- the current GUI/game runtime
- jaw modeling and runtime support
- offline `LEFT/RIGHT` validation and modeling paths
- the LRJ benchmark family
- the stepwise offline benchmark

## The Clean Taxonomy

The easiest way to avoid confusion is to separate this folder into three roles.

### 1. User-Facing Runtime

Main GUI/game:

- `run_bci_tracking_game.py`
- `bci_tracking_game.py`
- `bci_game_runtime.py`
- `bci_session_flow.py`

This is the actual guided session plus game.

### 2. Runtime Support / Prototype Layer

- `hybrid_bci_tester.py`

This is an older support module.
Some runtime pieces are still reused by the game stack, but this is not the main professor-facing GUI.

### 3. Offline Analysis

- `realtime_clench_detector.py`
- `lr_event_validation/`
- `lr_event_classifier/`
- `run_eeg_direction_clean_cross_session.py`
- `run_lrj_offline_benchmark.py`
- `run_stepwise_benchmark.py`

These are offline analysis and benchmarking paths, not GUIs.

## Dataset Families

### LR

Yaniv `EEG_LR` block-based `LEFT` / `RIGHT` sessions.

Best current sessions:

- `LR-2-27-26-(01).csv`
- `LR-3-15-26-(04).csv`

Stress sessions:

- `LR-3-8-26-(02).csv`
- `LR-3-8-26-(03).csv`

### HR / EMG_JvsN

Jaw-focused data family used for jaw modeling and replay/live jaw evaluation.

### LRJ

Structured hybrid protocol family with `LEFT`, `RIGHT`, and `JAW`.

Main files:

- `../OPENBCI_runs/Ben/Ben-LRJ(1-6)-4:9.csv`
- `../OPENBCI_runs/Yaniv/Yaniv-LRJ(6)-4:7.csv`

Important framing:

- these are not just random one-offs
- they are structured hybrid event/count data for all three movements
- they are the closest current data family to the guided hybrid collection logic used by the GUI
- they are current shared offline benchmark data
- they are a strong candidate family for future retraining work

### Diagnostic Dalin Files

The Dalin files are diagnostic-only and are not part of the professor-facing package.
The one exception is the frozen `LR(6)` counter helper that is still reused internally by the current event-counting path.

## Current Runtime Behavior

The runtime does **not** retrain weights during the GUI session.

Current behavior is:

1. load saved jaw and hand artifacts
2. collect baseline/calibration information
3. run guided collection
4. apply session-local runtime adaptation
5. start gameplay

That means:

- jaw branch: saved model plus runtime threshold interpretation
- hand branch: saved model plus feature calibration plus runtime threshold/margin tuning

It does **not** mean:

- one universal model trained on everything
- or full online retraining during the session

## Main Active Paths

### Tracking Game Runtime

Purpose:

- guided hybrid collection
- session adaptation
- gameplay
- logging

Entry point:

```bash
python3 analysis/run_bci_tracking_game.py --mode LIVE --board cyton --serial-port "<your-port>"
```

Logs now default to:

- `../final_package/live_run_examples/`

### Jaw Modeling

Main script:

- `analysis/realtime_clench_detector.py`

Purpose:

- train/export the jaw artifact
- replay jaw sessions
- run live jaw evaluation
- support the jaw branch used by the runtime

This is a modeling/runtime module, not a GUI.

### LR Event Validation

Folder:

- `analysis/lr_event_validation/`

Purpose:

- verify that detected EEG events stay inside trusted marker blocks

### Event-Level LEFT vs RIGHT

Folder:

- `analysis/lr_event_classifier/`

Purpose:

- classify detected EEG events as `LEFT` or `RIGHT`

This is offline analysis.

### Clean Window-Based LEFT vs RIGHT

Main script:

- `analysis/run_eeg_direction_clean_cross_session.py`

Purpose:

- run the window-based benchmark with session calibration

This is offline analysis.

### LRJ Offline Benchmark

Main files:

- `analysis/lrj_dataset.py`
- `analysis/lrj_benchmark.py`
- `analysis/run_lrj_offline_benchmark.py`

Purpose:

- evaluate structured LRJ interval/event/count behavior for `LEFT`, `RIGHT`, and `JAW`

### Stepwise Benchmark

Main files:

- `analysis/stepwise_protocol_registry.py`
- `analysis/stepwise_benchmark.py`
- `analysis/run_stepwise_benchmark.py`

Purpose:

- provide the broad offline benchmark spine for jaw, LR, and LRJ

## Current Decisions To Preserve

- jaw is the stronger branch
- hand `LEFT/RIGHT` remains weaker and session-sensitive
- LR, HR, and LRJ are related but distinct data families
- LRJ should be described as structured hybrid protocol data, not as random one-offs
- do not add last-minute live retraining to the GUI path for the professor-facing branch
- if we retrain next, LRJ/guided-session family data should be the basis of that next offline retraining cycle
