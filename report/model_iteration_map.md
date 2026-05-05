# Model And Iteration Map

This file is the shortest answer to:

- where is the model
- what script produced it
- what data it used
- what result file explains it
- whether it was chosen

## 1. Jaw Runtime Model

- Artifact:
  - `models/realtime_clench_model.pkl`
- Main code:
  - `analysis/realtime_clench_detector.py`
  - `analysis/live_jaw_click_detector.py`
  - `analysis/jaw_fast_trigger_sweep.py`
  - `analysis/validate_fast_jaw_model_all_datasets.py`
- Runtime use:
  - `analysis/bci_game_runtime.py`
- Data family:
  - jaw-focused sessions
- Result file:
  - `results/jaw_results.md`
- Status:
  - chosen
  - strongest branch in the project
  - promoted runtime artifact uses a `0.12 s` window, `0.12 s` smoothing, `0.65` click threshold, and `50 ms` cooldown

## 2. Left/Right Windowed Runtime Model

- Artifact:
  - `models/clean_left_right_window_model.pkl`
- Main code:
  - `analysis/run_eeg_direction_clean_cross_session.py`
- Runtime use:
  - `analysis/bci_game_runtime.py`
  - `analysis/bci_session_flow.py`
- Data family:
  - clean `LR` sessions
- Result file:
  - `results/left_right_windowed_results.md`
- Status:
  - kept as the current hand runtime artifact
  - weak and calibration-dependent

## 3. Left/Right Event-Level Baseline

- Main code:
  - `analysis/review_time_domain_baseline.py`
  - `analysis/lr_event_classifier/`
  - `analysis/lr_event_validation/`
- Data family:
  - main `LR` sessions
  - optional related `LRJ` comparison
- Result file:
  - `results/left_right_lr_results.md`
- Status:
  - important benchmark
  - not the direct runtime artifact

## 4. LRJ Hybrid Benchmark

- Main code:
  - `analysis/lrj_dataset.py`
  - `analysis/lrj_benchmark.py`
  - `analysis/run_lrj_offline_benchmark.py`
- Data family:
  - `LRJ`
- Result file:
  - `results/left_right_lrj_results.md`
- Status:
  - supporting hybrid evidence
  - not the main runtime scorecard

## 5. Alternative Left/Right Iterations

- Spectral:
  - code: `analysis/review_spectral_baseline.py`
  - result: `results/spectral_results.md`
  - status: not chosen
- CSP:
  - code: `analysis/review_csp_baseline.py`
  - result: `results/csp_results.md`
  - status: not chosen
- EEGNet:
  - code: `analysis/review_eegnet_baseline.py`
  - result: `results/eegnet_results.md`
  - status: not chosen

## 6. Broad Benchmark Spine

- Main code:
  - `analysis/stepwise_protocol_registry.py`
  - `analysis/stepwise_benchmark.py`
  - `analysis/run_stepwise_benchmark.py`
- Result files:
  - `results/stepwise_benchmark/summary.md`
  - stage subfolders in `results/stepwise_benchmark/`
- Status:
  - broad project-level benchmark summary
  - best place to explain the full hybrid evaluation story
