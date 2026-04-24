# Analysis Code

This folder holds the minimal analysis scripts that support the reported results.

Current top-level files:

- `run_stepwise_benchmark.py`
- `stepwise_benchmark.py`
- `stepwise_protocol_registry.py`
- `realtime_clench_detector.py`
- `run_eeg_direction_clean_cross_session.py`
- `run_lrj_offline_benchmark.py`
- `lrj_benchmark.py`
- `lrj_dataset.py`
- `review_time_domain_baseline.py`
- `review_spectral_baseline.py`
- `review_csp_baseline.py`
- `review_eegnet_baseline.py`
- `utils.py`

Current subfolders:

- `lr_event_validation/`
  - validation code for Yaniv LR event alignment
- `lr_event_classifier/`
  - event-level left/right feature-comparison support modules

Notes:

- this folder is intentionally narrower than the full `analysis/` directory
- it includes the core scripts needed to regenerate the benchmark summaries, not every exploratory helper from the full repo
