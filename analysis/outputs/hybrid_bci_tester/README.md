# Hybrid BCI Tester Logs

This folder stores structured outputs from `analysis/run_hybrid_bci_tester.py`.

Each run creates a timestamped subfolder such as:

- `20260408_102508_replay_LR-2-27-26-_01_`
- `20260408_102414_live_fixed`

## Typical Files Per Run

- `run_metadata.json`
  - mode, config, source file or live settings, and other run-level metadata
- `detections.csv`
  - discrete detections fired by the tester
- `confidence_trace.csv`
  - continuous confidence/state trace over time
- `ground_truth_segments.csv`
  - replay ground truth or live prompt-ground-truth segments when available
- `trial_log.csv`
  - live prompt/trial summary rows; in replay mode this may be empty or only carry headers

## How To Read These Logs

- replay runs are for visual/debug validation against trusted markers
- live runs are for prompted testing, success/failure logging, and later latency review
- the most useful first files are usually `run_metadata.json`, `detections.csv`, and `confidence_trace.csv`

This folder is generated output, not source code. For behavior changes, start with:

- `analysis/hybrid_bci_tester.py`
- `analysis/run_hybrid_bci_tester.py`
- `analysis/README.md`
