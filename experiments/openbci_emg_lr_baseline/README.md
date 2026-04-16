# OpenBCI EMG Left/Right Baseline

Legacy note: this folder documents an older EMG-only baseline around `Y_EMG.csv`. It is still useful for historical comparison, but it is not the active path for the current EEG/jaw hybrid control work. For the current project direction, start with `README.md` and `analysis/README.md`.

This experiment is a first-pass offline baseline that applies OpenBCI-style EMG joystick logic to the existing `Y_EMG.csv` dataset.

It is intentionally narrow:

- Only `1D` left-right control is implemented.
- `X-` uses the left-side channels `ch1` and `ch4`.
- `X+` uses the right-side channels `ch3` and `ch6`.
- `Jaws` is treated as a strong non-joystick / near-center candidate rather than as a left-right target class.

## Why This Dataset Is Only A Partial Fit

The current data comes from an older hold-based protocol with long sustained `Jaws`, `Jleft`, and `Jright` labels.

That is useful for a baseline, but it is not the ideal regime for joystick logic. OpenBCI-style EMG joystick control usually benefits from clearer dynamic directional motions rather than long sustained holds. This experiment preserves that framing on purpose: it is a clean transfer baseline before any redesign of data collection.

## Channel Mapping

- Left side / `X-`: `ch1`, `ch4`
- Right side / `X+`: `ch3`, `ch6`
- Aggregations compared:
  - mean of normalized left channels vs mean of normalized right channels
  - max of normalized left channels vs max of normalized right channels

For each aggregation method:

- `x_raw = right_aggregate - left_aggregate`
- `x_smooth = prev_x + (1 - smoothing) * (x_raw - prev_x)`
- `total_activation = left_aggregate + right_aggregate`
- `abs_direction = abs(x_smooth)`

## OpenBCI Parameters

Defaults match the requested OpenBCI-style baseline:

- Window: `1.0 s`
- `uvLimit`: `200 uV`
- `creepIncreasing`: `0.9`
- `creepDecreasing`: `0.99999`
- `minimumDeltaUV`: `10`
- `lowerThresholdMinimum`: `6`
- Joystick smoothing: `0.9`

Initial thresholds are set to:

- `lowerThreshold = 6`
- `upperThreshold = 16`

That keeps the processor stable at startup while preserving the requested adaptive update rules.

## Scale And Sample-Rate Assumptions

The experiment inspects the raw CSV before processing and writes the decision into `outputs/summary_report.md`.

For the provided `Y_EMG.csv`, the tracked channels appear to be large-count OpenBCI-style values with a strong DC offset rather than ready-to-use microvolts. To make the OpenBCI threshold logic meaningful, the experiment may:

- center each tracked channel by its median
- convert counts to microvolts with the OpenBCI Cyton factor `0.02235 uV/count`

This is not hidden. The applied mode, offsets, and transformed ranges are all written to the outputs.

The default sample rate is `250 Hz` because the existing repo already uses `250.0 Hz` throughout its OpenBCI processing scripts. If that assumption is wrong for this file, override it on the command line.

## How To Run

From the repo root:

```bash
python3 experiments/openbci_emg_lr_baseline/run_experiment.py
```

With explicit paths:

```bash
python3 experiments/openbci_emg_lr_baseline/run_experiment.py \
  --csv "/Users/yanivnaggar/Desktop/Fall 2025/Independent Study/CogniSync-main/Clench_1/Y_EMG.csv" \
  --output-dir experiments/openbci_emg_lr_baseline/outputs \
  --fs 250
```

## Outputs

The script writes the following files under `outputs/`:

- `processed_timeseries.csv`
- `summary_metrics.json`
- `summary_report.md`
- `label_summary_stats.csv`
- `mean_direction_threshold_scan.csv`
- `max_direction_threshold_scan.csv`
- `mean_activation_threshold_scan.csv`
- `max_activation_threshold_scan.csv`
- `mean_jleft_jright_confusion.csv`
- `max_jleft_jright_confusion.csv`
- `mean_norm_active_confusion.csv`
- `max_norm_active_confusion.csv`
- plot PNGs for raw channels, normalized channels, thresholds, joystick traces, histograms, scatters, and threshold scans

## What To Inspect First

1. `outputs/summary_report.md`
2. `outputs/x_smooth_histograms.png`
3. `outputs/activation_direction_scatter.png`
4. `outputs/x_raw_x_smooth_window.png`
5. `outputs/tracked_channel_thresholds_window.png`
