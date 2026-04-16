# OpenBCI EMG Processing Audit

Legacy note: this audit belongs to the older EMG-only `Y_EMG.csv` workflow. It is still a useful record of that preprocessing work, but it is not the main path for the current EEG/jaw BCI project. For the active stack, start with `analysis/README.md`.

This experiment exists because the original `experiments/openbci_emg_lr_baseline/` results were weak, and that weakness may have come from preprocessing assumptions rather than from the OpenBCI joystick logic alone.

The audit keeps the baseline intact and asks a narrower question: are the inputs being prepared in a way that the OpenBCI EMG logic can reasonably use?

## Modes

The tracked channels are `ch1`, `ch4`, `ch3`, and `ch6`, using the same left/right mapping as the baseline.

The audit compares these processing modes:

- `raw`: use the loaded values exactly as they appear in the CSV
- `centered`: subtract the per-channel median
- `scaled`: subtract the per-channel median, then multiply by `0.02235 uV/count`
- `filtered`: subtract the per-channel median, scale by `0.02235 uV/count`, then apply an EMG-focused filter stage

The default filtered stage is:

- `60 Hz` notch
- Butterworth bandpass `20-100 Hz`

If `scipy` is unavailable, the audit falls back to a documented rolling-detrend approximation instead of failing silently.

## Why This Audit Exists

OpenBCI’s EMG logic expects filtered data. Your CSV appears to contain raw or raw-like OpenBCI values with large DC offsets, so this audit checks whether:

- the values look like ADS1299-style counts
- the `0.02235 uV/count` assumption is plausible
- the channels carry large offsets or drifts
- filtered preprocessing makes the adaptive thresholds and normalized outputs behave more sensibly
- the weak baseline is more likely explained by preprocessing, dataset mismatch, or channel mapping

## How To Run

From the repo root:

```bash
MPLCONFIGDIR=/tmp/mpl-openbci-audit python3 experiments/openbci_emg_processing_audit/run_audit.py \
  --csv "/Users/yanivnaggar/Desktop/Fall 2025/Independent Study/CogniSync-main/Clench_1/Y_EMG.csv" \
  --output-dir experiments/openbci_emg_processing_audit/outputs \
  --fs 250
```

## Outputs

The audit writes:

- `audit_summary.md`
- `audit_summary.json`
- `channel_diagnostics.csv`
- `mode_comparison.csv`
- `label_level_summary.csv`
- `processed_raw.csv`
- `processed_centered.csv`
- `processed_scaled.csv`
- `processed_filtered.csv`
- per-mode threshold scan CSVs and confusion tables
- PNG diagnostics for tracked signals, threshold dynamics, normalized outputs, x traces, histograms, scatters, and channel-quality overview
- `sanity_window_ch1.csv` for a short hand-checkable sample window

## What To Inspect First

1. `outputs/audit_summary.md`
2. `outputs/mode_comparison.csv`
3. `outputs/tracked_channels_filtered.png`
4. `outputs/threshold_dynamics_filtered.png`
5. `outputs/x_smooth_histograms_mean.png`
6. `outputs/sanity_window_ch1.csv`

## How To Compare It To The Baseline

Compare the best mode in this audit to the original baseline outputs in `experiments/openbci_emg_lr_baseline/outputs/`.

Evidence that preprocessing was the issue would look like:

- much lower OpenBCI clipping behavior after centering/scaling/filtering
- threshold dynamics that stop collapsing to trivial behavior
- more mid-range normalized outputs instead of mostly dead or saturated outputs
- better `Jleft` vs `Jright` threshold-scan metrics in one of the processed modes
- `Jaws` looking more like active-but-near-center in a processed mode than in raw mode

If those improvements are small and all modes remain weak, the stronger explanation shifts toward dataset/protocol mismatch or possible channel-mapping issues rather than simple preprocessing failure.
