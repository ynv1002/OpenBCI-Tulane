# OpenBCI EMG Processing Audit

## Why This Audit Exists

The original baseline applied the OpenBCI-style EMG joystick logic closely, but left/right results were weak.
This audit checks whether the weakness is explained by preprocessing and signal assumptions before the algorithm itself is blamed or the collection protocol is redesigned.

## Input And Assumptions

- CSV: `/Users/yanivnaggar/Desktop/Fall 2025/Independent Study/CogniSync-main/Clench_1/Y_EMG.csv`
- Sample rate: `250.00 Hz`
- Sample-rate note: Defaulted to 250 Hz because the existing repo uses 250.0 Hz throughout the OpenBCI processing scripts.
- Tracked channels: `ch1, ch4, ch3, ch6`
- Left mapping: `ch1, ch4`
- Right mapping: `ch3, ch6`
- Count-to-uV assumption tested: `0.02235 uV/count`

## Audit Questions

1. Likely input type: `raw_ads1299_like_counts`.
2. `0.02235 uV/count` plausible: `True`.
3. Channels near the nominal rail: `ch5`.
4. Filter pipeline used for Mode D: `scipy: notch 60.0 Hz (Q=30.0), butter bandpass 20.0-100.0 Hz order 4`.

## High-Level Findings

- Best Jleft vs Jright separation in this audit: mode `scaled` with `mean` aggregation (macro F1 `0.4406`, ROC AUC `0.3735`).
- Most sensible OpenBCI threshold behavior by heuristic: mode `scaled` with `mean` aggregation (clip fraction `0.0000`, normalized mid fraction `0.4767`).
- Best Jaws near-center behavior among non-degenerate modes: mode `scaled` with `max` aggregation (Jaws center score `0.4439`).

## Preprocessing Interpretation

- Filtered preprocessing did not materially rescue the left/right metric over raw mode, which suggests preprocessing is not the whole explanation.
- The best-performing mode still has a suspicious direction ordering, so channel mapping or label-side alignment should remain under suspicion.
- If all modes remain weak after thresholds and normalization look reasonable, that points more toward dataset/protocol mismatch than simple preprocessing failure.

## Raw Channel Flags

| channel | flag_suspicious_saturation | flag_near_constant | flag_unusually_noisy | flag_large_drift |
| --- | --- | --- | --- | --- |
| ch1 | False | False | False | True |
| ch2 | False | False | False | True |
| ch3 | False | False | False | True |
| ch4 | False | False | False | True |
| ch5 | True | False | False | False |
| ch6 | False | False | False | True |
| ch7 | False | False | False | True |
| ch8 | False | False | False | True |

## Mode Comparison

| mode | method | direction_macro_f1 | roc_auc_x_smooth | average_uv_clip_fraction_mean | normalized_mid_fraction_mean | activation_macro_f1 | jaws_center_score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw | mean | 0.0 | 0.5 | 1.0 | 0.0 | 0.406 | 1.0 |
| raw | max | 0.0 | 0.5 | 1.0 | 0.0 | 0.406 | 1.0 |
| centered | mean | 0.3331 | 0.4692 | 0.9129 | 0.0213 | 0.4061 | -3.7502 |
| centered | max | 0.2679 | 0.4001 | 0.9129 | 0.0213 | 0.4061 | -2.5241 |
| scaled | mean | 0.4406 | 0.3735 | 0.0 | 0.4767 | 0.4757 | -0.0192 |
| scaled | max | 0.3015 | 0.2621 | 0.0 | 0.4767 | 0.5055 | 0.4439 |
| filtered | mean | 0.0 | 0.5 | 0.0 | 0.0 | 0.406 | 0.0 |
| filtered | max | 0.0 | 0.5 | 0.0 | 0.0 | 0.406 | 0.0 |

## What To Inspect First

1. `audit_summary.md` for the mode ranking and interpretation.
2. `mode_comparison.csv` for the numeric comparison across modes.
3. `tracked_channels_filtered.png`, `threshold_dynamics_filtered.png`, and `normalized_outputs_filtered.png` to see whether filtering makes the processor behave more sensibly.
4. `x_smooth_histograms_mean.png` and `activation_scatter_mean.png` to compare label structure across modes.
5. `sanity_window_ch1.csv` to hand-check stage values and threshold updates over a short sample window.
