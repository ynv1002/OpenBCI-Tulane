# OpenBCI EMG Left/Right Baseline Report

## Framing

This is a first-pass offline baseline that applies OpenBCI-style EMG joystick logic to an older hold-based dataset.
The dataset contains sustained `Jaws`, `Jleft`, and `Jright` holds, which is only a partial fit for joystick-style directional control.
That mismatch matters: OpenBCI joystick logic is usually more natural with distinct dynamic left/right motions than long sustained holds.
This report should therefore be read as an algorithm-transfer baseline, not as a statement that the current collection protocol is already ideal.

## Input And Assumptions

- CSV: `/Users/yanivnaggar/Desktop/Fall 2025/Independent Study/CogniSync-main/Clench_1/Y_EMG.csv`
- Sample rate: `250.00 Hz`
- Sample-rate note: Defaulted to 250 Hz because the existing repo uses 250.0 Hz throughout the OpenBCI processing scripts.
- Left channels mapped to X-: `ch1, ch4`
- Right channels mapped to X+: `ch3, ch6`
- Aggregations compared: `mean, max`
- OpenBCI window: `1.00 s`
- OpenBCI uvLimit: `200.00 uV`
- OpenBCI smoothing: `0.90`

## Scale Inspection

- Applied scale mode: `openbci_counts_centered`
- Scale comment: Tracked channels were centered and converted from assumed OpenBCI counts to microvolts; median abs p95=135.52 uV and median abs p99=146.11 uV, which is broadly compatible with the OpenBCI defaults (uvLimit=200.00 uV).
- Detection reasons: median absolute raw channel level 81639.38 exceeds 1000.00, max absolute raw channel value 93573.52 exceeds 10000.00

## Sample Counts

| label | count |
| --- | --- |
| Jaws | 10015 |
| Jleft | 10014 |
| Jright | 10011 |
| missing | 1 |
| norm | 64905 |

## Per-Label Summary

| method | label | left_aggregate_mean | left_aggregate_std | right_aggregate_mean | right_aggregate_std | x_raw_mean | x_raw_std | x_smooth_mean | x_smooth_std | total_activation_mean | total_activation_std | abs_direction_mean | abs_direction_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean | Jaws | 0.7551 | 0.3805 | 0.7363 | 0.3701 | -0.0188 | 0.0406 | -0.0189 | 0.0402 | 1.4913 | 0.7496 | 0.0271 | 0.0352 |
| mean | Jleft | 0.8755 | 0.1199 | 0.8852 | 0.1198 | 0.0097 | 0.0239 | 0.0096 | 0.0238 | 1.7607 | 0.2385 | 0.0204 | 0.0156 |
| mean | Jright | 0.7674 | 0.2765 | 0.7529 | 0.2665 | -0.0145 | 0.0487 | -0.0145 | 0.0485 | 1.5202 | 0.5409 | 0.0368 | 0.0347 |
| mean | missing | 0.5 | nan | 0.4998 | nan | -0.0002 | nan | -0.0001 | nan | 0.9998 | nan | 0.0001 | nan |
| mean | norm | 0.8098 | 0.2956 | 0.7985 | 0.2941 | -0.0113 | 0.035 | -0.0113 | 0.0349 | 1.6083 | 0.5887 | 0.0234 | 0.0282 |
| max | Jaws | 0.7764 | 0.3895 | 0.7658 | 0.3841 | -0.0106 | 0.0257 | -0.0104 | 0.0253 | 1.5422 | 0.7732 | 0.0184 | 0.0202 |
| max | Jleft | 0.9612 | 0.0452 | 0.9468 | 0.061 | -0.0144 | 0.0263 | -0.0145 | 0.0263 | 1.9081 | 0.1041 | 0.0219 | 0.0206 |
| max | Jright | 0.8535 | 0.1811 | 0.7997 | 0.222 | -0.0538 | 0.0547 | -0.0539 | 0.0544 | 1.6532 | 0.4015 | 0.0561 | 0.0521 |
| max | missing | 1.0 | nan | 0.9996 | nan | -0.0004 | nan | -0.0 | nan | 1.9996 | nan | 0.0 | nan |
| max | norm | 0.8549 | 0.2725 | 0.8281 | 0.2827 | -0.0268 | 0.0423 | -0.0268 | 0.0421 | 1.683 | 0.5537 | 0.0299 | 0.0399 |

## Aggregation Results

### `mean` aggregation

- Jleft vs Jright separability: Cohen's d `-0.6308`, overlap `0.5383`, ROC AUC `0.3735`.
- Best directional threshold on `x_smooth`: `0.00` (macro F1 `0.4406`, coverage `1.0000`, accuracy-all `0.4409`).
- Best norm vs active threshold on `total_activation`: `1.88` (macro F1 `0.4757`, accuracy `0.4935`).
- Jaws summary: mean total activation `1.4913`, mean abs direction `0.0271`, activation-vs-norm ratio `0.9273`, abs-direction-vs-lateral ratio `0.9464`.

Directional confusion table:

| index | Jleft | Jright | Undecided |
| --- | --- | --- | --- |
| Jleft | 4637 | 5819 | 0 |
| Jright | 5377 | 4192 | 0 |
| Undecided | 0 | 0 | 0 |

Norm vs active confusion table:

| index | norm | active |
| --- | --- | --- |
| norm | 32167 | 15354 |
| active | 32738 | 14686 |

### `max` aggregation

- Jleft vs Jright separability: Cohen's d `-0.9229`, overlap `0.4612`, ROC AUC `0.2621`.
- Best directional threshold on `x_smooth`: `0.00` (macro F1 `0.3015`, coverage `1.0000`, accuracy-all `0.3555`).
- Best norm vs active threshold on `total_activation`: `1.95` (macro F1 `0.5055`, accuracy `0.5498`).
- Jaws summary: mean total activation `1.5422`, mean abs direction `0.0184`, activation-vs-norm ratio `0.9164`, abs-direction-vs-lateral ratio `0.4725`.

Directional confusion table:

| index | Jleft | Jright | Undecided |
| --- | --- | --- | --- |
| Jleft | 6342 | 9235 | 0 |
| Jright | 3672 | 776 | 0 |
| Undecided | 0 | 0 | 0 |

Norm vs active confusion table:

| index | norm | active |
| --- | --- | --- |
| norm | 40316 | 18154 |
| active | 24589 | 11886 |

## What To Inspect First

1. `summary_report.md` for the scale assumption, threshold scan winners, and Jaws behavior.
2. `x_smooth_histograms.png` to see whether `Jleft`, `Jright`, `Jaws`, and `norm` separate cleanly.
3. `activation_direction_scatter.png` to see whether `Jaws` is active but near-center.
4. `x_raw_x_smooth_window.png` and `tracked_channel_thresholds_window.png` to see whether the adaptive thresholds behave sensibly during label transitions.
