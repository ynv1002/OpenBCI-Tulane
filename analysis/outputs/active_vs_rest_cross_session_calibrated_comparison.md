# ACTIVE vs REST Cross-Session Calibration Normalization

- Train file: `LR-2-27-26-(01).csv`
- Test file: `LR-3-15-26-(04).csv`
- Model: `RandomForest`
- Features: `combined + asymmetry`
- Threshold: `0.4`
- Calibration period: first `45` seconds
- Train calibration windows used: `44`
- Test calibration windows used: `44`
- Output CSV: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/active_vs_rest_cross_session_calibrated_comparison.csv`

## Results

| condition | threshold | accuracy | macro_f1 | active_precision | active_recall | misses | false_triggers | train_calibration_windows | test_calibration_windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_normalization | 0.4 | 0.5605839416058395 | 0.3850162108635155 | 0.2 | 0.0328467153284671 | 265 | 36 | 44 | 44 |
| calibrated_per_session_baseline | 0.4 | 0.7138686131386861 | 0.640770152831885 | 0.8823529411764706 | 0.3284671532846715 | 184 | 12 | 44 | 44 |

## Comparison

- ACTIVE recall change vs no normalization: `+0.296`
- Macro-F1 change vs no normalization: `+0.256`
- False trigger change vs no normalization: `-24`
- Miss change vs no normalization: `-81`

## Interpretation

- Calibrated ACTIVE recall: `0.328`
- Calibrated macro-F1: `0.641`
- Calibrated false triggers: `12`