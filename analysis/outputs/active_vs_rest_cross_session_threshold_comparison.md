# ACTIVE vs REST Cross-Session Threshold Comparison

- Train file: `LR-2-27-26-(01).csv`
- Test file: `LR-3-15-26-(04).csv`
- Model: `RandomForest`
- Features: `combined + asymmetry`
- Output CSV: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/active_vs_rest_cross_session_threshold_comparison.csv`

## Results

| threshold | accuracy | macro_f1 | active_precision | active_recall | misses | false_triggers |
| --- | --- | --- | --- | --- | --- | --- |
| 0.4 | 0.5605839416058395 | 0.38501621086351556 | 0.2 | 0.032846715328467155 | 265.0 | 36.0 |
| 0.5 | 0.5795620437956205 | 0.36691312384473196 | 0.0 | 0.0 | 274.0 | 14.0 |

## Comparison to Previous 0.5 Baseline

- ACTIVE recall change at `0.4` vs `0.5`: `+0.033`
- False trigger change at `0.4` vs `0.5`: `+22`
- Macro-F1 change at `0.4` vs `0.5`: `+0.018`

## Interpretation

- Threshold `0.4` ACTIVE recall: `0.033`
- Threshold `0.5` ACTIVE recall: `0.000`
- Threshold `0.4` false triggers: `36`
- Threshold `0.5` false triggers: `14`