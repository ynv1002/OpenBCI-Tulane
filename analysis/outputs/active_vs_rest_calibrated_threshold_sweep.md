# ACTIVE vs REST Calibrated Threshold Sweep

- Train file: `LR-2-27-26-(01).csv`
- Test file: `LR-3-15-26-(04).csv`
- Model: `RandomForest`
- Features: `combined + asymmetry`
- Calibration period: first `45` seconds
- Train calibration windows used: `44`
- Test calibration windows used: `44`
- Output CSV: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/active_vs_rest_calibrated_threshold_sweep.csv`

## Results

| threshold | accuracy | macro_f1 | active_precision | active_recall | misses | false_triggers |
| --- | --- | --- | --- | --- | --- | --- |
| 0.2 | 0.5357664233576642 | 0.5307248849594126 | 0.43529411764705883 | 0.5401459854014599 | 126.0 | 192.0 |
| 0.3 | 0.7489051094890511 | 0.7054912312276281 | 0.8445945945945946 | 0.4562043795620438 | 149.0 | 23.0 |
| 0.4 | 0.7138686131386861 | 0.640770152831885 | 0.8823529411764706 | 0.3284671532846715 | 184.0 | 12.0 |
| 0.5 | 0.6 | 0.375 | 0.0 | 0.0 | 274.0 | 0.0 |

## Interpretation

- Best macro-F1 threshold: `0.3` with macro-F1 `0.705`
- Best recall threshold: `0.2` with ACTIVE recall `0.540`
- Best practical balance: `0.3` with macro-F1 `0.705`, ACTIVE recall `0.456`, and `23` false triggers