# ACTIVE vs REST Threshold Sweep

- Session: `LR-2-27-26-(01).csv`
- Split: first `70%` train, last `30%` test
- Model: `RandomForest`
- Output CSV: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/active_vs_rest_threshold_sweep.csv`

## Results

| threshold | accuracy | macro_f1 | active_precision | active_recall | misses | false_triggers |
| --- | --- | --- | --- | --- | --- | --- |
| 0.3 | 0.7563025210084033 | 0.7518158935634665 | 0.7551020408163265 | 0.6851851851851852 | 17.0 | 12.0 |
| 0.4 | 0.7815126050420168 | 0.7714243498817968 | 0.85 | 0.6296296296296297 | 20.0 | 6.0 |
| 0.5 | 0.7815126050420168 | 0.7656060606060606 | 0.9117647058823529 | 0.5740740740740741 | 23.0 | 3.0 |
| 0.6 | 0.7394957983193278 | 0.7099158606589604 | 0.9259259259259259 | 0.46296296296296297 | 29.0 | 2.0 |

## Interpretation

- Best macro-F1 threshold: `0.4` with macro-F1 `0.771`
- Best recall threshold: `0.3` with ACTIVE recall `0.685`
- Best balance threshold: `0.4` with macro-F1 `0.771`, ACTIVE recall `0.630`, and `6` false triggers