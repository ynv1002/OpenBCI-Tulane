# ACTIVE vs REST Experiment

- Output CSV: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/active_vs_rest_results.csv`
- Best model: `RandomForest`
- Best asymmetry setting: `with_asymmetry=True`

## Results

| with_asymmetry | model_name | feature_count | accuracy | macro_f1 | active_precision | active_recall | active_to_rest_errors | rest_to_active_errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | RandomForest | 44 | 0.7051094890510949 | 0.6433762886597938 | 0.7857142857142857 | 0.3613138686131387 | 175 | 27 |
| False | RandomForest | 32 | 0.6992700729927007 | 0.6341495032460125 | 0.7786885245901639 | 0.3467153284671533 | 179 | 27 |
| False | LDA | 32 | 0.6 | 0.375 | 0.0 | 0.0 | 274 | 0 |
| True | LDA | 44 | 0.6 | 0.375 | 0.0 | 0.0 | 274 | 0 |
| False | LogisticRegression | 32 | 0.3781021897810219 | 0.28341649804530183 | 0.38484848484848483 | 0.927007299270073 | 20 | 406 |
| True | LogisticRegression | 44 | 0.37664233576642336 | 0.28260992198096246 | 0.38391502276176026 | 0.9233576642335767 | 21 | 406 |

## Interpretation

- Best setup: `RandomForest` with `with_asymmetry=True`
- Best accuracy: `0.705`
- Best macro-F1: `0.643`
- ACTIVE vs REST is still below the strong-control target of 0.80 macro-F1.
- Asymmetry helped the best observed result.
- Errors are dominated by ACTIVE -> REST misses, so the detector is more conservative than trigger-happy.