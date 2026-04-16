# ACTIVE vs REST Experiment

- Output CSV: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/active_vs_rest_results_feb27_mar15.csv`
- Best model: `LDA`
- Best asymmetry setting: `with_asymmetry=True`

## Results

| with_asymmetry | model_name | feature_count | accuracy | macro_f1 | active_precision | active_recall | active_to_rest_errors | rest_to_active_errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | LDA | 78 | 0.6773722627737226 | 0.6313626080879362 | 0.6568047337278107 | 0.4051094890510949 | 163 | 58 |
| False | LDA | 48 | 0.6744525547445256 | 0.6124435175530065 | 0.6861313868613139 | 0.34306569343065696 | 180 | 43 |
| True | LogisticRegression | 78 | 0.6175182481751825 | 0.5927282464644263 | 0.5247933884297521 | 0.4635036496350365 | 147 | 115 |
| False | LogisticRegression | 48 | 0.5868613138686132 | 0.5693843143091335 | 0.4835164835164835 | 0.48175182481751827 | 142 | 141 |
| False | RandomForest | 48 | 0.6204379562043796 | 0.49538737023437146 | 0.6 | 0.15328467153284672 | 232 | 28 |
| True | RandomForest | 78 | 0.5795620437956205 | 0.36691312384473196 | 0.0 | 0.0 | 274 | 14 |

## Interpretation

- Best setup: `LDA` with `with_asymmetry=True`
- Best accuracy: `0.677`
- Best macro-F1: `0.631`
- ACTIVE vs REST is still below the strong-control target of 0.80 macro-F1.
- Asymmetry helped the best observed result.
- Errors are dominated by ACTIVE -> REST misses, so the detector is more conservative than trigger-happy.