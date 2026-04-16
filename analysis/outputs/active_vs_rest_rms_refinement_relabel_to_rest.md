# ACTIVE vs REST RMS Refinement Experiment

- Train file: `LR-2-27-26-(01).csv`
- Test file: `LR-3-15-26-(04).csv`
- Weak ACTIVE policy: `relabel_to_rest`
- Output CSV: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/active_vs_rest_rms_refinement_relabel_to_rest.csv`
- Best overall setup: keep top `100%` ACTIVE windows with `LDA`

## Results

| keep_percent | model_name | accuracy | macro_f1 | active_precision | active_recall | train_kept_active_windows | train_weaker_active_windows | test_kept_active_windows | test_weaker_active_windows | active_to_rest_errors | rest_to_active_errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | LogisticRegression | 0.6759124087591241 | 0.5569055944055944 | 0.2523364485981308 | 0.46551724137931033 | 61 | 103 | 116 | 158 | 62 | 160 |
| 30 | LDA | 0.8248175182481752 | 0.5454947582607157 | 0.4375 | 0.1206896551724138 | 61 | 103 | 116 | 158 | 102 | 18 |
| 30 | RandomForest | 0.8306569343065694 | 0.45374800637958534 | 0.0 | 0.0 | 61 | 103 | 116 | 158 | 116 | 0 |
| 40 | LDA | 0.7941605839416058 | 0.5634004005044729 | 0.3333333333333333 | 0.19491525423728814 | 77 | 87 | 118 | 156 | 95 | 46 |
| 40 | LogisticRegression | 0.672992700729927 | 0.5601119214714921 | 0.2590909090909091 | 0.4830508474576271 | 77 | 87 | 118 | 156 | 61 | 163 |
| 40 | RandomForest | 0.8277372262773722 | 0.4528753993610224 | 0.0 | 0.0 | 77 | 87 | 118 | 156 | 118 | 0 |
| 50 | LDA | 0.7693430656934307 | 0.5872234935163997 | 0.48 | 0.23225806451612904 | 86 | 78 | 155 | 119 | 119 | 39 |
| 50 | LogisticRegression | 0.656934306569343 | 0.5714065108403068 | 0.32142857142857145 | 0.4645161290322581 | 86 | 78 | 155 | 119 | 83 | 152 |
| 50 | RandomForest | 0.7737226277372263 | 0.43621399176954734 | 0.0 | 0.0 | 86 | 78 | 155 | 119 | 155 | 0 |
| 100 | LDA | 0.6773722627737226 | 0.6313626080879362 | 0.6568047337278107 | 0.4051094890510949 | 164 | 0 | 274 | 0 | 163 | 58 |
| 100 | LogisticRegression | 0.6175182481751825 | 0.5927282464644263 | 0.5247933884297521 | 0.4635036496350365 | 164 | 0 | 274 | 0 | 147 | 115 |
| 100 | RandomForest | 0.5795620437956205 | 0.36691312384473196 | 0.0 | 0.0 | 164 | 0 | 274 | 0 | 274 | 14 |

## Best Per Threshold

| keep_percent | model_name | accuracy | macro_f1 | active_precision | active_recall | active_to_rest_errors | rest_to_active_errors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | LogisticRegression | 0.6759124087591241 | 0.5569055944055944 | 0.2523364485981308 | 0.46551724137931033 | 62 | 160 |
| 40 | LDA | 0.7941605839416058 | 0.5634004005044729 | 0.3333333333333333 | 0.19491525423728814 | 95 | 46 |
| 50 | LDA | 0.7693430656934307 | 0.5872234935163997 | 0.48 | 0.23225806451612904 | 119 | 39 |
| 100 | LDA | 0.6773722627737226 | 0.6313626080879362 | 0.6568047337278107 | 0.4051094890510949 | 163 | 58 |

## Interpretation

- Baseline best (100% ACTIVE kept): `LDA` with macro-F1 `0.631` and ACTIVE recall `0.405`
- Best refined threshold: keep top `50%` ACTIVE windows with `LDA`
- Best refined macro-F1: `0.587`
- Best refined ACTIVE recall: `0.232`
- ACTIVE recall got worse relative to the unrefined baseline.
- Macro-F1 got worse relative to the unrefined baseline.