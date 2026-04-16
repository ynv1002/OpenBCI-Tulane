# Left/Right Spectral Feature Experiment

- Output CSV: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/left_right_spectral_feature_grid_results.csv`
- Total model runs: `72`
- Configuration grid size: `24`

## Top Runs By Macro-F1

| dataset_condition | task_type | feature_mode | with_asymmetry | model_name | feature_count | accuracy | macro_f1 | train_files | test_file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_sessions | 2_class | combined | True | RandomForest | 44 | 0.7518248175182481 | 0.7511752136752137 | LR-2-27-26-(01).csv, LR-3-8-26-(02).csv, LR-3-8-26-(03).csv | LR-3-15-26-(04).csv |
| all_sessions | 2_class | combined | False | RandomForest | 32 | 0.7299270072992701 | 0.7270769396435686 | LR-2-27-26-(01).csv, LR-3-8-26-(02).csv, LR-3-8-26-(03).csv | LR-3-15-26-(04).csv |
| all_sessions | 2_class | time_only | False | RandomForest | 24 | 0.6715328467153284 | 0.6697734211794955 | LR-2-27-26-(01).csv, LR-3-8-26-(02).csv, LR-3-8-26-(03).csv | LR-3-15-26-(04).csv |
| all_sessions | 2_class | time_only | True | RandomForest | 24 | 0.6715328467153284 | 0.6697734211794955 | LR-2-27-26-(01).csv, LR-3-8-26-(02).csv, LR-3-8-26-(03).csv | LR-3-15-26-(04).csv |
| all_sessions | 2_class | combined | False | LogisticRegression | 32 | 0.656934306569343 | 0.6569160272804775 | LR-2-27-26-(01).csv, LR-3-8-26-(02).csv, LR-3-8-26-(03).csv | LR-3-15-26-(04).csv |
| all_sessions | 2_class | combined | True | LogisticRegression | 44 | 0.6532846715328468 | 0.6532800532800533 | LR-2-27-26-(01).csv, LR-3-8-26-(02).csv, LR-3-8-26-(03).csv | LR-3-15-26-(04).csv |
| all_sessions | 2_class | spectral_only | False | RandomForest | 8 | 0.6313868613138686 | 0.6313426672172859 | LR-2-27-26-(01).csv, LR-3-8-26-(02).csv, LR-3-8-26-(03).csv | LR-3-15-26-(04).csv |
| all_sessions | 2_class | combined | False | LDA | 32 | 0.6131386861313869 | 0.612953091684435 | LR-2-27-26-(01).csv, LR-3-8-26-(02).csv, LR-3-8-26-(03).csv | LR-3-15-26-(04).csv |
| all_sessions | 2_class | combined | True | LDA | 44 | 0.6131386861313869 | 0.612953091684435 | LR-2-27-26-(01).csv, LR-3-8-26-(02).csv, LR-3-8-26-(03).csv | LR-3-15-26-(04).csv |
| all_sessions | 2_class | time_only | False | LogisticRegression | 24 | 0.6167883211678832 | 0.611819469742967 | LR-2-27-26-(01).csv, LR-3-8-26-(02).csv, LR-3-8-26-(03).csv | LR-3-15-26-(04).csv |
| all_sessions | 2_class | time_only | True | LogisticRegression | 24 | 0.6167883211678832 | 0.611819469742967 | LR-2-27-26-(01).csv, LR-3-8-26-(02).csv, LR-3-8-26-(03).csv | LR-3-15-26-(04).csv |
| all_sessions | 2_class | time_only | False | LDA | 24 | 0.6094890510948905 | 0.60906727115141 | LR-2-27-26-(01).csv, LR-3-8-26-(02).csv, LR-3-8-26-(03).csv | LR-3-15-26-(04).csv |

## Interpretation

- Best macro-F1 by feature mode: {'combined': 0.7511752136752137, 'time_only': 0.6697734211794955, 'spectral_only': 0.6313426672172859}
- Spectral features helped when combined with time-domain features.
- Asymmetry improved the best observed result.
- Removing the March 8 sessions did not improve the best result.
- The 2-class LEFT vs RIGHT task performed better than the 3-class task.