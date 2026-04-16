# Two-Stage Window Size Experiment

- Train file: `LR-2-27-26-(01).csv`
- Test file: `LR-3-15-26-(04).csv`
- Calibration period: first `45` seconds
- Gate threshold: `0.3`
- Output CSV: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/window_size_experiment.csv`

## Results

| window_sec | accuracy | macro_f1 | active_recall | active_detection_rate | left_right_accuracy_on_active_windows | left_right_accuracy_when_detected |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 0.5603015075376885 | 0.4963472634022625 | 0.6954397394136808 | 0.4956030150753769 | 0.4495114006514658 | 0.6463700234192038 |
| 2.0 | 0.6680384087791496 | 0.47272130168681886 | 0.4562043795620438 | 0.23182441700960219 | 0.2773722627737226 | 0.608 |
| 0.5 | 0.3168587022437841 | 0.31666120727915287 | 0.7542768273716952 | 0.769860521528199 | 0.4681181959564541 | 0.6206185567010309 |

## Interpretation

- Best macro-F1 window: `1.0` s with macro-F1 `0.496`
- Best ACTIVE recall window: `0.5` s with ACTIVE recall `0.754`
- Best practical balance: `1.0` s with macro-F1 `0.496`, ACTIVE recall `0.695`, and LEFT/RIGHT accuracy when detected `0.646`

## Confusion Matrices

- `0.5` s: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/two_stage_lr_pipeline_confusion_matrix_0_5.csv`
- `1.0` s: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/two_stage_lr_pipeline_confusion_matrix_1_0.csv`
- `2.0` s: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/two_stage_lr_pipeline_confusion_matrix_2_0.csv`