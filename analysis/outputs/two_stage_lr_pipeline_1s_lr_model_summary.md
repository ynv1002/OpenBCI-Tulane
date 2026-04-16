# Two-Stage LEFT/RIGHT Pipeline With 1.0s LR Model

- Gate model remains unchanged: calibrated `RandomForest` at `2.0` s with threshold `0.3`
- LEFT/RIGHT model retrained at `1.0` s windows
- Output CSV: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/two_stage_lr_pipeline_1s_lr_model.csv`
- Confusion matrix: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/two_stage_lr_pipeline_1s_lr_model_confusion_matrix.csv`

## Results

| train_file | test_file | gate_window_sec | left_right_window_sec | overlap | calibration_sec | gate_threshold | accuracy | macro_f1 | active_detection_rate | active_recall | left_right_accuracy_on_active_windows | left_right_accuracy_when_detected | gate_selected_channels | left_right_selected_channels | gate_train_calibration_windows | gate_test_calibration_windows | left_right_train_calibration_windows | left_right_test_calibration_windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LR-2-27-26-(01).csv | LR-3-15-26-(04).csv | 2.0 | 1.0 | 0.5 | 45.0 | 0.3 | 0.6608040201005025 | 0.5117428294737782 | 0.25690954773869346 | 0.46416938110749184 | 0.32247557003257327 | 0.6947368421052632 | Channel_1, Channel_3, Channel_4, Channel_6, Channel_7, Channel_8 | Channel_1, Channel_3, Channel_7, Channel_8 | 44 | 44 | {'LR-2-27-26-(01).csv': 89, 'LR-3-8-26-(02).csv': 89, 'LR-3-8-26-(03).csv': 89} | {'LR-3-15-26-(04).csv': 89} |

## Confusion Matrix

| label | pred_REST | pred_LEFT | pred_RIGHT |
| --- | --- | --- | --- |
| true_REST | 854 | 66 | 58 |
| true_LEFT | 175 | 58 | 83 |
| true_RIGHT | 154 | 4 | 140 |

## Comparison vs Previous 2.0s LR Pipeline

- Previous pipeline macro-F1 (2.0s LR): `0.473`
- New pipeline macro-F1 (1.0s LR): `0.512`
- Macro-F1 change: `+0.039`
- Previous LEFT/RIGHT accuracy on true ACTIVE windows: `0.277`
- New LEFT/RIGHT accuracy on true ACTIVE windows: `0.322`
- LEFT/RIGHT-on-active change: `+0.045`
- Previous LEFT/RIGHT accuracy when gate opens: `0.608`
- New LEFT/RIGHT accuracy when gate opens: `0.695`
- LEFT/RIGHT-when-detected change: `+0.087`
- Previous ACTIVE recall: `0.456`
- New ACTIVE recall: `0.464`
- ACTIVE recall change: `+0.008`