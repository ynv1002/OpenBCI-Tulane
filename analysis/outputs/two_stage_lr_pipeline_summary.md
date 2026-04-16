# Two-Stage LEFT/RIGHT Pipeline

- Train file: `LR-2-27-26-(01).csv`
- Test file: `LR-3-15-26-(04).csv`
- Calibration period: first `45` seconds
- Gate model: `RandomForest` with threshold `0.3` (trained on `LR-2-27-26-(01).csv`)
- LEFT vs RIGHT model: `RandomForest` (trained on `LR-2-27-26-(01).csv, LR-3-8-26-(02).csv, LR-3-8-26-(03).csv`)
- Output CSV: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/two_stage_lr_pipeline_results.csv`
- Confusion matrix: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/two_stage_lr_pipeline_confusion_matrix.csv`

## Results

| train_file | test_file | calibration_sec | gate_model | gate_threshold | left_right_model | gate_selected_channels | left_right_selected_channels | accuracy | macro_f1 | active_detection_rate | active_recall | left_right_accuracy_on_active_windows | left_right_accuracy_when_detected | gate_train_calibration_windows | gate_test_calibration_windows | left_right_train_calibration_windows | left_right_test_calibration_windows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LR-2-27-26-(01).csv | LR-3-15-26-(04).csv | 45.0 | RandomForest | 0.3 | RandomForest | Channel_1, Channel_3, Channel_4, Channel_6, Channel_7, Channel_8 | Channel_1, Channel_3, Channel_7, Channel_8 | 0.6680384087791496 | 0.47272130168681886 | 0.23182441700960219 | 0.4562043795620438 | 0.2773722627737226 | 0.608 | 44 | 44 | {'LR-2-27-26-(01).csv': 44, 'LR-3-8-26-(02).csv': 44, 'LR-3-8-26-(03).csv': 44} | {'LR-3-15-26-(04).csv': 44} |

## Confusion Matrix

| label | pred_REST | pred_LEFT | pred_RIGHT |
| --- | --- | --- | --- |
| true_REST | 411 | 15 | 29 |
| true_LEFT | 80 | 12 | 49 |
| true_RIGHT | 69 | 0 | 64 |

## Interpretation

- Overall 3-class macro-F1: `0.473`
- ACTIVE detection rate on all test windows: `0.232`
- ACTIVE recall of the gate on true ACTIVE windows: `0.456`
- LEFT vs RIGHT accuracy on true ACTIVE windows after gating: `0.277`
- LEFT vs RIGHT accuracy when the gate opens on true ACTIVE windows: `0.608`