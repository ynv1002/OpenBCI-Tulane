# Clean Cross-Session LEFT vs RIGHT EEG Retrain

- Sessions: `LR-2-27-26-(01).csv, LR-3-15-26-(04).csv`
- Shared selected channels: `Channel_1, Channel_3, Channel_4, Channel_6, Channel_7, Channel_8`
- Excluded channels: `Channel_2: unsafe in training sessions: LR-2-27-26-(01).csv (rail fraction 0.1460), LR-3-15-26-(04).csv (rail fraction 0.9991) | Channel_5: unsafe in training sessions: LR-2-27-26-(01).csv (dominant exact value fraction 1.0000; rail fraction 1.0000), LR-3-15-26-(04).csv (rail fraction 0.9959)`
- Windowing: `1.0` s with overlap `0.5`
- Calibration: first `45` seconds per session using baseline/rest windows
- Features: `combined` with asymmetry `True`

## Per-Fold Results

| model_name | train_session | test_session | accuracy | macro_f1 | feature_count | confusion_matrix_csv |
| --- | --- | --- | --- | --- | --- | --- |
| RandomForest | LR-3-15-26-(04).csv | LR-2-27-26-(01).csv | 0.5137362637362637 | 0.49335912618839795 | 78 | /Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/confusion_eeg_direction_clean_cross_session_RandomForest_LR-3-15-26-04_csv_to_LR-2-27-26-01_csv.csv |
| LogisticRegression | LR-3-15-26-(04).csv | LR-2-27-26-(01).csv | 0.49175824175824173 | 0.45944210315071243 | 78 | /Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/confusion_eeg_direction_clean_cross_session_LogisticRegression_LR-3-15-26-04_csv_to_LR-2-27-26-01_csv.csv |
| LDA | LR-3-15-26-(04).csv | LR-2-27-26-(01).csv | 0.510989010989011 | 0.34314679643146795 | 78 | /Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/confusion_eeg_direction_clean_cross_session_LDA_LR-3-15-26-04_csv_to_LR-2-27-26-01_csv.csv |
| LogisticRegression | LR-2-27-26-(01).csv | LR-3-15-26-(04).csv | 0.4788273615635179 | 0.47882183176300824 | 78 | /Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/confusion_eeg_direction_clean_cross_session_LogisticRegression_LR-2-27-26-01_csv_to_LR-3-15-26-04_csv.csv |
| RandomForest | LR-2-27-26-(01).csv | LR-3-15-26-(04).csv | 0.5146579804560261 | 0.33978494623655914 | 78 | /Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/confusion_eeg_direction_clean_cross_session_RandomForest_LR-2-27-26-01_csv_to_LR-3-15-26-04_csv.csv |
| LDA | LR-2-27-26-(01).csv | LR-3-15-26-(04).csv | 0.3566775244299674 | 0.336528544892996 | 78 | /Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/confusion_eeg_direction_clean_cross_session_LDA_LR-2-27-26-01_csv_to_LR-3-15-26-04_csv.csv |

## Pooled LOSO Results

| model_name | accuracy | macro_f1 | feature_count | confusion_matrix_csv |
| --- | --- | --- | --- | --- |
| LogisticRegression | 0.483640081799591 | 0.47952134806594904 | 78 | /Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/confusion_eeg_direction_clean_cross_session_LogisticRegression_pooled.csv |
| RandomForest | 0.5143149284253579 | 0.4230360808929372 | 78 | /Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/confusion_eeg_direction_clean_cross_session_RandomForest_pooled.csv |
| LDA | 0.41411042944785276 | 0.4104559967597877 | 78 | /Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/confusion_eeg_direction_clean_cross_session_LDA_pooled.csv |

## Recommendation

- Winning model: `LogisticRegression`
- Winning pooled macro-F1: `0.480`
- Winning pooled accuracy: `0.484`
- Hybrid reuse readout: not strong enough to reuse directly; keep as a clean benchmark and retrain again after hybrid integration changes
- Saved artifact: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/clean_left_right_window_model.pkl`