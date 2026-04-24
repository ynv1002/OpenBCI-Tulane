# Clean Cross-Session LEFT vs RIGHT EEG Retrain

- Sessions: `LR-2-27-26-(01).csv, LR-3-15-26-(04).csv`
- Shared selected channels: `Channel_1, Channel_3, Channel_4, Channel_6, Channel_7, Channel_8`
- Excluded channels: `Channel_2: unsafe in training sessions: LR-2-27-26-(01).csv (rail fraction 0.1460), LR-3-15-26-(04).csv (rail fraction 0.9991) | Channel_5: unsafe in training sessions: LR-2-27-26-(01).csv (dominant exact value fraction 1.0000; rail fraction 1.0000), LR-3-15-26-(04).csv (rail fraction 0.9959)`
- Windowing: `1.0` s with overlap `0.5`
- Calibration: first `45` seconds per session using baseline/rest windows
- Features: `combined` with asymmetry `True`

## Per-Fold Results

| model_name | train_session | test_session | accuracy | macro_f1 | feature_count |
| --- | --- | --- | --- | --- | --- |
| RandomForest | LR-3-15-26-(04).csv | LR-2-27-26-(01).csv | 0.514 | 0.493 | 78 |
| LogisticRegression | LR-3-15-26-(04).csv | LR-2-27-26-(01).csv | 0.492 | 0.459 | 78 |
| LDA | LR-3-15-26-(04).csv | LR-2-27-26-(01).csv | 0.511 | 0.343 | 78 |
| LogisticRegression | LR-2-27-26-(01).csv | LR-3-15-26-(04).csv | 0.479 | 0.479 | 78 |
| RandomForest | LR-2-27-26-(01).csv | LR-3-15-26-(04).csv | 0.515 | 0.340 | 78 |
| LDA | LR-2-27-26-(01).csv | LR-3-15-26-(04).csv | 0.357 | 0.337 | 78 |

## Pooled LOSO Results

| model_name | accuracy | macro_f1 | feature_count |
| --- | --- | --- | --- |
| LogisticRegression | 0.484 | 0.480 | 78 |
| RandomForest | 0.514 | 0.423 | 78 |
| LDA | 0.414 | 0.410 | 78 |

## Recommendation

- Winning model: `LogisticRegression`
- Winning pooled macro-F1: `0.480`
- Winning pooled accuracy: `0.484`
- Hybrid reuse readout: not strong enough to reuse directly; keep as a clean benchmark and retrain again after hybrid integration changes
- Saved artifact: `final_package/models/clean_left_right_window_model.pkl`
