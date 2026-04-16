# Baseline Report: left_right | multiclass

- Train files: `LR-2-27-26-(01).csv, LR-3-8-26-(02).csv, LR-3-8-26-(03).csv`
- Test file: `LR-3-15-26-(04).csv`
- Selected channels: `['Channel_1', 'Channel_3', 'Channel_7', 'Channel_8']`
- Excluded channels: `Channel_2: unsafe in training sessions: LR-2-27-26-(01).csv (rail fraction 0.1460), LR-3-8-26-(02).csv (rail fraction 0.9271), LR-3-8-26-(03).csv (dominant exact value fraction 1.0000; rail fraction 1.0000) | Channel_4: unsafe in training sessions: LR-3-8-26-(02).csv (rail fraction 0.0604) | Channel_5: unsafe in training sessions: LR-2-27-26-(01).csv (dominant exact value fraction 1.0000; rail fraction 1.0000), LR-3-8-26-(02).csv (rail fraction 0.9715), LR-3-8-26-(03).csv (dominant exact value fraction 1.0000; rail fraction 1.0000) | Channel_6: unsafe in training sessions: LR-3-8-26-(02).csv (rail fraction 0.1665)`
- Window setup: `2.00 s`, overlap `0.50`
- Feature count: `24`
- Train class counts: `{'BASELINE': 147, 'LEFT': 284, 'REST': 1280, 'RIGHT': 260}`
- Test class counts: `{'BASELINE': 44, 'LEFT': 141, 'REST': 411, 'RIGHT': 133}`

- LDA: accuracy `0.564`, macro-F1 `0.180`
- LogisticRegression: accuracy `0.132`, macro-F1 `0.078`
- RandomForest: accuracy `0.601`, macro-F1 `0.365`