# Baseline Report: jaw | multiclass

- Train files: `HR-2-27-26-(01).csv, HR-3-8-26-(02).csv`
- Test file: `HR-3-15-26-(03).csv`
- Selected channels: `['Channel_1', 'Channel_3', 'Channel_4', 'Channel_6', 'Channel_7', 'Channel_8']`
- Excluded channels: `Channel_2: unsafe in training sessions: HR-2-27-26-(01).csv (rail fraction 0.4969), HR-3-8-26-(02).csv (dominant exact value fraction 1.0000; rail fraction 1.0000) | Channel_5: unsafe in training sessions: HR-2-27-26-(01).csv (dominant exact value fraction 1.0000; rail fraction 1.0000), HR-3-8-26-(02).csv (rail fraction 1.0000)`
- Window setup: `2.00 s`, overlap `0.50`
- Feature count: `36`
- Train class counts: `{'BASELINE': 95, 'HOLD': 174, 'REPEATED': 178, 'REST': 608}`
- Test class counts: `{'BASELINE': 44, 'HOLD': 68, 'REPEATED': 63, 'REST': 188}`

- LDA: accuracy `0.777`, macro-F1 `0.627`
- LogisticRegression: accuracy `0.471`, macro-F1 `0.527`
- RandomForest: accuracy `0.691`, macro-F1 `0.523`