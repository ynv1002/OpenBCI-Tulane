# Baseline Report: jaw | binary_clench_vs_nonclench

- Train files: `HR-2-27-26-(01).csv, HR-3-8-26-(02).csv`
- Test file: `HR-3-15-26-(03).csv`
- Selected channels: `['Channel_1', 'Channel_3', 'Channel_4', 'Channel_6', 'Channel_7', 'Channel_8']`
- Excluded channels: `Channel_2: unsafe in training sessions: HR-2-27-26-(01).csv (rail fraction 0.4969), HR-3-8-26-(02).csv (dominant exact value fraction 1.0000; rail fraction 1.0000) | Channel_5: unsafe in training sessions: HR-2-27-26-(01).csv (dominant exact value fraction 1.0000; rail fraction 1.0000), HR-3-8-26-(02).csv (rail fraction 1.0000)`
- Window setup: `2.00 s`, overlap `0.50`
- Feature count: `36`
- Train class counts: `{'CLENCH': 352, 'NON_CLENCH': 703}`
- Test class counts: `{'CLENCH': 131, 'NON_CLENCH': 232}`

- LDA: accuracy `0.917`, macro-F1 `0.908`
- LogisticRegression: accuracy `0.964`, macro-F1 `0.962`
- RandomForest: accuracy `0.813`, macro-F1 `0.765`