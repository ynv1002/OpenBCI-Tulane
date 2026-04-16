# Event Model Report: active_vs_other

- Train files: `HR-2-27-26-(01).csv, HR-3-8-26-(02).csv`
- Test file: `HR-3-15-26-(03).csv`
- Selected channels: `['Channel_1', 'Channel_3', 'Channel_4', 'Channel_6', 'Channel_7', 'Channel_8']`
- Excluded channels: `Channel_2: unsafe in training sessions: HR-2-27-26-(01).csv (rail fraction 0.4969), HR-3-8-26-(02).csv (dominant exact value fraction 1.0000; rail fraction 1.0000) | Channel_5: unsafe in training sessions: HR-2-27-26-(01).csv (dominant exact value fraction 1.0000; rail fraction 1.0000), HR-3-8-26-(02).csv (rail fraction 1.0000)`
- Window setup: `0.200 s`, overlap `0.50`
- Feature count: `58`
- Train class counts: `{'ACTIVE': 3145, 'OTHER': 9156}`
- Test class counts: `{'ACTIVE': 991, 'OTHER': 3206}`

- LDA: accuracy `0.904`, macro-F1 `0.853`, per-class recall `{'OTHER': 0.9766063630692452, 'ACTIVE': 0.6680121089808274}`
- LogisticRegression: accuracy `0.942`, macro-F1 `0.923`, per-class recall `{'OTHER': 0.9429195258889582, 'ACTIVE': 0.9384460141271443}`
- RandomForest: accuracy `0.911`, macro-F1 `0.864`, per-class recall `{'OTHER': 0.9809731752963194, 'ACTIVE': 0.6851664984863775}`

- Best model: `LogisticRegression`
- Easiest state: `{'label': 'OTHER', 'recall': 0.9429195258889582}`
- Hardest state: `{'label': 'ACTIVE', 'recall': 0.9384460141271443}`