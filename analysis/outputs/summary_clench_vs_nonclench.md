# Event Model Report: clench_vs_nonclench

- Train files: `HR-2-27-26-(01).csv, HR-3-8-26-(02).csv`
- Test file: `HR-3-15-26-(03).csv`
- Selected channels: `['Channel_1', 'Channel_3', 'Channel_4', 'Channel_6', 'Channel_7', 'Channel_8']`
- Excluded channels: `Channel_2: unsafe in training sessions: HR-2-27-26-(01).csv (rail fraction 0.4969), HR-3-8-26-(02).csv (dominant exact value fraction 1.0000; rail fraction 1.0000) | Channel_5: unsafe in training sessions: HR-2-27-26-(01).csv (dominant exact value fraction 1.0000; rail fraction 1.0000), HR-3-8-26-(02).csv (rail fraction 1.0000)`
- Window setup: `0.200 s`, overlap `0.50`
- Feature count: `58`
- Train class counts: `{'CLENCH': 3989, 'NON_CLENCH': 8312}`
- Test class counts: `{'CLENCH': 1392, 'NON_CLENCH': 2805}`

- LDA: accuracy `0.911`, macro-F1 `0.897`, per-class recall `{'NON_CLENCH': 0.9607843137254902, 'CLENCH': 0.8117816091954023}`
- LogisticRegression: accuracy `0.936`, macro-F1 `0.929`, per-class recall `{'NON_CLENCH': 0.930837789661319, 'CLENCH': 0.9468390804597702}`
- RandomForest: accuracy `0.912`, macro-F1 `0.899`, per-class recall `{'NON_CLENCH': 0.9529411764705882, 'CLENCH': 0.8304597701149425}`

- Best model: `LogisticRegression`
- Easiest state: `{'label': 'CLENCH', 'recall': 0.9468390804597702}`
- Hardest state: `{'label': 'NON_CLENCH', 'recall': 0.930837789661319}`