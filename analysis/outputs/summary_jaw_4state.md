# Event Model Report: jaw_4state

- Train files: `HR-2-27-26-(01).csv, HR-3-8-26-(02).csv`
- Test file: `HR-3-15-26-(03).csv`
- Selected channels: `['Channel_1', 'Channel_3', 'Channel_4', 'Channel_6', 'Channel_7', 'Channel_8']`
- Excluded channels: `Channel_2: unsafe in training sessions: HR-2-27-26-(01).csv (rail fraction 0.4969), HR-3-8-26-(02).csv (dominant exact value fraction 1.0000; rail fraction 1.0000) | Channel_5: unsafe in training sessions: HR-2-27-26-(01).csv (dominant exact value fraction 1.0000; rail fraction 1.0000), HR-3-8-26-(02).csv (rail fraction 1.0000)`
- Window setup: `0.200 s`, overlap `0.50`
- Feature count: `58`
- Train class counts: `{'ACTIVE': 3145, 'INACTIVE': 8312, 'OFFSET': 426, 'ONSET': 418}`
- Test class counts: `{'ACTIVE': 991, 'INACTIVE': 2805, 'OFFSET': 200, 'ONSET': 201}`

- LDA: accuracy `0.867`, macro-F1 `0.705`, per-class recall `{'INACTIVE': 0.9611408199643494, 'ONSET': 0.4975124378109453, 'ACTIVE': 0.739656912209889, 'OFFSET': 0.555}`
- LogisticRegression: accuracy `0.885`, macro-F1 `0.783`, per-class recall `{'INACTIVE': 0.8898395721925134, 'ONSET': 0.8507462686567164, 'ACTIVE': 0.8829465186680121, 'OFFSET': 0.87}`
- RandomForest: accuracy `0.889`, macro-F1 `0.771`, per-class recall `{'INACTIVE': 0.960427807486631, 'ONSET': 0.5572139303482587, 'ACTIVE': 0.8082744702320888, 'OFFSET': 0.625}`

- Best model: `LogisticRegression`
- Easiest state: `{'label': 'INACTIVE', 'recall': 0.8898395721925134}`
- Hardest state: `{'label': 'ONSET', 'recall': 0.8507462686567164}`