# Channel Quality

## Left/Right hand squeeze

- Train files: `LR-2-27-26-(01).csv, LR-3-8-26-(02).csv, LR-3-8-26-(03).csv`
- Test file: `LR-3-15-26-(04).csv`
- Selected channels: `['Channel_1', 'Channel_3', 'Channel_7', 'Channel_8']`
- Excluded channels: `Channel_2: unsafe in training sessions: LR-2-27-26-(01).csv (rail fraction 0.1460), LR-3-8-26-(02).csv (rail fraction 0.9271), LR-3-8-26-(03).csv (dominant exact value fraction 1.0000; rail fraction 1.0000) | Channel_4: unsafe in training sessions: LR-3-8-26-(02).csv (rail fraction 0.0604) | Channel_5: unsafe in training sessions: LR-2-27-26-(01).csv (dominant exact value fraction 1.0000; rail fraction 1.0000), LR-3-8-26-(02).csv (rail fraction 0.9715), LR-3-8-26-(03).csv (dominant exact value fraction 1.0000; rail fraction 1.0000) | Channel_6: unsafe in training sessions: LR-3-8-26-(02).csv (rail fraction 0.1665)`

## Jaw hold vs repeated clench

- Train files: `HR-2-27-26-(01).csv, HR-3-8-26-(02).csv`
- Test file: `HR-3-15-26-(03).csv`
- Selected channels: `['Channel_1', 'Channel_3', 'Channel_4', 'Channel_6', 'Channel_7', 'Channel_8']`
- Excluded channels: `Channel_2: unsafe in training sessions: HR-2-27-26-(01).csv (rail fraction 0.4969), HR-3-8-26-(02).csv (dominant exact value fraction 1.0000; rail fraction 1.0000) | Channel_5: unsafe in training sessions: HR-2-27-26-(01).csv (dominant exact value fraction 1.0000; rail fraction 1.0000), HR-3-8-26-(02).csv (rail fraction 1.0000)`

| family | train_files | test_file | selected_channels | excluded_channels |
| --- | --- | --- | --- | --- |
| left_right | LR-2-27-26-(01).csv, LR-3-8-26-(02).csv, LR-3-8-26-(03).csv | LR-3-15-26-(04).csv | Channel_1, Channel_3, Channel_7, Channel_8 | Channel_2: unsafe in training sessions: LR-2-27-26-(01).csv (rail fraction 0.1460), LR-3-8-26-(02).csv (rail fraction 0.9271), LR-3-8-26-(03).csv (dominant exact value fraction 1.0000; rail fraction 1.0000) | Channel_4: unsafe in training sessions: LR-3-8-26-(02).csv (rail fraction 0.0604) | Channel_5: unsafe in training sessions: LR-2-27-26-(01).csv (dominant exact value fraction 1.0000; rail fraction 1.0000), LR-3-8-26-(02).csv (rail fraction 0.9715), LR-3-8-26-(03).csv (dominant exact value fraction 1.0000; rail fraction 1.0000) | Channel_6: unsafe in training sessions: LR-3-8-26-(02).csv (rail fraction 0.1665) |
| jaw | HR-2-27-26-(01).csv, HR-3-8-26-(02).csv | HR-3-15-26-(03).csv | Channel_1, Channel_3, Channel_4, Channel_6, Channel_7, Channel_8 | Channel_2: unsafe in training sessions: HR-2-27-26-(01).csv (rail fraction 0.4969), HR-3-8-26-(02).csv (dominant exact value fraction 1.0000; rail fraction 1.0000) | Channel_5: unsafe in training sessions: HR-2-27-26-(01).csv (dominant exact value fraction 1.0000; rail fraction 1.0000), HR-3-8-26-(02).csv (rail fraction 1.0000) |