# RMS vs Time Validation

- Session: `LR-3-15-26-(04).csv`
- Selected channels: `Channel_1, Channel_3, Channel_7, Channel_8`
- Excluded channels from training-screen logic: `{'Channel_2': 'unsafe in training sessions: LR-2-27-26-(01).csv (rail fraction 0.1460), LR-3-8-26-(02).csv (rail fraction 0.9271), LR-3-8-26-(03).csv (dominant exact value fraction 1.0000; rail fraction 1.0000)', 'Channel_4': 'unsafe in training sessions: LR-3-8-26-(02).csv (rail fraction 0.0604)', 'Channel_5': 'unsafe in training sessions: LR-2-27-26-(01).csv (dominant exact value fraction 1.0000; rail fraction 1.0000), LR-3-8-26-(02).csv (rail fraction 0.9715), LR-3-8-26-(03).csv (dominant exact value fraction 1.0000; rail fraction 1.0000)', 'Channel_6': 'unsafe in training sessions: LR-3-8-26-(02).csv (rail fraction 0.1665)'}`
- Window length: `2.00` s
- Overlap: `0.50`
- Top RMS fraction highlighted: `0.35`

## Summary

- Selected channels: Channel_1, Channel_3, Channel_7, Channel_8
- Overall ACTIVE window fraction (ACTIVE + REST only): 0.417
- ACTIVE fraction among top-RMS windows: 0.567
- Median aggregate RMS in ACTIVE windows: 0.419 uV
- Median aggregate RMS in REST windows: 0.366 uV
- Zoom segment: RIGHT from 488.23s to 496.92s
- Broad RMS peaks detected inside zoom segment: 1
- Zoom-segment RMS peak times (broad window centers): [490.52]