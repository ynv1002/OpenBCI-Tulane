# RMS vs Time Validation

- Session: `LR-3-8-26-(02).csv`
- Selected channels: `Channel_1, Channel_3, Channel_7, Channel_8`
- Excluded channels from training-screen logic: `{'Channel_2': 'unsafe in training sessions: LR-2-27-26-(01).csv (rail fraction 0.1460), LR-3-8-26-(02).csv (rail fraction 0.9271), LR-3-8-26-(03).csv (dominant exact value fraction 1.0000; rail fraction 1.0000)', 'Channel_4': 'unsafe in training sessions: LR-3-8-26-(02).csv (rail fraction 0.0604)', 'Channel_5': 'unsafe in training sessions: LR-2-27-26-(01).csv (dominant exact value fraction 1.0000; rail fraction 1.0000), LR-3-8-26-(02).csv (rail fraction 0.9715), LR-3-8-26-(03).csv (dominant exact value fraction 1.0000; rail fraction 1.0000)', 'Channel_6': 'unsafe in training sessions: LR-3-8-26-(02).csv (rail fraction 0.1665)'}`
- Window length: `2.00` s
- Overlap: `0.50`
- Top RMS fraction highlighted: `0.35`

## Summary

- Selected channels: Channel_1, Channel_3, Channel_7, Channel_8
- Overall ACTIVE window fraction (ACTIVE + REST only): 0.273
- ACTIVE fraction among top-RMS windows: 0.159
- Median aggregate RMS in ACTIVE windows: 0.371 uV
- Median aggregate RMS in REST windows: 0.473 uV
- ACTIVE segments averaged: 54
- Median ACTIVE duration used for average plot: 5.57 s
- Broad RMS peaks detected in the average ACTIVE profile: 13
- Average-profile RMS peak times (segment-relative seconds): [0.39, 0.8, 0.93, 1.17, 1.38, 1.82, 2.46, 3.09, 3.22, 3.7, 4.39, 4.99, 5.29]