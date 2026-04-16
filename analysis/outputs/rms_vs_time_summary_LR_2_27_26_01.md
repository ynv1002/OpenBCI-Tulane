# RMS vs Time Validation

- Session: `LR-2-27-26-(01).csv`
- Selected channels: `Channel_1, Channel_3, Channel_7, Channel_8`
- Excluded channels from training-screen logic: `{'Channel_2': 'unsafe in training sessions: LR-2-27-26-(01).csv (rail fraction 0.1460), LR-3-8-26-(02).csv (rail fraction 0.9271), LR-3-8-26-(03).csv (dominant exact value fraction 1.0000; rail fraction 1.0000)', 'Channel_4': 'unsafe in training sessions: LR-3-8-26-(02).csv (rail fraction 0.0604)', 'Channel_5': 'unsafe in training sessions: LR-2-27-26-(01).csv (dominant exact value fraction 1.0000; rail fraction 1.0000), LR-3-8-26-(02).csv (rail fraction 0.9715), LR-3-8-26-(03).csv (dominant exact value fraction 1.0000; rail fraction 1.0000)', 'Channel_6': 'unsafe in training sessions: LR-3-8-26-(02).csv (rail fraction 0.1665)'}`
- Window length: `2.00` s
- Overlap: `0.50`
- Top RMS fraction highlighted: `0.35`

## Summary

- Selected channels: Channel_1, Channel_3, Channel_7, Channel_8
- Overall ACTIVE window fraction (ACTIVE + REST only): 0.432
- ACTIVE fraction among top-RMS windows: 0.802
- Median aggregate RMS in ACTIVE windows: 0.452 uV
- Median aggregate RMS in REST windows: 0.328 uV
- ACTIVE segments averaged: 23
- Median ACTIVE duration used for average plot: 8.03 s
- Broad RMS peaks detected in the average ACTIVE profile: 21
- Average-profile RMS peak times (segment-relative seconds): [0.54, 0.89, 1.05, 1.69, 1.83, 2.42, 2.66, 2.76, 3.19, 3.41, 3.76, 4.24, 4.48, 4.67, 4.97, 5.1, 5.26, 5.53, 5.85, 6.58, 7.41]