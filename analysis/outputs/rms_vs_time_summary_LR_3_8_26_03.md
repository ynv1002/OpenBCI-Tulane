# RMS vs Time Validation

- Session: `LR-3-8-26-(03).csv`
- Selected channels: `Channel_1, Channel_3, Channel_7, Channel_8`
- Excluded channels from training-screen logic: `{'Channel_2': 'unsafe in training sessions: LR-2-27-26-(01).csv (rail fraction 0.1460), LR-3-8-26-(02).csv (rail fraction 0.9271), LR-3-8-26-(03).csv (dominant exact value fraction 1.0000; rail fraction 1.0000)', 'Channel_4': 'unsafe in training sessions: LR-3-8-26-(02).csv (rail fraction 0.0604)', 'Channel_5': 'unsafe in training sessions: LR-2-27-26-(01).csv (dominant exact value fraction 1.0000; rail fraction 1.0000), LR-3-8-26-(02).csv (rail fraction 0.9715), LR-3-8-26-(03).csv (dominant exact value fraction 1.0000; rail fraction 1.0000)', 'Channel_6': 'unsafe in training sessions: LR-3-8-26-(02).csv (rail fraction 0.1665)'}`
- Window length: `2.00` s
- Overlap: `0.50`
- Top RMS fraction highlighted: `0.35`

## Summary

- Selected channels: Channel_1, Channel_3, Channel_7, Channel_8
- Overall ACTIVE window fraction (ACTIVE + REST only): 0.378
- ACTIVE fraction among top-RMS windows: 0.134
- Median aggregate RMS in ACTIVE windows: 0.312 uV
- Median aggregate RMS in REST windows: 0.392 uV
- ACTIVE segments averaged: 18
- Median ACTIVE duration used for average plot: 8.59 s
- Broad RMS peaks detected in the average ACTIVE profile: 18
- Average-profile RMS peak times (segment-relative seconds): [0.63, 1.18, 1.55, 1.81, 2.04, 2.82, 3.33, 3.51, 3.82, 4.25, 4.8, 5.4, 5.86, 6.38, 7.01, 7.39, 7.7, 8.1]