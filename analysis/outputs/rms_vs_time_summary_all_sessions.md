# RMS vs Time Validation Across Left/Right Sessions

- Selected channels: `Channel_1, Channel_3, Channel_7, Channel_8`
- Excluded channels from training-screen logic: `{'Channel_2': 'unsafe in training sessions: LR-2-27-26-(01).csv (rail fraction 0.1460), LR-3-8-26-(02).csv (rail fraction 0.9271), LR-3-8-26-(03).csv (dominant exact value fraction 1.0000; rail fraction 1.0000)', 'Channel_4': 'unsafe in training sessions: LR-3-8-26-(02).csv (rail fraction 0.0604)', 'Channel_5': 'unsafe in training sessions: LR-2-27-26-(01).csv (dominant exact value fraction 1.0000; rail fraction 1.0000), LR-3-8-26-(02).csv (rail fraction 0.9715), LR-3-8-26-(03).csv (dominant exact value fraction 1.0000; rail fraction 1.0000)', 'Channel_6': 'unsafe in training sessions: LR-3-8-26-(02).csv (rail fraction 0.1665)'}`
- Window length: `2.00` s
- Overlap: `0.50`
- Top RMS fraction highlighted: `0.35`

## LR-2-27-26-(01).csv

- Full-session RMS plot: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/rms_vs_time_visualization_LR_2_27_26_01.png`
- Average ACTIVE profile plot: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/rms_vs_time_average_active_LR_2_27_26_01.png`
- Window table: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/rms_vs_time_windows_LR_2_27_26_01.csv`
- High-RMS windows inside ACTIVE: `0.802` (overall ACTIVE rate `0.432`)
- Median aggregate RMS: `ACTIVE=0.452 uV`, `REST=0.328 uV`
- ACTIVE segments averaged: `23` (median duration `8.03 s`)
- Broad RMS peaks in average ACTIVE profile: `21` at `[0.54, 0.89, 1.05, 1.69, 1.83, 2.42, 2.66, 2.76, 3.19, 3.41, 3.76, 4.24, 4.48, 4.67, 4.97, 5.1, 5.26, 5.53, 5.85, 6.58, 7.41]`

## LR-3-8-26-(02).csv

- Full-session RMS plot: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/rms_vs_time_visualization_LR_3_8_26_02.png`
- Average ACTIVE profile plot: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/rms_vs_time_average_active_LR_3_8_26_02.png`
- Window table: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/rms_vs_time_windows_LR_3_8_26_02.csv`
- High-RMS windows inside ACTIVE: `0.159` (overall ACTIVE rate `0.273`)
- Median aggregate RMS: `ACTIVE=0.371 uV`, `REST=0.473 uV`
- ACTIVE segments averaged: `54` (median duration `5.57 s`)
- Broad RMS peaks in average ACTIVE profile: `13` at `[0.39, 0.8, 0.93, 1.17, 1.38, 1.82, 2.46, 3.09, 3.22, 3.7, 4.39, 4.99, 5.29]`

## LR-3-8-26-(03).csv

- Full-session RMS plot: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/rms_vs_time_visualization_LR_3_8_26_03.png`
- Average ACTIVE profile plot: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/rms_vs_time_average_active_LR_3_8_26_03.png`
- Window table: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/rms_vs_time_windows_LR_3_8_26_03.csv`
- High-RMS windows inside ACTIVE: `0.134` (overall ACTIVE rate `0.378`)
- Median aggregate RMS: `ACTIVE=0.312 uV`, `REST=0.392 uV`
- ACTIVE segments averaged: `18` (median duration `8.59 s`)
- Broad RMS peaks in average ACTIVE profile: `18` at `[0.63, 1.18, 1.55, 1.81, 2.04, 2.82, 3.33, 3.51, 3.82, 4.25, 4.8, 5.4, 5.86, 6.38, 7.01, 7.39, 7.7, 8.1]`

## LR-3-15-26-(04).csv

- Full-session RMS plot: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/rms_vs_time_visualization_LR_3_15_26_04.png`
- Average ACTIVE profile plot: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/rms_vs_time_average_active_LR_3_15_26_04.png`
- Window table: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/rms_vs_time_windows_LR_3_15_26_04.csv`
- High-RMS windows inside ACTIVE: `0.567` (overall ACTIVE rate `0.417`)
- Median aggregate RMS: `ACTIVE=0.419 uV`, `REST=0.366 uV`
- ACTIVE segments averaged: `39` (median duration `8.65 s`)
- Broad RMS peaks in average ACTIVE profile: `24` at `[0.61, 0.75, 1.62, 2.02, 2.23, 2.49, 2.86, 3.09, 3.33, 3.64, 3.9, 4.22, 4.66, 5.03, 5.26, 5.47, 5.67, 6.01, 6.56, 6.88, 7.29, 7.58, 7.89, 8.47]`
