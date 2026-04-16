# Marker Audit

| family | session_rank | filename | movement1_count | movement2_count | baseline_median_sec | movement1_median_sec | movement2_median_sec | marker_counts | pairing_issue_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jaw | 1 | HR-2-27-26-(01).csv | 11 | 12 | 45.59 | 8.38 | 8.81 | {0: 127168, 1: 12, 2: 12, 3: 14, 4: 12} | 13 |
| jaw | 2 | HR-3-8-26-(02).csv | 18 | 18 | 52.59 | 7.55 | 6.41 | {0: 180304, 1: 18, 2: 18, 3: 18, 4: 18} | 8 |
| jaw | 3 | HR-3-15-26-(03).csv | 10 | 9 | 45.15 | 8.74 | 8.49 | {0: 104928, 1: 10, 2: 11, 3: 9, 4: 10} | 7 |
| left_right | 1 | LR-2-27-26-(01).csv | 12 | 11 | 47.79 | 8.03 | 8.03 | {0: 126724, 1: 12, 2: 12, 3: 11, 4: 11} | 5 |
| left_right | 2 | LR-3-8-26-(02).csv | 27 | 27 | 54.9 | 5.51 | 5.89 | {0: 322461, 1: 27, 2: 27, 3: 27, 4: 27} | 10 |
| left_right | 3 | LR-3-8-26-(03).csv | 9 | 9 | 49.93 | 8.71 | 8.34 | {0: 113483, 1: 9, 2: 9, 3: 9, 4: 9} | 7 |
| left_right | 4 | LR-3-15-26-(04).csv | 20 | 19 | 45.71 | 8.7 | 8.55 | {0: 211719, 1: 20, 2: 21, 3: 19, 4: 20} | 10 |

## jaw | HR-2-27-26-(01).csv

- File: `/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/EMG_JvsN/HR-2-27-26-(01).csv`
- Shape: `(127218, 24)`
- Sample-rate used: `249.260 Hz`
- Marker counts: `{0: 127168, 1: 12, 2: 12, 3: 14, 4: 12}`
- Movement 1 intervals: `11`
- Movement 2 intervals: `12`
- Baseline median duration: `45.59 s`
- Movement 1 median duration: `8.38 s`
- Movement 2 median duration: `8.81 s`
- Sampling notes: `Timestamp has 4 jump(s) larger than 5x the median step; windowing uses the median step instead of total-span duration.`
- Issues: `unexpected marker transition 1->3 between event 25 and 26 | unexpected marker transition 3->3 between event 26 and 27 | unexpected marker transition 4->2 between event 40 and 41 | unexpected marker transition 3->3 between event 48 and 49 | marker 1 reappeared before marker 2; replaced the open interval that started at sample 68960 | marker 2 appeared without a matching prior marker 1 at sample 109337 | marker 3 reappeared before marker 4; replaced the open interval that started at sample 71161 | marker 3 reappeared before marker 4; replaced the open interval that started at sample 123242 | movement 1 produced 11 interval(s); protocol expected about 10 | movement 2 produced 12 interval(s); protocol expected about 10 | Timestamp has 4 jump(s) larger than 5x the median step; windowing uses the median step instead of total-span duration.`

## jaw | HR-3-8-26-(02).csv

- File: `/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/EMG_JvsN/HR-3-8-26-(02).csv`
- Shape: `(180376, 24)`
- Sample-rate used: `249.186 Hz`
- Marker counts: `{0: 180304, 1: 18, 2: 18, 3: 18, 4: 18}`
- Movement 1 intervals: `18`
- Movement 2 intervals: `18`
- Baseline median duration: `52.59 s`
- Movement 1 median duration: `7.55 s`
- Movement 2 median duration: `6.41 s`
- Sampling notes: `Timestamp has 5 jump(s) larger than 5x the median step; windowing uses the median step instead of total-span duration. | Overall timestamp-derived rate (187.05 Hz) disagrees with median-step rate (249.19 Hz).`
- Issues: `movement 1 produced 18 interval(s); protocol expected about 10 | movement 2 produced 18 interval(s); protocol expected about 10 | movement2_to_movement1 rest median is 11.65 s instead of the expected 15 s | Timestamp has 5 jump(s) larger than 5x the median step; windowing uses the median step instead of total-span duration. | Overall timestamp-derived rate (187.05 Hz) disagrees with median-step rate (249.19 Hz).`

## jaw | HR-3-15-26-(03).csv

- File: `/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/EMG_JvsN/HR-3-15-26-(03).csv`
- Shape: `(104968, 24)`
- Sample-rate used: `249.186 Hz`
- Marker counts: `{0: 104928, 1: 10, 2: 11, 3: 9, 4: 10}`
- Movement 1 intervals: `10`
- Movement 2 intervals: `9`
- Baseline median duration: `45.15 s`
- Movement 1 median duration: `8.74 s`
- Movement 2 median duration: `8.49 s`
- Sampling notes: `Timestamp has 1 jump(s) larger than 5x the median step; windowing uses the median step instead of total-span duration.`
- Issues: `unexpected marker transition 2->2 between event 34 and 35 | unexpected marker transition 2->4 between event 35 and 36 | marker 2 appeared without a matching prior marker 1 at sample 91610 | marker 4 appeared without a matching prior marker 3 at sample 93717 | movement 2 produced 9 interval(s); protocol expected about 10 | Timestamp has 1 jump(s) larger than 5x the median step; windowing uses the median step instead of total-span duration.`

## left_right | LR-2-27-26-(01).csv

- File: `/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/EEG_LR/LR-2-27-26-(01).csv`
- Shape: `(126770, 24)`
- Sample-rate used: `249.186 Hz`
- Marker counts: `{0: 126724, 1: 12, 2: 12, 3: 11, 4: 11}`
- Movement 1 intervals: `12`
- Movement 2 intervals: `11`
- Baseline median duration: `47.79 s`
- Movement 1 median duration: `8.03 s`
- Movement 2 median duration: `8.03 s`
- Sampling notes: `Timestamp has 4 jump(s) larger than 5x the median step; windowing uses the median step instead of total-span duration.`
- Issues: `movement 1 produced 12 interval(s); protocol expected about 10 | movement 2 produced 11 interval(s); protocol expected about 10 | Timestamp has 4 jump(s) larger than 5x the median step; windowing uses the median step instead of total-span duration.`

## left_right | LR-3-8-26-(02).csv

- File: `/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/EEG_LR/LR-3-8-26-(02).csv`
- Shape: `(322569, 24)`
- Sample-rate used: `249.186 Hz`
- Marker counts: `{0: 322461, 1: 27, 2: 27, 3: 27, 4: 27}`
- Movement 1 intervals: `27`
- Movement 2 intervals: `27`
- Baseline median duration: `54.90 s`
- Movement 1 median duration: `5.51 s`
- Movement 2 median duration: `5.89 s`
- Sampling notes: `Timestamp has 9 jump(s) larger than 5x the median step; windowing uses the median step instead of total-span duration. | Overall timestamp-derived rate (148.12 Hz) disagrees with median-step rate (249.19 Hz).`
- Issues: `movement 1 produced 27 interval(s); protocol expected about 10 | movement 2 produced 27 interval(s); protocol expected about 10 | movement1 median duration is 5.51 s instead of the expected 8 s | movement2 median duration is 5.89 s instead of the expected 8 s | Timestamp has 9 jump(s) larger than 5x the median step; windowing uses the median step instead of total-span duration. | Overall timestamp-derived rate (148.12 Hz) disagrees with median-step rate (249.19 Hz).`

## left_right | LR-3-8-26-(03).csv

- File: `/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/EEG_LR/LR-3-8-26-(03).csv`
- Shape: `(113519, 24)`
- Sample-rate used: `249.186 Hz`
- Marker counts: `{0: 113483, 1: 9, 2: 9, 3: 9, 4: 9}`
- Movement 1 intervals: `9`
- Movement 2 intervals: `9`
- Baseline median duration: `49.93 s`
- Movement 1 median duration: `8.71 s`
- Movement 2 median duration: `8.34 s`
- Sampling notes: `Timestamp has 3 jump(s) larger than 5x the median step; windowing uses the median step instead of total-span duration.`
- Issues: `movement 1 produced 9 interval(s); protocol expected about 10 | movement 2 produced 9 interval(s); protocol expected about 10 | movement2_to_movement1 rest median is 10.47 s instead of the expected 15 s | Timestamp has 3 jump(s) larger than 5x the median step; windowing uses the median step instead of total-span duration.`

## left_right | LR-3-15-26-(04).csv

- File: `/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/EEG_LR/LR-3-15-26-(04).csv`
- Shape: `(211799, 24)`
- Sample-rate used: `249.246 Hz`
- Marker counts: `{0: 211719, 1: 20, 2: 21, 3: 19, 4: 20}`
- Movement 1 intervals: `20`
- Movement 2 intervals: `19`
- Baseline median duration: `45.71 s`
- Movement 1 median duration: `8.70 s`
- Movement 2 median duration: `8.55 s`
- Sampling notes: `Timestamp has 35 jump(s) larger than 5x the median step; windowing uses the median step instead of total-span duration. | Overall timestamp-derived rate (180.45 Hz) disagrees with median-step rate (249.25 Hz).`
- Issues: `unexpected marker transition 2->2 between event 74 and 75 | unexpected marker transition 2->4 between event 75 and 76 | marker 2 appeared without a matching prior marker 1 at sample 198441 | marker 4 appeared without a matching prior marker 3 at sample 200548 | movement 1 produced 20 interval(s); protocol expected about 10 | movement 2 produced 19 interval(s); protocol expected about 10 | Timestamp has 35 jump(s) larger than 5x the median step; windowing uses the median step instead of total-span duration. | Overall timestamp-derived rate (180.45 Hz) disagrees with median-step rate (249.25 Hz).`