# Yaniv LRJ Review

- CSV: `/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/Yaniv-LRJ(6)-4:7.csv`
- Output directory: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/yaniv-lrj-review/outputs`
- Fixed reporting rate: `250.0 Hz`
- Timestamp-derived rate used by audit helper: `249.260 Hz`
- Structural pass: `True`
- Marker events: `36`
- Reconstructed trials: `18`
- Trial counts by label: `{'JAW': 6, 'LEFT': 6, 'RIGHT': 6}`
- Count matches: `8/18`
- Count channels: `Channel_1, Channel_3, Channel_4, Channel_6, Channel_7, Channel_8`

## Hard Issues

- none

## Soft Warnings

- Diagnostic count overlay mismatched 10/18 trials.

## Count Mismatches

| trial_index_overall | label | trial_index_within_label | expected_count | observed_count | count_error | issues |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | LEFT | 2 | 2 | 1 | -1 | observed_count 1 vs expected_count 2 |
| 4 | LEFT | 4 | 4 | 2 | -2 | observed_count 2 vs expected_count 4 |
| 5 | LEFT | 5 | 5 | 4 | -1 | observed_count 4 vs expected_count 5 |
| 6 | LEFT | 6 | 6 | 3 | -3 | observed_count 3 vs expected_count 6 |
| 9 | RIGHT | 3 | 3 | 2 | -1 | observed_count 2 vs expected_count 3 |
| 10 | RIGHT | 4 | 4 | 3 | -1 | observed_count 3 vs expected_count 4 |
| 11 | RIGHT | 5 | 5 | 4 | -1 | observed_count 4 vs expected_count 5 |
| 12 | RIGHT | 6 | 6 | 5 | -1 | observed_count 5 vs expected_count 6 |
| 15 | JAW | 3 | 3 | 2 | -1 | observed_count 2 vs expected_count 3 |
| 18 | JAW | 6 | 6 | 5 | -1 | observed_count 5 vs expected_count 6 |

## Trial Summary

| trial_index_overall | label | trial_index_within_label | expected_count | observed_count | count_match | duration_sec | gap_from_previous_trial_sec | issues |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | LEFT | 1 | 1 | 1 | True | 1.572 | nan |  |
| 2 | LEFT | 2 | 2 | 1 | False | 2.16 | 3.924 | observed_count 1 vs expected_count 2 |
| 3 | LEFT | 3 | 3 | 3 | True | 2.964 | 3.308 |  |
| 4 | LEFT | 4 | 4 | 2 | False | 4.024 | 3.644 | observed_count 2 vs expected_count 4 |
| 5 | LEFT | 5 | 5 | 4 | False | 4.76 | 4.98 | observed_count 4 vs expected_count 5 |
| 6 | LEFT | 6 | 6 | 3 | False | 4.98 | 4.512 | observed_count 3 vs expected_count 6 |
| 7 | RIGHT | 1 | 1 | 1 | True | 1.564 | 4.24 |  |
| 8 | RIGHT | 2 | 2 | 2 | True | 2.224 | 3.66 |  |
| 9 | RIGHT | 3 | 3 | 2 | False | 2.764 | 3.46 | observed_count 2 vs expected_count 3 |
| 10 | RIGHT | 4 | 4 | 3 | False | 3.78 | 3.316 | observed_count 3 vs expected_count 4 |
| 11 | RIGHT | 5 | 5 | 4 | False | 5.152 | 4.388 | observed_count 4 vs expected_count 5 |
| 12 | RIGHT | 6 | 6 | 5 | False | 7.004 | 4.08 | observed_count 5 vs expected_count 6 |
| 13 | JAW | 1 | 1 | 1 | True | 1.576 | 4.672 |  |
| 14 | JAW | 2 | 2 | 2 | True | 2.068 | 4.044 |  |
| 15 | JAW | 3 | 3 | 2 | False | 2.852 | 3.504 | observed_count 2 vs expected_count 3 |
| 16 | JAW | 4 | 4 | 4 | True | 4.164 | 4.116 |  |
| 17 | JAW | 5 | 5 | 5 | True | 4.924 | 3.564 |  |
| 18 | JAW | 6 | 6 | 5 | False | 5.392 | 3.288 | observed_count 5 vs expected_count 6 |

## Channel Quality

| channel | status | rail_fraction | centered_std | reasons |
| --- | --- | --- | --- | --- |
| Channel_1 | safe | 0.0 | 2994.7151087903776 |  |
| Channel_2 | unsafe | 0.9999781081021913 | 877.2895943383423 | dominant exact value fraction 1.0000; rail fraction 1.0000 |
| Channel_3 | safe | 0.0 | 6869.794491788131 |  |
| Channel_4 | safe | 0.0 | 2263.9457377827725 |  |
| Channel_5 | unsafe | 0.9999781081021913 | 877.2895943383423 | dominant exact value fraction 1.0000; rail fraction 1.0000 |
| Channel_6 | safe | 0.0 | 2334.2166937571847 |  |
| Channel_7 | safe | 0.0 | 1418.5740163082173 |  |
| Channel_8 | safe | 0.0 | 1345.4455612969948 |  |

## Marker Audit Preview

| event_id | marker_code | label | time_sec | gap_from_previous_sec | pair_index | pair_role |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | LEFT | 46.712 | nan | 1 | start |
| 2 | 1 | LEFT | 48.284 | 1.572 | 1 | end |
| 3 | 1 | LEFT | 52.208 | 3.924 | 2 | start |
| 4 | 1 | LEFT | 54.368 | 2.16 | 2 | end |
| 5 | 1 | LEFT | 57.676 | 3.308 | 3 | start |
| 6 | 1 | LEFT | 60.64 | 2.964 | 3 | end |
| 7 | 1 | LEFT | 64.284 | 3.644 | 4 | start |
| 8 | 1 | LEFT | 68.308 | 4.024 | 4 | end |
| 9 | 1 | LEFT | 73.288 | 4.98 | 5 | start |
| 10 | 1 | LEFT | 78.048 | 4.76 | 5 | end |
| 11 | 1 | LEFT | 82.56 | 4.512 | 6 | start |
| 12 | 1 | LEFT | 87.54 | 4.98 | 6 | end |
| 13 | 2 | RIGHT | 91.78 | 4.24 | 1 | start |
| 14 | 2 | RIGHT | 93.344 | 1.564 | 1 | end |
| 15 | 2 | RIGHT | 97.004 | 3.66 | 2 | start |
| 16 | 2 | RIGHT | 99.228 | 2.224 | 2 | end |
| 17 | 2 | RIGHT | 102.688 | 3.46 | 3 | start |
| 18 | 2 | RIGHT | 105.452 | 2.764 | 3 | end |
| 19 | 2 | RIGHT | 108.768 | 3.316 | 4 | start |
| 20 | 2 | RIGHT | 112.548 | 3.78 | 4 | end |
| 21 | 2 | RIGHT | 116.936 | 4.388 | 5 | start |
| 22 | 2 | RIGHT | 122.088 | 5.152 | 5 | end |
| 23 | 2 | RIGHT | 126.168 | 4.08 | 6 | start |
| 24 | 2 | RIGHT | 133.172 | 7.004 | 6 | end |
| 25 | 3 | JAW | 137.844 | 4.672 | 1 | start |
| 26 | 3 | JAW | 139.42 | 1.576 | 1 | end |
| 27 | 3 | JAW | 143.464 | 4.044 | 2 | start |
| 28 | 3 | JAW | 145.532 | 2.068 | 2 | end |
| 29 | 3 | JAW | 149.036 | 3.504 | 3 | start |
| 30 | 3 | JAW | 151.888 | 2.852 | 3 | end |
| 31 | 3 | JAW | 156.004 | 4.116 | 4 | start |
| 32 | 3 | JAW | 160.168 | 4.164 | 4 | end |
| 33 | 3 | JAW | 163.732 | 3.564 | 5 | start |
| 34 | 3 | JAW | 168.656 | 4.924 | 5 | end |
| 35 | 3 | JAW | 171.944 | 3.288 | 6 | start |
| 36 | 3 | JAW | 177.336 | 5.392 | 6 | end |
