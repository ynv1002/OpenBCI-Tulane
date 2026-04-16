# Ben LRJ Review

- CSV: `/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Ben/Ben-LRJ(1-6)-4:9.csv`
- Output directory: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/ben-lrj-review/outputs`
- Fixed reporting rate: `250.0 Hz`
- Timestamp-derived rate used by audit helper: `249.246 Hz`
- Structural pass: `True`
- Marker events: `36`
- Reconstructed trials: `18`
- Trial counts by label: `{'JAW': 6, 'LEFT': 6, 'RIGHT': 6}`
- Count matches: `10/18`
- Count channels: `Channel_1, Channel_3, Channel_7, Channel_8`

## Hard Issues

- none

## Soft Warnings

- Diagnostic count overlay mismatched 8/18 trials.

## Count Mismatches

| trial_index_overall | label | trial_index_within_label | expected_count | observed_count | count_error | issues |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | LEFT | 4 | 4 | 2 | -2 | observed_count 2 vs expected_count 4 |
| 5 | LEFT | 5 | 5 | 2 | -3 | observed_count 2 vs expected_count 5 |
| 6 | LEFT | 6 | 6 | 5 | -1 | observed_count 5 vs expected_count 6 |
| 8 | RIGHT | 2 | 2 | 1 | -1 | observed_count 1 vs expected_count 2 |
| 12 | RIGHT | 6 | 6 | 3 | -3 | observed_count 3 vs expected_count 6 |
| 16 | JAW | 4 | 4 | 2 | -2 | observed_count 2 vs expected_count 4 |
| 17 | JAW | 5 | 5 | 3 | -2 | observed_count 3 vs expected_count 5 |
| 18 | JAW | 6 | 6 | 3 | -3 | observed_count 3 vs expected_count 6 |

## Trial Summary

| trial_index_overall | label | trial_index_within_label | expected_count | observed_count | count_match | duration_sec | gap_from_previous_trial_sec | issues |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | LEFT | 1 | 1 | 1 | True | 2.632 | nan |  |
| 2 | LEFT | 2 | 2 | 2 | True | 2.468 | 5.368 |  |
| 3 | LEFT | 3 | 3 | 3 | True | 2.468 | 4.804 |  |
| 4 | LEFT | 4 | 4 | 2 | False | 3.544 | 4.768 | observed_count 2 vs expected_count 4 |
| 5 | LEFT | 5 | 5 | 2 | False | 5.168 | 5.216 | observed_count 2 vs expected_count 5 |
| 6 | LEFT | 6 | 6 | 5 | False | 5.992 | 4.096 | observed_count 5 vs expected_count 6 |
| 7 | RIGHT | 1 | 1 | 1 | True | 2.348 | 2.664 |  |
| 8 | RIGHT | 2 | 2 | 1 | False | 2.768 | 5.712 | observed_count 1 vs expected_count 2 |
| 9 | RIGHT | 3 | 3 | 3 | True | 4.584 | 5.112 |  |
| 10 | RIGHT | 4 | 4 | 4 | True | 6.348 | 4.524 |  |
| 11 | RIGHT | 5 | 5 | 5 | True | 6.204 | 3.512 |  |
| 12 | RIGHT | 6 | 6 | 3 | False | 5.856 | 5.136 | observed_count 3 vs expected_count 6 |
| 13 | JAW | 1 | 1 | 1 | True | 5.472 | 7.076 |  |
| 14 | JAW | 2 | 2 | 2 | True | 3.064 | 5.056 |  |
| 15 | JAW | 3 | 3 | 3 | True | 4.704 | 5.124 |  |
| 16 | JAW | 4 | 4 | 2 | False | 3.244 | 9.212 | observed_count 2 vs expected_count 4 |
| 17 | JAW | 5 | 5 | 3 | False | 6.124 | 4.896 | observed_count 3 vs expected_count 5 |
| 18 | JAW | 6 | 6 | 3 | False | 5.796 | 4.952 | observed_count 3 vs expected_count 6 |

## Channel Quality

| channel | status | rail_fraction | centered_std | reasons |
| --- | --- | --- | --- | --- |
| Channel_1 | safe | 0.0 | 1335.8913334315175 |  |
| Channel_2 | unsafe | 0.6771943559723137 | 41937.302787771194 | rail fraction 0.6772 |
| Channel_3 | safe | 0.0 | 7044.175989959226 |  |
| Channel_4 | unsafe | 0.06754328214800448 | 50659.19058947122 | rail fraction 0.0675 |
| Channel_5 | unsafe | 1.0 | 0.0 | dominant exact value fraction 1.0000; centered standard deviation is 0; rail fraction 1.0000 |
| Channel_6 | unsafe | 0.9530079535951318 | 18102.264091393576 | rail fraction 0.9530 |
| Channel_7 | safe | 0.0 | 58393.395824112085 |  |
| Channel_8 | safe | 0.0 | 3265.9690128665416 |  |

## Marker Audit Preview

| event_id | marker_code | label | time_sec | gap_from_previous_sec | pair_index | pair_role |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | LEFT | 52.852 | nan | 1 | start |
| 2 | 1 | LEFT | 55.484 | 2.632 | 1 | end |
| 3 | 1 | LEFT | 60.852 | 5.368 | 2 | start |
| 4 | 1 | LEFT | 63.32 | 2.468 | 2 | end |
| 5 | 1 | LEFT | 68.124 | 4.804 | 3 | start |
| 6 | 1 | LEFT | 70.592 | 2.468 | 3 | end |
| 7 | 1 | LEFT | 75.36 | 4.768 | 4 | start |
| 8 | 1 | LEFT | 78.904 | 3.544 | 4 | end |
| 9 | 1 | LEFT | 84.12 | 5.216 | 5 | start |
| 10 | 1 | LEFT | 89.288 | 5.168 | 5 | end |
| 11 | 1 | LEFT | 93.384 | 4.096 | 6 | start |
| 12 | 1 | LEFT | 99.376 | 5.992 | 6 | end |
| 13 | 2 | RIGHT | 102.04 | 2.664 | 1 | start |
| 14 | 2 | RIGHT | 104.388 | 2.348 | 1 | end |
| 15 | 2 | RIGHT | 110.1 | 5.712 | 2 | start |
| 16 | 2 | RIGHT | 112.868 | 2.768 | 2 | end |
| 17 | 2 | RIGHT | 117.98 | 5.112 | 3 | start |
| 18 | 2 | RIGHT | 122.564 | 4.584 | 3 | end |
| 19 | 2 | RIGHT | 127.088 | 4.524 | 4 | start |
| 20 | 2 | RIGHT | 133.436 | 6.348 | 4 | end |
| 21 | 2 | RIGHT | 136.948 | 3.512 | 5 | start |
| 22 | 2 | RIGHT | 143.152 | 6.204 | 5 | end |
| 23 | 2 | RIGHT | 148.288 | 5.136 | 6 | start |
| 24 | 2 | RIGHT | 154.144 | 5.856 | 6 | end |
| 25 | 3 | JAW | 161.22 | 7.076 | 1 | start |
| 26 | 3 | JAW | 166.692 | 5.472 | 1 | end |
| 27 | 3 | JAW | 171.748 | 5.056 | 2 | start |
| 28 | 3 | JAW | 174.812 | 3.064 | 2 | end |
| 29 | 3 | JAW | 179.936 | 5.124 | 3 | start |
| 30 | 3 | JAW | 184.64 | 4.704 | 3 | end |
| 31 | 3 | JAW | 193.852 | 9.212 | 4 | start |
| 32 | 3 | JAW | 197.096 | 3.244 | 4 | end |
| 33 | 3 | JAW | 201.992 | 4.896 | 5 | start |
| 34 | 3 | JAW | 208.116 | 6.124 | 5 | end |
| 35 | 3 | JAW | 213.068 | 4.952 | 6 | start |
| 36 | 3 | JAW | 218.864 | 5.796 | 6 | end |
