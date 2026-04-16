# Jaw Event Labels

- Train files: `HR-2-27-26-(01).csv, HR-3-8-26-(02).csv`
- Test file: `HR-3-15-26-(03).csv`
- Selected channels: `['Channel_1', 'Channel_3', 'Channel_4', 'Channel_6', 'Channel_7', 'Channel_8']`
- Excluded channels: `Channel_2: unsafe in training sessions: HR-2-27-26-(01).csv (rail fraction 0.4969), HR-3-8-26-(02).csv (dominant exact value fraction 1.0000; rail fraction 1.0000) | Channel_5: unsafe in training sessions: HR-2-27-26-(01).csv (dominant exact value fraction 1.0000; rail fraction 1.0000), HR-3-8-26-(02).csv (rail fraction 1.0000)`
- Config: `JawEventConfig(smoothing_sec=0.25, onset_duration_sec=0.2, offset_duration_sec=0.2, minimum_active_duration_sec=0.15, minimum_peak_distance_sec=0.35, inactive_quantile=0.9, active_quantile=0.75, threshold_mix=0.35, release_threshold_mix=0.15, minimum_reference_gap_uv=0.05, peak_prominence_scale=0.25, minimum_peak_prominence_uv=0.05, manual_active_threshold_uv=None, manual_release_threshold_uv=None, repeated_fallback_mode='coarse_single_event')`

## Event State Counts

| event_label | sample_count |
| --- | --- |
| ACTIVE | 103296 |
| INACTIVE | 278096 |
| OFFSET | 15609 |
| ONSET | 15561 |

## Repeated Segment Detection

- Repeated segments: `39`
- Repeated fallback count: `9`
- Total detected repeated events: `337`
- Median detected events per repeated segment: `10.00`

| filename | coarse_segment_index | duration_sec | detection_mode | detected_peak_count | detected_event_count | fallback_used | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HR-2-27-26-(01).csv | 3 | 8.051821947097778 | repeated_peaks | 12 | 12 | False |  |
| HR-2-27-26-(01).csv | 7 | 8.132059335708618 | repeated_peaks | 12 | 12 | False |  |
| HR-2-27-26-(01).csv | 11 | 8.521210670471191 | repeated_peaks | 13 | 13 | False |  |
| HR-2-27-26-(01).csv | 15 | 8.481091976165771 | repeated_peaks | 12 | 12 | False |  |
| HR-2-27-26-(01).csv | 19 | 8.810065269470215 | repeated_peaks | 11 | 11 | False |  |
| HR-2-27-26-(01).csv | 23 | 9.207240343093872 | repeated_peaks | 13 | 13 | False |  |
| HR-2-27-26-(01).csv | 25 | 8.721804141998291 | repeated_peaks | 10 | 10 | False |  |
| HR-2-27-26-(01).csv | 29 | 10.077816009521484 | repeated_peaks | 12 | 12 | False |  |
| HR-2-27-26-(01).csv | 33 | 8.906350135803223 | repeated_peaks | 12 | 12 | False |  |
| HR-2-27-26-(01).csv | 37 | 8.818089008331299 | repeated_peaks | 12 | 12 | False |  |
| HR-2-27-26-(01).csv | 41 | 9.724771499633789 | repeated_peaks | 13 | 13 | False |  |
| HR-2-27-26-(01).csv | 45 | 8.90233826637268 | repeated_peaks | 14 | 14 | False |  |
| HR-3-8-26-(02).csv | 3 | 4.7153472900390625 | repeated_peaks | 8 | 8 | False |  |
| HR-3-8-26-(02).csv | 7 | 6.3847808837890625 | repeated_peaks | 11 | 11 | False |  |
| HR-3-8-26-(02).csv | 11 | 5.321319580078125 | repeated_peaks | 8 | 8 | False |  |
| HR-3-8-26-(02).csv | 15 | 4.9721832275390625 | repeated_peaks | 7 | 7 | False |  |
| HR-3-8-26-(02).csv | 19 | 4.7233734130859375 | repeated_peaks | 2 | 2 | False |  |
| HR-3-8-26-(02).csv | 23 | 4.598968505859375 | repeated_peaks | 8 | 8 | False |  |
| HR-3-8-26-(02).csv | 27 | 4.6029815673828125 | repeated_peaks | 8 | 8 | False |  |
| HR-3-8-26-(02).csv | 31 | 4.2979888916015625 | repeated_peaks | 7 | 7 | False |  |
