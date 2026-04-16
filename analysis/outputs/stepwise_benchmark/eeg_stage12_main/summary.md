# EEG Stage 1/2 Validation: main

- Files: `LR-2-27-26-(01).csv, LR-3-15-26-(04).csv`
- Kept events inside blocks: `375`
- Kept events outside blocks: `15`
- Overall inside-block fraction: `0.962`

| file | valid_block_count | left_block_count | right_block_count | kept_events_inside_blocks | kept_events_outside_blocks | total_kept_events | inside_block_fraction | excluded_early_peak_count | count_channels | audit_issue_count | event_alignment_note | count_reasonableness_note | trust_tier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LR-2-27-26-(01).csv | 23 | 12 | 11 | 143 | 7 | 150 | 0.9533333333333334 | 1 | Channel_1, Channel_3, Channel_4, Channel_6, Channel_7, Channel_8 | 3 | Event alignment is directionally usable, but outside-block events still need review before classification. | Per-block counts look broadly reasonable; median detected count is 6.0. | main |
| LR-3-15-26-(04).csv | 39 | 20 | 19 | 232 | 8 | 240 | 0.9666666666666667 | 1 | Channel_1, Channel_3, Channel_4, Channel_6, Channel_7, Channel_8 | 8 | Event alignment is directionally usable, but outside-block events still need review before classification. | Per-block counts are mixed but mostly usable; 0 block(s) had no kept events. | main |