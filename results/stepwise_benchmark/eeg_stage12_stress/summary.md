# EEG Stage 1/2 Validation: stress

- Files: `LR-3-8-26-(02).csv, LR-3-8-26-(03).csv`
- Kept events inside blocks: `333`
- Kept events outside blocks: `79`
- Overall inside-block fraction: `0.808`

| file | valid_block_count | left_block_count | right_block_count | kept_events_inside_blocks | kept_events_outside_blocks | total_kept_events | inside_block_fraction | excluded_early_peak_count | count_channels | audit_issue_count | event_alignment_note | count_reasonableness_note | trust_tier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LR-3-8-26-(02).csv | 54 | 27 | 27 | 236 | 59 | 295 | 0.8 | 0 | Channel_1, Channel_3, Channel_7, Channel_8 | 6 | Event alignment is directionally usable, but outside-block events still need review before classification. | Per-block counts look broadly reasonable; median detected count is 4.0. | stress |
| LR-3-8-26-(03).csv | 18 | 9 | 9 | 97 | 20 | 117 | 0.8290598290598291 | 0 | Channel_1, Channel_3, Channel_4, Channel_6, Channel_7, Channel_8 | 4 | Event alignment is directionally usable, but outside-block events still need review before classification. | Per-block counts look broadly reasonable; median detected count is 6.0. | stress |