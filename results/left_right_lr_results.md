# Left/Right Time-Domain Baseline Review

## Data Used

- Primary LR family: `LR-2-27-26-(01).csv`, `LR-3-15-26-(04).csv`, `LR-3-8-26-(02).csv`, `LR-3-8-26-(03).csv`
- Related supporting LRJ family: `Ben-LRJ(1-6)-4:9.csv`, `Yaniv-LRJ(6)-4:7.csv`
- Main scorecard: the four Yaniv `LR` sessions

## Model Used

- Event-level time-domain feature baseline
- Compared `LDA` and `LogisticRegression`
- Winning expanded-LR model: `LDA`

## Split Rule

- Cross-session evaluation across full files
- Main reported score: pooled accuracy and macro-F1 across held-out sessions

## Main Result

## Recommendation
- The current LR event classifier should expand from the original two-file scorecard to all four Yaniv `LR` runs: `LR-2-27-26-(01).csv, LR-3-15-26-(04).csv, LR-3-8-26-(02).csv, LR-3-8-26-(03).csv`.
- We have also executed a `combo_lr_lrj` pass folding in the `LRJ` benchmark files to test whether training on repeating pulse-blocks improves cross-session transfer.
- The Dalin `LR` / `LRJ` files should remain diagnostic-only for now.

## Relevant Files
| filename | protocol_family | trust_tier | current_left_right_use | include_in_current_lr_event_model |
| --- | --- | --- | --- | --- |
| Ben-LRJ(1-6)-4:9.csv | LRJ | main | Separate LRJ benchmark only | no |
| Yaniv-LRJ(6)-4:7.csv | LRJ | main | Separate LRJ benchmark only | no |
| LR-2-27-26-(01).csv | LR | main | Primary LR event-model scorecard and expanded coverage | yes |
| LR-3-15-26-(04).csv | LR | main | Primary LR event-model scorecard and expanded coverage | yes |
| LR-3-8-26-(02).csv | LR | stress | Expanded LR event coverage only | yes |
| LR-3-8-26-(03).csv | LR | stress | Expanded LR event coverage only | yes |

## LR Event-Level Coverage
| evaluation_set | files_used | selected_channels | event_count | left_events | right_events | best_model | pooled_accuracy | pooled_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strict_main_lr | LR-2-27-26-(01).csv, LR-3-15-26-(04).csv | Channel_1, Channel_3, Channel_4, Channel_6, Channel_7, Channel_8 | 375 | 183 | 192 | LogisticRegression | 0.525 | 0.524 |
| expanded_all_lr | LR-2-27-26-(01).csv, LR-3-15-26-(04).csv, LR-3-8-26-(02).csv, LR-3-8-26-(03).csv | Channel_1, Channel_3, Channel_7, Channel_8 | 708 | 355 | 353 | LDA | 0.613 | 0.613 |
| combo_lr_lrj | Ben-LRJ(1-6)-4:9.csv, LR-2-27-26-(01).csv, LR-3-15-26-(04).csv, LR-3-8-26-(02).csv, LR-3-8-26-(03).csv, Yaniv-LRJ(6)-4:7.csv | Channel_1, Channel_3, Channel_7, Channel_8 | 771 | 384 | 387 | LDA | 0.628 | 0.628 |

### Per-File LR Validation
| file | inside_block_fraction | kept_events_inside_blocks | kept_events_outside_blocks | count_channels |
| --- | --- | --- | --- | --- |
| LR-2-27-26-(01).csv | 0.953 | 143 | 7 | Channel_1, Channel_3, Channel_4, Channel_6, Channel_7, Channel_8 |
| LR-3-15-26-(04).csv | 0.967 | 232 | 8 | Channel_1, Channel_3, Channel_4, Channel_6, Channel_7, Channel_8 |
| LR-3-8-26-(02).csv | 0.800 | 236 | 59 | Channel_1, Channel_3, Channel_7, Channel_8 |
| LR-3-8-26-(03).csv | 0.829 | 97 | 20 | Channel_1, Channel_3, Channel_4, Channel_6, Channel_7, Channel_8 |

- Expanded all-LR run selected channels: `Channel_1, Channel_3, Channel_7, Channel_8`
- Expanded all-LR best pooled model: `LDA` with accuracy `0.613` and macro-F1 `0.613`.

### Best Expanded LR Fold Results
| train_file | test_file | accuracy | macro_f1 |
| --- | --- | --- | --- |
| LR-3-15-26-(04).csv, LR-3-8-26-(02).csv, LR-3-8-26-(03).csv | LR-2-27-26-(01).csv | 0.469 | 0.466 |
| LR-2-27-26-(01).csv, LR-3-8-26-(02).csv, LR-3-8-26-(03).csv | LR-3-15-26-(04).csv | 0.608 | 0.607 |
| LR-2-27-26-(01).csv, LR-3-15-26-(04).csv, LR-3-8-26-(03).csv | LR-3-8-26-(02).csv | 0.669 | 0.669 |
| LR-2-27-26-(01).csv, LR-3-15-26-(04).csv, LR-3-8-26-(02).csv | LR-3-8-26-(03).csv | 0.701 | 0.688 |

## Related LRJ Evidence

- `LRJ` is related supporting evidence, not the main LR scorecard
- The combo `LR + LRJ` pass reached pooled accuracy / macro-F1 `0.628 / 0.628`
- We still keep `LRJ` separate because it includes the extra hybrid jaw-event structure

## Scope Decision
- Include now in the LR event-model coverage: all four Yaniv `LR` files.
- Keep as separate supporting left/right evidence: `Ben-LRJ(1-6)-4:9.csv`, `Yaniv-LRJ(6)-4:7.csv`.
- Exclude from the report scorecard: `Dalin-LR(6)-4:7.csv`, `Dalin-LRJ(1,2)-4:7.csv`.

## Why It Was Not The Final Runtime Choice

- The signal is real but still fragile and session-sensitive
- The weakest fold is still poor enough that this should be presented honestly as a benchmark, not a solved decoder
