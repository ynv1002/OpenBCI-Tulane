# Jaw Stage 1: Event Detection

- Gate pass: `True`

- Train files: `HR-2-27-26-(01).csv, HR-3-8-26-(02).csv`
- Test file: `HR-3-15-26-(03).csv`
- Selected channels: `Channel_1, Channel_3, Channel_4, Channel_6, Channel_7, Channel_8`
- Best strategy: `binary_clench_threshold`
- Test weighted event-F1: `0.767`
- Test weighted precision: `0.793`
- Test weighted recall: `0.742`

## Strategy Summary

| strategy_name | test_weighted_event_f1 | test_weighted_precision | test_weighted_recall | test_total_references | test_total_clicks | test_total_extra_clicks |
| --- | --- | --- | --- | --- | --- | --- |
| binary_clench_threshold | 0.7666666666666667 | 0.7931034482758621 | 0.7419354838709677 | 124 | 116 | 24 |
| hybrid_transition | 0.2631578947368421 | 0.7142857142857143 | 0.16129032258064516 | 124 | 28 | 8 |
| onset_threshold | 0.1408450704225352 | 0.5555555555555556 | 0.08064516129032258 | 124 | 18 | 8 |