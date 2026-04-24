# Shared LRJ Offline Benchmark

## Data Used

- Sessions: `Ben-LRJ(1-6)-4:9.csv`, `Yaniv-LRJ(6)-4:7.csv`
- Task family: structured hybrid protocol with `LEFT`, `RIGHT`, and `JAW`

## Model Used

- Interval/window benchmark models: `RandomForest`, `LogisticRegression`, `LDA`
- Shared selected channels: `Channel_1, Channel_3, Channel_7, Channel_8`

## Split Rule

- Leave-one-session-out across the two LRJ sessions
- Metrics reported at both the window level and the interval level

## Main Result

- Sessions: `Ben-LRJ(1-6)-4:9.csv, Yaniv-LRJ(6)-4:7.csv`
- Shared selected channels: `Channel_1, Channel_3, Channel_7, Channel_8`
- Interval count: `36`
- Window count: `232`
- Feature mode: `combined`
- With asymmetry: `True`

## Session Summary

| subject | filename | interval_count | count_matches | count_channels | soft_warnings |
| --- | --- | --- | --- | --- | --- |
| Ben | Ben-LRJ(1-6)-4:9.csv | 18 | 10 | Channel_1, Channel_3, Channel_7, Channel_8 | Diagnostic count overlay mismatched 8/18 trials. |
| Yaniv | Yaniv-LRJ(6)-4:7.csv | 18 | 8 | Channel_1, Channel_3, Channel_4, Channel_6, Channel_7, Channel_8 | Diagnostic count overlay mismatched 10/18 trials. |

## Per-Fold Results

| fold_name | model_name | movement_window_macro_f1 | count_window_macro_f1 | joint_window_exact | movement_interval_macro_f1 | count_interval_macro_f1 | joint_interval_exact |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ben-LRJ(1-6)-4:9.csv_to_Yaniv-LRJ(6)-4:7.csv | RandomForest | 0.6676478458656677 | 0.14738751814223514 | 0.16666666666666666 | 0.6638655462184874 | 0.1574074074074074 | 0.16666666666666666 |
| Ben-LRJ(1-6)-4:9.csv_to_Yaniv-LRJ(6)-4:7.csv | LogisticRegression | 0.648157707702091 | 0.1981552891739108 | 0.16666666666666666 | 0.6638655462184874 | 0.1531986531986532 | 0.16666666666666666 |
| Yaniv-LRJ(6)-4:7.csv_to_Ben-LRJ(1-6)-4:9.csv | LDA | 0.5838983581902494 | 0.1494281045751634 | 0.17692307692307693 | 0.5586635586635587 | 0.1606060606060606 | 0.16666666666666666 |
| Yaniv-LRJ(6)-4:7.csv_to_Ben-LRJ(1-6)-4:9.csv | LogisticRegression | 0.4818750954344175 | 0.19864813549024074 | 0.09230769230769231 | 0.5175213675213676 | 0.2722222222222222 | 0.16666666666666666 |
| Yaniv-LRJ(6)-4:7.csv_to_Ben-LRJ(1-6)-4:9.csv | RandomForest | 0.5060197663971249 | 0.24846529814271753 | 0.17692307692307693 | 0.44528619528619534 | 0.21031746031746032 | 0.16666666666666666 |
| Ben-LRJ(1-6)-4:9.csv_to_Yaniv-LRJ(6)-4:7.csv | LDA | 0.5279518850947422 | 0.10265795206971677 | 0.12745098039215685 | 0.6666666666666666 | 0.041666666666666664 | 0.05555555555555555 |

## Pooled LOSO Results

| model_name | movement_window_macro_f1 | count_window_macro_f1 | joint_window_exact | movement_interval_macro_f1 | count_interval_macro_f1 | joint_interval_exact |
| --- | --- | --- | --- | --- | --- | --- |
| LogisticRegression | 0.5895361380798274 | 0.19670008043470355 | 0.125 | 0.6416666666666667 | 0.2115079365079365 | 0.16666666666666666 |
| RandomForest | 0.615790409908057 | 0.2038361669231584 | 0.1724137931034483 | 0.6155969634230504 | 0.2018456583673975 | 0.16666666666666666 |
| LDA | 0.5698542483450284 | 0.12842985283141065 | 0.15517241379310345 | 0.6232323232323232 | 0.1111111111111111 | 0.1111111111111111 |

## Recommendation

- Best movement model: `LogisticRegression` with pooled interval macro-F1 `0.642`
- Best count model: `LogisticRegression` with pooled interval macro-F1 `0.212`
- Best joint model: `LogisticRegression` with pooled interval exact-match `0.167`

## Why It Was Not The Main Runtime Scorecard

- `LRJ` is valuable hybrid evidence, but it is not identical to the plain `LR` left/right benchmark
- Exact joint decode stayed weak enough that it should be treated as supporting evidence rather than the main success claim
