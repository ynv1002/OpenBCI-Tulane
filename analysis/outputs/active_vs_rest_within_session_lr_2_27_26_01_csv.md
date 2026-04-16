# ACTIVE vs REST Within-Session Experiment

- Session: `LR-2-27-26-(01).csv`
- Chronological split: first `70%` train, last `30%` test
- Output CSV: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/active_vs_rest_within_session_lr_2_27_26_01_csv.csv`
- Selected channels: `Channel_1, Channel_3, Channel_4, Channel_6, Channel_7, Channel_8`

## Results

| model_name | accuracy | macro_f1 | active_precision | active_recall | active_to_rest_errors | rest_to_active_errors |
| --- | --- | --- | --- | --- | --- | --- |
| RandomForest | 0.7815126050420168 | 0.7656060606060606 | 0.9117647058823529 | 0.5740740740740741 | 23 | 3 |
| LDA | 0.7310924369747899 | 0.6987341772151898 | 0.9230769230769231 | 0.4444444444444444 | 30 | 2 |
| LogisticRegression | 0.6722689075630253 | 0.6203680981595092 | 0.8571428571428571 | 0.3333333333333333 | 36 | 3 |

## Cross-Session Comparison

- Best within-session setup: `RandomForest` with macro-F1 `0.766` and ACTIVE recall `0.574`
- Best cross-session setup (Feb 27 -> March 15): `LDA` with macro-F1 `0.631` and ACTIVE recall `0.405`
- Macro-F1 difference (within - cross): `+0.134`
- ACTIVE recall difference (within - cross): `+0.169`