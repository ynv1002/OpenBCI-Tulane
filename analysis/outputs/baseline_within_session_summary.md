# Baseline Within-Session Summary

- Coarse tasks use 2.0 s windows with 0.5 overlap.
- Jaw 4-state uses the existing event-scale windowing so onset/offset labels remain meaningful.
- BASELINE windows are excluded from the coarse tasks to keep label targets aligned with the requested task definitions.

- Output CSV: `/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/outputs/baseline_within_session_summary.csv`

## Best Overall Run

- Dataset: `jaw`
- Session: `HR-2-27-26-(01).csv`
- Task: `jaw_vs_rest`
- Accuracy: `0.992`
- Macro-F1: `0.991`
- Notes: `strong separation`

## Average By Task

| task | accuracy | macro_f1 | precision | recall |
| --- | --- | --- | --- | --- |
| jaw_vs_rest | 0.8140751122934381 | 0.778441570619385 | 0.7822378975487781 | 0.7896331358318299 |
| left_vs_right | 0.6654011187983313 | 0.62799340476727 | 0.7212388572905094 | 0.6932908681765388 |
| active_vs_rest | 0.7301624514018255 | 0.612081424451726 | 0.6430572521557792 | 0.6118039202686903 |
| jaw_4state | 0.771350993086159 | 0.47921863782856283 | 0.49452852166224753 | 0.47806459204625024 |
| left_vs_right_vs_rest | 0.7248278024295786 | 0.4731666839666603 | 0.47717366067153577 | 0.4904906875922268 |

## Average By Dataset

| dataset | accuracy | macro_f1 | precision | recall |
| --- | --- | --- | --- | --- |
| jaw | 0.7927130526897984 | 0.628830104223974 | 0.6383832096055129 | 0.63384886393904 |
| EEG | 0.7067971242099117 | 0.5710805043952187 | 0.6138232567059415 | 0.5985284920124854 |

## Recommendations

- Jaw: jaw is the strongest simple control signal; 4-state jaw decoding is notably weaker than binary jaw vs rest.
- EEG direction: EEG direction is only moderate within-session.
- EEG activity: EEG activity is the weakest role and should not be the first-stage gate without adaptation.