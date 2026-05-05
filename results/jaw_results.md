# Jaw Click Replay Summary

## Data Used

- Sessions: jaw-focused replay sessions from the hybrid project benchmark path
- Task: detect discrete jaw-triggered click events

## Model Used

- Saved artifact: `models/realtime_clench_model.pkl`
- Main strategy: `binary_clench_threshold`
- Runtime interpretation: one jaw artifact drives clench/onset/active/offset probabilities, then the runtime emits click, hold start, and hold end outputs
- Runtime window: `0.12 s`
- Smoothing: `0.12 s`
- Click threshold: `0.65`
- Cooldown: `50 ms`

## Split Rule

- Replay evaluation with train/test separation from the jaw benchmark pipeline
- Matching against approximate onset neighborhoods derived from event labels

## Main Result

The replay evaluation uses approximate onset neighborhoods from derived event labels.
Marker edges are experimenter timing, not exact physiological onset truth.

- Best strategy: `binary_clench_threshold`
- Trigger config: `{'strategy_name': 'binary_clench_threshold', 'clench_probability_threshold': 0.65, 'onset_probability_threshold': 0.45, 'active_probability_threshold': 0.55, 'rearm_clench_probability_threshold': 0.5, 'cooldown_ms': 50, 'minimum_separation_ms': 50, 'smoothing_windows': 1, 'hold_suppression': True, 'require_transition_from_inactive': False, 'minimum_clench_rise': 0.0, 'minimum_envelope_uv': 0.0}`
- Train weighted event-F1: `0.739`
- Test weighted event-F1: `0.835`
- Test weighted precision: `0.782`
- Test weighted recall: `0.895`
- Test total clicks: `142`
- Test total approximate onset references: `124`
- Test extra clicks: `31`
- Test median lag (ms): `56.2`

## Why It Was Chosen

- Jaw was the strongest and most reusable control branch in the project
- This is the branch that best supports the live hybrid runtime

## Strategy Comparison

| strategy_name | train_weighted_event_f1 | test_weighted_event_f1 | test_weighted_precision | test_weighted_recall | test_total_clicks | test_total_extra_clicks | test_median_lag_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary_clench_threshold_fast_window | 0.739 | 0.835 | 0.782 | 0.895 | 142 | 31 | 56.2 |
| binary_clench_threshold_original_saved_summary | 0.612 | 0.767 | 0.793 | 0.742 | 116 | 24 | 132.4 |
| hybrid_transition | 0.293 | 0.263 | 0.714 | 0.161 | 28 | 8 | -16.1 |
| onset_threshold | 0.159 | 0.141 | 0.556 | 0.081 | 18 | 8 | -2.0 |

## Evaluation Window

- early allowance before approximate onset: `0.15 s`
- late allowance after approximate onset: `0.45 s`
