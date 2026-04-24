# Jaw Click Replay Summary

The replay evaluation uses approximate onset neighborhoods from derived event labels.
Marker edges are experimenter timing, not exact physiological onset truth.

- Best strategy: `binary_clench_threshold`
- Trigger config: `{'strategy_name': 'binary_clench_threshold', 'clench_probability_threshold': 0.8, 'onset_probability_threshold': 0.45, 'active_probability_threshold': 0.55, 'rearm_clench_probability_threshold': 0.5, 'cooldown_ms': 300, 'minimum_separation_ms': 300, 'smoothing_windows': 1, 'hold_suppression': True, 'require_transition_from_inactive': False, 'minimum_clench_rise': 0.0, 'minimum_envelope_uv': 0.0}`
- Train weighted event-F1: `0.612`
- Test weighted event-F1: `0.767`
- Test weighted precision: `0.793`
- Test weighted recall: `0.742`
- Test total clicks: `116`
- Test total approximate onset references: `124`
- Test extra clicks: `24`
- Test median lag (ms): `132.4`

## Strategy Comparison

| strategy_name | train_weighted_event_f1 | test_weighted_event_f1 | test_weighted_precision | test_weighted_recall | test_total_clicks | test_total_extra_clicks | test_median_lag_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary_clench_threshold | 0.612 | 0.767 | 0.793 | 0.742 | 116 | 24 | 132.4 |
| hybrid_transition | 0.293 | 0.263 | 0.714 | 0.161 | 28 | 8 | -16.1 |
| onset_threshold | 0.159 | 0.141 | 0.556 | 0.081 | 18 | 8 | -2.0 |

## Evaluation Window

- early allowance before approximate onset: `0.15 s`
- late allowance after approximate onset: `0.45 s`
