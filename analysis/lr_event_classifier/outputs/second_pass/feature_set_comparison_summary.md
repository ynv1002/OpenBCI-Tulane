# Second-Pass LR Event Feature Comparison

## Setup
- Files used: `LR-2-27-26-(01).csv, LR-3-15-26-(04).csv`
- Evaluation: `two-fold leave-one-session-out cross-session`
- Labeled events: `375 total` (`183 LEFT`, `192 RIGHT`)
- Baseline config reuses the existing event-level baseline feature builder exactly.
- Spectral features use the repo-native `mu_power`, `beta_power`, and pairwise `mu_asym_*` / `beta_asym_*` features from `analysis.utils.extract_window_features`.
- CSP: skipped because there is no clean existing CSP implementation path in the repo.

## Pooled Results
| feature_config | window_sec | model_name | feature_count | baseline_feature_count | spectral_feature_count | asymmetry_feature_count | accuracy | macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_time_0p50 | 0.5 | LDA | 46 | 46 | 0 | 0 | 0.499 | 0.496 |
| baseline_time_0p50 | 0.5 | LogisticRegression | 46 | 46 | 0 | 0 | 0.525 | 0.524 |
| baseline_plus_spectral_0p50 | 0.5 | LDA | 88 | 46 | 12 | 30 | 0.496 | 0.494 |
| baseline_plus_spectral_0p50 | 0.5 | LogisticRegression | 88 | 46 | 12 | 30 | 0.523 | 0.521 |
| baseline_plus_spectral_0p75 | 0.75 | LDA | 88 | 46 | 12 | 30 | 0.501 | 0.495 |
| baseline_plus_spectral_0p75 | 0.75 | LogisticRegression | 88 | 46 | 12 | 30 | 0.507 | 0.501 |

## Interpretation
- Best baseline config: `baseline_time_0p50` / `LogisticRegression` with accuracy `0.525` and macro-F1 `0.524`.
- Best spectral config: `baseline_plus_spectral_0p50` / `LogisticRegression` with accuracy `0.523` and macro-F1 `0.521`.
- Delta vs baseline: accuracy `-0.003`, macro-F1 `-0.003`.
- The added spectral/spatial features did not materially improve the first baseline.
- Widening the event window to `0.75 s` did not help the spectral configuration.
- Event-level LEFT vs RIGHT still looks weak cross-session and likely needs a better event representation or session calibration.