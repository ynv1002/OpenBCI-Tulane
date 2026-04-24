# Spectral Motor-Imagery Baseline Review

## Data Used

- Combined `LR + LRJ` event-level comparison set
- Common channels: `Channel_1, Channel_3, Channel_7, Channel_8`

## Model Used

- Spectral bandpower baselines with `LogisticRegression`
- Compared against the time-domain event baseline

## Split Rule

- Cross-session pooled evaluation on the shared event-level benchmark

## Main Result

## Feature Dimension Guardrails
- Common Channels Active: `Channel_1, Channel_3, Channel_7, Channel_8`
- Spectral v1 feature count: `12` (Base Bandpowers)
- Spectral v2 feature count: `30` (Base + Pairwise Differences)

## Spectral vs Time-Domain Results
| evaluation_set | feature_count | event_count | best_model | pooled_accuracy | pooled_macro_f1 |
| --- | --- | --- | --- | --- | --- |
| time_domain_baseline (combo) | 32 | 771 | LDA | 0.628 | 0.628 |
| spectral_v1_1.0s | 12 | 771 | LogisticRegression | 0.508 | 0.508 |
| spectral_v1_1.5s | 12 | 771 | LogisticRegression | 0.508 | 0.508 |
| spectral_v2_1.5s_asym | 30 | 771 | LogisticRegression | 0.501 | 0.501 |

### Interpretation
If `spectral_v1` outperforms the time domain baseline, the core motor imagery signal geometry is highly linearly separable.
If `spectral_v2` provides a major jump, relative left-right structure is the dominant directional component.

## Why It Was Not Chosen

- Spectral-only features stayed below the time-domain baseline
- This path did not improve the practical left/right story
