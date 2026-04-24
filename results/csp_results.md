# CSP Motor-Imagery Baseline Review

## Data Used

- Combined `LR + LRJ` event-level comparison set
- Common channels: `Channel_1, Channel_3, Channel_7, Channel_8`

## Model Used

- CSP features with `LogisticRegression`
- Compared against the time-domain event baseline

## Split Rule

- Cross-session pooled evaluation on the shared event-level benchmark

## Main Result

## Feature Dimension Guardrails
- Common Channels Active: `Channel_1, Channel_3, Channel_7, Channel_8`
- CSP Extracted Features: Dynamic based on `n_components` clamp.

## CSP vs Time-Domain Results
| evaluation_set | feature_count | event_count | best_model | pooled_accuracy | pooled_macro_f1 |
| --- | --- | --- | --- | --- | --- |
| time_domain_baseline (combo) | 32 | 771 | LDA | 0.628 | 0.628 |
| csp_components=2 (1.5s window) | 2 | 771 | LogisticRegression | 0.546 | 0.546 |
| csp_components=4 (1.5s window) | 4 | 771 | LogisticRegression | 0.546 | 0.545 |
| csp_components=6 (max) (1.5s window) | 4 | 771 | LogisticRegression | 0.546 | 0.545 |

### Interpretation
If CSP outperforms the time domain baseline, dimensional spatial filters are definitively required to untangle intent variance.

## Why It Was Not Chosen

- CSP stayed below the time-domain baseline
- It did not make left/right decoding robust enough to justify adopting it as the main path
