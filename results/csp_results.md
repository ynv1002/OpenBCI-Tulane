# CSP Motor-Imagery Baseline Review

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