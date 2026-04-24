# EEGNet Deep Learning Motor-Imagery Review

## Deep Representational Shift
- Common Channels Active: `Channel_1, Channel_3, Channel_7, Channel_8`
- Target Classes: `Left vs Right`

## EEGNet vs CSP vs Legacy Time-Domain
| evaluation_set | feature_count | event_count | best_model | pooled_accuracy | pooled_macro_f1 |
| --- | --- | --- | --- | --- | --- |
| time_domain_baseline (combo) | 32 | 771 | LDA | 0.628 | 0.628 |
| csp_components=2 (Phase 3 Baseline) | 2 | 771 | LogisticRegression | 0.546 | 0.546 |
| EEGNet-4,2 (1.5s window) | Deep Network Representation | 771 | EEGNet-4,2 | 0.629 | 0.620 |

### Interpretation
If EEGNet bridges the gap from 0.546 (CSP) toward or surpassing 0.628, the inclusion of temporal convolutions actively unmasks intentionality invisible to simple geometric bounds.