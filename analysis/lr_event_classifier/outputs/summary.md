# Event-Level LEFT vs RIGHT Baseline

## Setup
- Files used: `LR-2-27-26-(01).csv, LR-3-15-26-(04).csv`
- Excluded for this first pass: `LR-3-8-26-(02).csv`, `LR-3-8-26-(03).csv`
- Selected channels: `Channel_1, Channel_3, Channel_4, Channel_6, Channel_7, Channel_8`
- Event window: `0.50 s` centered on each kept in-block event
- Features: conservative time-domain event features only
  RMS, MAV, variance, peak absolute amplitude, waveform length, aggregate RMS summaries, and simple channel-vs-mean asymmetry
- Spectral mu/beta features: not included in this first baseline
- CSP: intentionally out of scope for this first pass

## Dataset
- Total labeled events: `375`
- LEFT events: `183`
- RIGHT events: `192`
- Feature count: `46`

## Cross-Session Results
| model_name | accuracy | macro_f1 |
| --- | --- | --- |
| LDA | 0.499 | 0.496 |
| LogisticRegression | 0.525 | 0.524 |

- Best pooled model: `LogisticRegression` with accuracy `0.525` and macro-F1 `0.524`

## Best Confusion Matrix
| label | pred_LEFT | pred_RIGHT |
| --- | --- | --- |
| true_LEFT | 88 | 95 |
| true_RIGHT | 83 | 109 |

## Interpretation
- Event-level LEFT vs RIGHT separation is only weakly learnable with this first conservative baseline.
- The next step should be refining event-centered features or calibration, not jumping to a larger model stack immediately.