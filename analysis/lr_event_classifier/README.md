# Event-Level LEFT vs RIGHT Experiments

This folder builds the event-level `LEFT` vs `RIGHT` experiments on top of the validated Yaniv EEG_LR event counter outputs.

It answers a specific question:

- once a kept event is already known to be inside a trusted `LEFT` or `RIGHT` marker-defined block, does the event-centered EEG around that moment separate direction well enough to classify the event itself?

## Data Scope

Only the two higher-trust Yaniv runs are used for the main experiments:

- `LR-2-27-26-(01).csv`
- `LR-3-15-26-(04).csv`

The March 8 sessions are intentionally excluded from the first clean event-level classifier pass and should be treated as later robustness data.

Labels come from the enclosing trusted marker block:

- each kept in-block event becomes one labeled example
- label is `LEFT` or `RIGHT`

## What This Folder Reuses

- the frozen event counter through `analysis/lr_event_validation/`
- trusted marker-defined block boundaries
- preprocessing and feature helpers from `analysis.utils`
- the same two-fold leave-one-session-out cross-session evaluation design in both passes

## First-Pass Baseline

Run:

```bash
python3 analysis/lr_event_classifier/run_lr_event_classifier.py
```

Setup:

- centered event window: `0.50 s`
- channels: `Channel_1, Channel_3, Channel_4, Channel_6, Channel_7, Channel_8`
- conservative time-domain features only
- lightweight models: `LDA`, `LogisticRegression`

Feature family:

- RMS
- MAV
- variance
- peak absolute amplitude
- waveform length
- aggregate RMS summaries
- simple channel-vs-mean asymmetry

Current baseline summary:

- `analysis/lr_event_classifier/outputs/summary.md`

Current baseline result:

- labeled events: `375` total
- class counts: `183 LEFT`, `192 RIGHT`
- best pooled model: Logistic Regression
- pooled accuracy about `0.525`
- pooled macro-F1 about `0.524`

Interpretation:

- event-level `LEFT` vs `RIGHT` is only weakly learnable with this conservative first-pass representation

## Second-Pass Spectral Comparison

Run:

```bash
python3 analysis/lr_event_classifier/run_lr_event_feature_comparison.py
```

This keeps the same data and evaluation design, then compares:

- `baseline_time_0p50`
- `baseline_plus_spectral_0p50`
- `baseline_plus_spectral_0p75`

Spectral additions:

- `mu_power`
- `beta_power`
- pairwise `mu_asym_*`
- pairwise `beta_asym_*`

CSP remains out of scope here because there is no clean CSP implementation path in the repo.

Current second-pass summary:

- `analysis/lr_event_classifier/outputs/second_pass/feature_set_comparison_summary.md`

Current outcome:

- best baseline config: Logistic Regression at about `0.525` accuracy / `0.524` macro-F1
- best spectral config: Logistic Regression at about `0.523` accuracy / `0.521` macro-F1
- widening the window to `0.75 s` did not help

Interpretation:

- mu/beta plus asymmetry did not materially improve the first baseline
- event-level directional separation still looks weak cross-session
- the next improvement probably needs better event representation or calibration, not a larger model stack

## Outputs

Main first-pass outputs:

- `event_feature_table.csv`
- `classifier_results.csv`
- `event_predictions.csv`
- `summary.json`
- `summary.md`

Second-pass outputs under `outputs/second_pass/`:

- `feature_set_results.csv`
- `feature_set_comparison_summary.md`
- `summary.json`
- `event_feature_table_baseline_plus_spectral_0p50.csv`
- `event_feature_table_baseline_plus_spectral_0p75.csv`

## How To Use This Folder

Use this folder when the prompt is about:

- event-centered direction learnability
- feature comparisons on the high-trust Yaniv events
- whether event-level `LEFT` / `RIGHT` appears promising

Do not treat this folder as the final deployed direction classifier. Right now it is still best read as a diagnostic branch that sits between marker-aligned event extraction and the later hybrid control logic.
