# Stepwise Benchmark

This folder holds the curated summaries from the stepwise jaw + EEG benchmark.

Start with:

1. `summary.md`
2. `jaw_stage1/summary.md`
3. `jaw_stage2/summary.md`
4. `eeg_stage12_main/summary.md`
5. `eeg_stage12_stress/summary.md`
6. `hand_lrj_decode/summary.md`
7. `jaw_stage3_lrj/summary.md`
8. `eeg_stage3/summary.md`

Current files:

- `summary.md`
- `protocol_registry.csv`
- `jaw_stage1/summary.md`
- `jaw_stage2/summary.md`
- `jaw_stage3_lrj/summary.md`
- `hand_lrj_decode/summary.md`
- `eeg_stage12_main/summary.md`
- `eeg_stage12_stress/summary.md`
- `eeg_stage3/summary.md`

Purpose:

- `summary.md`
  - top-level offline hybrid benchmark summary
- `jaw_stage1`
  - jaw event-detection result
- `jaw_stage2`
  - jaw `HOLD` vs `REPEATED` result
- `jaw_stage3_lrj`
  - LRJ jaw exact-count result
- `hand_lrj_decode`
  - LRJ hand exact-count result
- `eeg_stage12_main`
  - high-trust LR validation summary
- `eeg_stage12_stress`
  - lower-consistency LR stress summary
- `eeg_stage3`
  - current left/right stage-gate outcome
