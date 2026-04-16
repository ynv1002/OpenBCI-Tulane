# BCI Runtime Helpers

This folder is small on purpose. It currently holds lightweight runtime-decoder helpers that the analysis-side tester can reuse without duplicating configuration logic.

## Current Role

- `realtime_decoder.py`
  - shared runtime configuration helpers for the hybrid BCI tester in `analysis/hybrid_bci_tester.py`
  - not the main location for offline model training

## What This Folder Is Not

- not the source of truth for dataset audits
- not the main model-training area
- not a separate GUI application

Most of the project's active experimentation still lives in `analysis/`.

If a future prompt is about:

- training or evaluating models
- trusted datasets
- event counters
- jaw click replay/live logic
- hybrid tester behavior

start with `analysis/README.md` first, then come back here only if the task touches runtime helper code directly.
