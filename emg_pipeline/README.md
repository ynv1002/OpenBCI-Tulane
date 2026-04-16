# Legacy EMG Pipeline Package

This package contains the older offline EMG pipeline that powers `scripts/run_pipeline.py`.

## Current Role

This code is still useful for:

- historical comparison
- the older EMG-only dataset workflow
- supporting the original top-level CLI

Main modules:

- `io.py`: CSV loading
- `dataset.py`: dataset assembly helpers
- `preprocess.py`: label cleanup and segmentation
- `features.py`: EMG feature extraction
- `models.py`: model wrappers
- `evaluate.py`: metrics and evaluation helpers
- `cli.py`: pipeline entrypoint logic

## Important Context

This package is no longer the center of the project.

The active BCI work has moved toward:

- EEG left/right event analysis
- jaw click detection
- hybrid replay/live testing

Those newer paths live in `analysis/`.

If a future prompt is about the current control stack, read:

1. `README.md`
2. `analysis/README.md`

before changing anything under `emg_pipeline/`.
