# Hybrid BCI Project

This branch is the cleaned professor-facing version of the project.

The core story is:

- we built a hybrid BCI system with `LEFT`, `RIGHT`, `JAW`, and `REST`
- the live flow runs a guided baseline and calibration phase before gameplay
- jaw was the strongest and most reusable control branch
- left/right remained weaker, benefited from calibration, and stayed session-sensitive
- we tested several alternative modeling directions and kept the results even when they did not win

## Start Here

If you want the cleanest overview, open these in order:

1. `report/`
2. `results/`
3. `gui_game_code/`
4. `live_runs/examples/`

If you want the backend code after that, use:

5. `analysis/`
6. `models/`

## Repo Layout

- `report/`
  - write-up, dataset table, modeling table, and evidence map
- `results/`
  - benchmark summaries and failed-method comparisons
- `gui_game_code/`
  - professor-facing copy of the hybrid GUI/game stack
- `live_runs/`
  - curated examples plus the full run log history
- `models/`
  - saved jaw and left/right runtime artifacts
- `analysis/`
  - active backend code for the final runtime and cited analyses
- `bci_pipeline/`
  - lightweight runtime decoding helper reused by the game code

## Main Runtime

The main GUI/game entry point is:

```bash
python3 analysis/run_bci_tracking_game.py --mode LIVE --board cyton --serial-port "<your-port>"
```

Replay mode:

```bash
python3 analysis/run_bci_tracking_game.py --mode REPLAY
```

New run bundles now land under:

- `live_runs/all_runs/`

## Runtime Model Policy

The GUI session does not retrain model weights live.

The current runtime does:

- load the saved jaw artifact
- load the saved left/right artifact
- collect baseline and guided calibration data
- tune runtime thresholds and decision settings for that session
- start gameplay

So the honest description is:

- fixed saved models
- plus session calibration
- plus session-local runtime adaptation

not:

- one universal model trained on all data
- or online weight retraining during the session

## Current Result Summary

- `jaw`
  - strongest branch and best practical control signal
- `left/right LR`
  - weak but non-random offline signal
- `left/right LRJ`
  - related hybrid-protocol evidence with the extra jaw event included
- `windowed left/right`
  - weaker benchmark, kept for honesty
- `spectral`, `CSP`, `EEGNet`
  - explored and documented, but not selected

## Dataset Framing

The repo should be read as several related protocol families, not one pooled dataset.

- `LR`
  - block-based `LEFT` / `RIGHT` EEG sessions
- `HR / jaw`
  - jaw-focused sessions used for the jaw branch
- `LRJ`
  - structured hybrid sessions with `LEFT`, `RIGHT`, and `JAW`

`LRJ` is not a random one-off family. It is the closest current dataset family to the guided hybrid collection flow used by the GUI.

## Install

```bash
pip install -r requirements.txt
```
