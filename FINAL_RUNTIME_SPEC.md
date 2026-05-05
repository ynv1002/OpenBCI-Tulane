# Final Runtime Specification

This file defines the intended final product behavior for the cleaned project branch.

## Runtime Goal

The project is a hybrid OpenBCI control system with fixed saved model artifacts plus session-local calibration. Raw Cyton signals are transformed into discrete control outputs:

- `LEFT`
- `RIGHT`
- `JAW click`
- `JAW hold start/end`

The runtime prioritizes usable control and honest reporting over claiming one universal BCI model.

## Canonical Entry Point

Use the analysis runtime as the source of truth:

```bash
python3 analysis/run_bci_tracking_game.py --mode LIVE --board cyton --serial-port "<your-port>"
```

Replay/debug mode uses the same runtime logic:

```bash
python3 analysis/run_bci_tracking_game.py --mode REPLAY
```

## Model Policy

The GUI does not retrain model weights during a session.

It loads:

- `models/realtime_clench_model.pkl`
  - jaw runtime artifact
  - `0.12 s` window
  - `0.12 s` smoothing
  - fast click trigger at `0.65`
  - `50 ms` cooldown
- `models/clean_left_right_window_model.pkl`
  - current left/right runtime artifact
  - weaker and more calibration-dependent

Session calibration only tunes runtime thresholds and gating settings. It does not update sklearn weights.

## Live Session Flow

The intended live flow is:

1. Participant/session prompt
2. `45 s` baseline
3. Guided collection
4. Session-local threshold adaptation
5. Session ready screen
6. Guided-session review
7. Gameplay
8. Gameplay review

Guided collection is count-major and uses the same marker code at the start and end of each active block:

| Marker | Label | Guided action |
|---:|---|---|
| 1 | `LEFT` | left hand clench |
| 2 | `RIGHT` | right hand clench |
| 3 | `JAW_TAP` | discrete jaw clench/tap |
| 4 | `HOLD` | jaw clench and hold |

Default guided order:

```text
LEFT x1, RIGHT x1, JAW x1, HOLD 1
LEFT x2, RIGHT x2, JAW x2, HOLD 2
...
LEFT x6, RIGHT x6, JAW x6, HOLD 6
```

## Jaw Interpretation

The jaw branch should be described as one jaw runtime artifact feeding multiple outputs, not separate unrelated models.

The artifact produces jaw class/probability signals including clench, onset, active, inactive, and offset information. The runtime maps those signals into:

- fast jaw click events from short clench/onset spikes
- jaw hold start when jaw activity remains engaged long enough
- jaw hold end when jaw activity releases long enough

The output layer keeps click and hold as separate commands derived from the same jaw signal stream.

## Left/Right Interpretation

The left/right branch emits discrete commands only. It uses event-gated direction decisions and stays neutral when confidence or margin is too weak.

## Evidence Policy

Strong final claims should point to:

- `models/`
- `analysis/`
- `results/`
- `report/`
- `live_runs/`

Generated folders such as `analysis/outputs/`, `presentation_outputs/`, and `live_runs/all_runs/` are useful local evidence but are not the main source of truth for code review.
