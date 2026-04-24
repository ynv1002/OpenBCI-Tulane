# BCI Tracking Rhythm Game

## Purpose

This is the project’s closed-loop demo layer. It extends the existing live/replay tester skeleton instead of replacing it.

The live path is now a guided session:

1. Participant prompt
2. 45 second baseline
3. Guided collection
4. Session-local adaptation
5. Session Ready screen
6. Guided-session review prompt when you press `Review + Start Game`
7. Simple rhythm game
8. Gameplay review prompt

The replay path is still available for debugging and demo work.

## Architecture

- Runtime decoding and logging live in [/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/bci_game_runtime.py](/Users/yanivnaggar/Desktop/Spring%202026/IS/BCI-project/analysis/bci_game_runtime.py).
- Session flow, adaptation heuristics, and the note chart logic live in [/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/bci_session_flow.py](/Users/yanivnaggar/Desktop/Spring%202026/IS/BCI-project/analysis/bci_session_flow.py).
- The Tk app lives in [/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/bci_tracking_game.py](/Users/yanivnaggar/Desktop/Spring%202026/IS/BCI-project/analysis/bci_tracking_game.py).
- The CLI entrypoint is [/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/analysis/run_bci_tracking_game.py](/Users/yanivnaggar/Desktop/Spring%202026/IS/BCI-project/analysis/run_bci_tracking_game.py).

The decoder still keeps the branches separate:

- `jaw branch`
  - reuses the current jaw runtime artifact and click trigger
  - short jaw pulse becomes `click`
  - sustained jaw activity becomes `click + hold`
- `hand branch`
  - reuses the current event-gated left/right runtime logic
  - emits conservative `LEFT` and `RIGHT`
  - stays neutral when the signal is uncertain

## Guided Live Session

The live session uses OpenBCI-style same-code start/end markers that match the on-screen protocol:

- `1/1 = LEFT`
- `2/2 = RIGHT`
- `3/3 = JAW tap`
- `4/4 = HOLD`

Guided collection order:

- `LEFT 1..6`
- `RIGHT 1..6`
- `JAW tap 1..6`
- `HOLD x6`

In v1 the system adapts session-local thresholds and gating on top of the existing saved artifacts. It does not retrain weights yet.

## Control Mapping

- `LEFT tap`
  - accepted hand `LEFT`
- `RIGHT tap`
  - accepted hand `RIGHT`
- `JAW tap`
  - jaw `click`
- `JAW hold`
  - jaw `click + hold`

The gameplay chart is intentionally simple:

- one required note at a time
- no overlapping inputs
- `JAW hold` is the only sustained note

## Live vs Replay

`LIVE`

- starts with a participant label prompt
- runs baseline, guided collection, adaptation, a `Session Ready` screen, guided-session review, and the game
- writes markers during scripted collection
- ends with a gameplay review prompt for perceived control quality

`REPLAY`

- bypasses guided collection
- drives the same rhythm chart directly from a replay CSV
- keeps play/pause, restart, and speed controls
- is intended for decoder/game debugging rather than session adaptation

## Logging

Each run now writes a timestamped folder under [/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/live_runs/all_runs](/Users/yanivnaggar/Desktop/Spring%202026/IS/BCI-project/live_runs/all_runs).

Important files now include:

- `run_metadata.json`
- `classifier_provenance.json`
- `adaptation_summary.json`
- `participant.json`
- `session_feedback.json`
- `control_trace.csv`
- `control_events.csv`
- `game_trace.csv`
- `game_events.csv`
- `protocol_truth.csv`
- `marker_log.csv`
- `phase_log.csv`
- `session_signal_trace.csv`
- `ground_truth_segments.csv` when replay truth could be resolved

`classifier_provenance.json` records what classifier/runtime logic was used and the current runtime settings for each branch.

`adaptation_summary.json` records whether the branch stayed on base settings or used session-local tweaks.

`session_feedback.json` now stores separate guided-session and gameplay review payloads when those prompts are completed.

For live sessions, the logger now checkpoints the key forensic files during the run instead of waiting until shutdown. That means `run_metadata.json`, `phase_log.csv`, `protocol_truth.csv`, `marker_log.csv`, `participant.json`, `classifier_provenance.json`, `adaptation_summary.json`, and rolling control/game traces should still exist even if the GUI hangs before a clean stop.

## Run

Replay:

```bash
python3 analysis/run_bci_tracking_game.py \
  --mode REPLAY \
  --replay-csv "/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/Yaniv-LRJ(6)-4:7.csv"
```

Live:

```bash
python3 analysis/run_bci_tracking_game.py \
  --mode LIVE \
  --board cyton \
  --serial-port /dev/tty.usbserial-DM00XXXX
```

Headless replay smoke test:

```bash
python3 analysis/run_bci_tracking_game.py \
  --mode REPLAY \
  --headless-smoke \
  --replay-csv "/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/Yaniv-LRJ(6)-4:7.csv" \
  --replay-speed 2 \
  --calibration-sec 2
```

## Known Limitations

- Session adaptation is threshold/gating adaptation only in v1. It does not retrain model weights.
- The hand branch is still the weaker branch, so the chart will feel steadier on jaw actions than on left/right actions.
- Guided markers are logged explicitly by the app and inserted into the live board when available, but full manual live validation with the intended hardware is still required.
- Tk GUI launch could not be fully confirmed from the shell-only environment, so the desktop open/close path still needs a quick manual check.
