# GUI And Game Code

This folder is a professor-facing mirror of the current hybrid GUI/game deliverable.

The canonical runtime source is still `../analysis/`. Keep this folder synchronized with `analysis/` if it is shown or submitted.

Entry point:

- `run_bci_tracking_game.py`

Current files:

- `run_bci_tracking_game.py`
- `bci_tracking_game.py`
- `bci_game_runtime.py`
- `bci_session_flow.py`
- `bci_tracking_game.md`
- `hybrid_bci_tester.py`
- `jaw_trigger_rules.py`
- `realtime_decoder.py`

Notes:

- this is the newer hybrid GUI/game path only
- the older EMG-only GUI is intentionally excluded
- `hybrid_bci_tester.py` is included because the tracking-game runtime still depends on it
- `realtime_decoder.py` is included as a lightweight runtime helper dependency
- the current runtime uses saved artifacts plus session calibration/adaptation; it does not do full weight retraining during the GUI session
- runtime artifacts live in `../models/`
- new logs now land in `../live_runs/all_runs/`

Runtime flow:

1. participant/session setup
2. `45 second` baseline
3. guided `LEFT`, `RIGHT`, `JAW tap`, and `HOLD` collection in count-major order: `L1/R1/J1/Hold1` through `L6/R6/J6/Hold6`
4. session-local adaptation
5. gameplay
