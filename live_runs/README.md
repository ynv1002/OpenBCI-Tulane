# Live Runs

This folder separates the live/replay evidence into two views.

- `examples/`
  - curated professor-facing run bundles
- `all_runs/`
  - the full current run history, including replay smoke tests and future game sessions

Each run bundle may include:

- `run_metadata.json`
- `classifier_provenance.json`
- `adaptation_summary.json`
- `phase_log.csv`
- `protocol_truth.csv`
- `marker_log.csv`
- `control_trace.csv`
- `control_events.csv`
- `game_trace.csv`
- `game_events.csv`
- `session_feedback.json`

New tracking-game runs now land under:

- `all_runs/`
