# OpenBCI EMG Replay GUI

Legacy note: this is an older EMG replay/debug viewer for `Y_EMG.csv`. Keep it for historical comparison and EMG-specific inspection, but do not confuse it with the newer hybrid EEG/jaw tester in `analysis/run_hybrid_bci_tester.py`.

This experiment is a replay/debug viewer for the validated OpenBCI-style EMG pipeline. It is not a training or evaluation script.

The viewer recomputes the processing pipeline from raw `Y_EMG.csv` at launch, then lets you replay the session over time while comparing preprocessing modes:

- `raw`
- `centered`
- `scaled`
- `filtered`

`scaled` is the recommended starting mode because it was the only non-degenerate mode in the processing audit.

## What The GUI Shows

The replay keeps three synchronized views centered on the current sample:

- current ground-truth label
- a left/right directional ticker driven by `mean_x_smooth`
- an activation gauge driven by `mean_total_activation`

It also includes:

- a recent trailing trace window for `mean_x_smooth` and `mean_total_activation`
- a full-session overview timeline with label shading
- playback controls, a time scrubber, a mode switcher, and playback speed controls

## Why This Exists

The baseline and audit established that:

- the file behaves like raw OpenBCI-style counts
- centering plus scaling was necessary
- `scaled` was the only mode where the OpenBCI thresholds behaved sensibly
- left/right separation is still weak, so manual visual inspection of label vs directional output over time is useful

This viewer is meant to make that inspection easy.

## How To Run

From the repo root:

```bash
python3 experiments/openbci_emg_replay_gui/run_gui.py \
  --csv "/Users/yanivnaggar/Desktop/Fall 2025/Independent Study/CogniSync-main/Clench_1/Y_EMG.csv" \
  --fs 250 \
  --mode scaled
```

Optional arguments:

- `--window-sec 8`
- `--speed 1.0`
- `--no-show` for a non-interactive smoke test

## What The Modes Mean

- `raw`: loaded values exactly as stored in the CSV
- `centered`: per-channel median subtraction only
- `scaled`: per-channel median subtraction plus `0.02235 uV/count`
- `filtered`: centered and scaled, then passed through the audit’s EMG-focused filter stage

The mode switcher updates the replay without recomputing everything from scratch after startup.

## What To Inspect First

1. Start in `scaled` mode and watch whether `Jleft`, `Jright`, `Jaws`, and `norm` line up with directional ticker behavior.
2. Switch to `raw` and confirm that the warning and pinned behavior match the processing audit’s saturation finding.
3. Switch to `filtered` and confirm whether the directional signal collapses as seen in the audit.
4. Compare what you see to `experiments/openbci_emg_processing_audit/outputs/mode_comparison.csv`.

## How To Compare To The Audit

The replay uses the same validated processing path as the audit:

- `load_emg_csv`
- `build_mode_signals`
- `OpenBCIEMGJoystick1DProcessor`

So the `scaled` mode in this viewer should match the audit’s `processed_scaled.csv` for key columns such as:

- `mean_x_smooth`
- `mean_total_activation`
- per-channel OpenBCI threshold outputs

If the GUI behavior does not match those files, that is a bug in the replay layer rather than in the signal-processing layer.
