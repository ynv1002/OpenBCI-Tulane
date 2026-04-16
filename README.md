# BCI Project

This repository now contains two different generations of work:

- the current OpenBCI EEG/jaw analysis and testing stack in `analysis/`
- older EMG-only pipeline and experiments in `emg_pipeline/`, `scripts/`, and `experiments/`

If a new Codex session starts here, the safest place to orient first is:

1. this root README
2. `analysis/README.md`
3. the README in the specific analysis subfolder you want to touch

## Current Project State

The active project direction is a hybrid BCI control path with four runtime states:

- `LEFT`
- `RIGHT`
- `JAW`
- `REST`

Current status at a glance:

- Jaw click detection is the strongest reusable path right now.
  - Artifact: `analysis/outputs/realtime_clench_model.pkl`
  - Replay summary: `analysis/outputs/jaw_click_best_strategy.md`
  - Current best replay strategy is `binary_clench_threshold` with test weighted event-F1 about `0.767`.
- Event alignment on Yaniv EEG left/right runs is good enough to support event-centered experimentation.
  - Validation summary: `analysis/lr_event_validation/outputs/overall_summary.csv`
  - Overall kept-event alignment is `708 / 802 = 88.3%` inside trusted marker-defined blocks.
- Event-level `LEFT` vs `RIGHT` classification is still weak.
  - Baseline summary: `analysis/lr_event_classifier/outputs/summary.md`
  - Best first-pass pooled result is Logistic Regression at about `0.525` accuracy / `0.524` macro-F1.
  - Adding mu/beta plus asymmetry did not materially improve it.
- Clean window-based `LEFT` vs `RIGHT` EEG direction also remains weak cross-session.
  - Clean retrain summary: `analysis/outputs/eeg_direction_clean_cross_session.md`
  - Best pooled result is Logistic Regression at about `0.484` accuracy / `0.480` macro-F1.
  - The saved artifact exists as a benchmark, not as a trusted production direction model.
- A hybrid tester GUI now exists and extends the earlier jaw-only visualization path.
  - Entry point: `analysis/run_hybrid_bci_tester.py`
  - It integrates jaw detection, event-gated left/right direction checks, replay mode, live mode, and structured logging.
- Ben and Yaniv LRJ now support two paths for the same-code `LEFT` / `RIGHT` / `JAW` count-ramp protocol.
  - Shared offline benchmark: `analysis/run_lrj_offline_benchmark.py`
  - Shared data core: `analysis/lrj_dataset.py`
  - Subject review wrappers: `analysis/ben-lrj-review/` and `analysis/yaniv-lrj-review/`
  - The LRJ files now form a shared offline benchmark track, but they are still not default live/runtime artifacts.
- A new stepwise offline benchmark now exists for slow, stage-gated jaw + EEG evaluation with explicit file contracts.
  - Entry point: `analysis/run_stepwise_benchmark.py`
  - Registry: `analysis/stepwise_protocol_registry.py`
  - Suite logic: `analysis/stepwise_benchmark.py`
  - It keeps `jaw`, `EEG_LR`, and `LRJ` linked but not prematurely pooled.

## Dataset Trust Map

These trust decisions matter and should not be silently changed in future prompts.

### Highest-trust EEG left/right sessions

From `/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/EEG_LR`:

- `LR-2-27-26-(01).csv`
- `LR-3-15-26-(04).csv`

Use these first for:

- clean cross-session EEG direction benchmarks
- event-level `LEFT` vs `RIGHT` experiments
- replay testing in the hybrid GUI

### Lower-consistency EEG left/right sessions

Also in Yaniv `EEG_LR`:

- `LR-3-8-26-(02).csv`
- `LR-3-8-26-(03).csv`

These are kept for:

- robustness testing
- transfer stress tests
- event-alignment review

Do not use them as the first training set for clean model claims unless the prompt explicitly says to widen the dataset.

### Jaw data

From `/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/EMG_JvsN`:

- this is the main source for jaw click training and replay evaluation
- the current live/replay jaw detector path is built around this family

### Dalin one-off datasets

- `analysis/dalin-lr-test/`
  - clean `LR(6)` count benchmark
  - useful for frozen event-counter tuning and sanity checks
- `analysis/dalin-test/`
  - corrupted LRJ run
  - frozen as a debugging artifact only
  - not a trusted training dataset

### Ben and Yaniv LRJ one-offs

- `../OPENBCI_runs/Ben/Ben-LRJ(1-6)-4:9.csv`
- `../OPENBCI_runs/Yaniv/Yaniv-LRJ(6)-4:7.csv`
- both follow the same LRJ formula:
  - `1 = LEFT`
  - `2 = RIGHT`
  - `3 = JAW`
  - six same-code start/end intervals per label
  - expected within-label counts `1..6`
- together they now form the shared offline LRJ benchmark track
- they still should not be treated as default runtime/live artifacts without a later validation step

## Active Entry Points

Install dependencies first:

```bash
pip install -r requirements.txt
```

Most useful current entry points:

### Hybrid tester

Replay mode:

```bash
python3 analysis/run_hybrid_bci_tester.py --mode REPLAY --replay-csv "/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/EEG_LR/LR-2-27-26-(01).csv"
```

Live mode:

```bash
python3 analysis/run_hybrid_bci_tester.py --mode LIVE --board cyton --serial-port "<your-port>"
```

### Jaw model

Train/export:

```bash
python3 -m analysis.realtime_clench_detector train
```

Replay:

```bash
python3 -m analysis.realtime_clench_detector replay --visualize --csv <jaw_csv>
```

Live:

```bash
python3 -m analysis.realtime_clench_detector live --serial-port /dev/cu.usbserial-XXXX
```

### EEG event validation

```bash
python3 analysis/lr_event_validation/run_yaniv_lr_validation.py --input-dir "/Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/EEG_LR"
```

### Event-level `LEFT` vs `RIGHT` experiments

```bash
python3 analysis/lr_event_classifier/run_lr_event_classifier.py
python3 analysis/lr_event_classifier/run_lr_event_feature_comparison.py
```

### Clean window-based `LEFT` vs `RIGHT` retrain

```bash
python3 analysis/run_eeg_direction_clean_cross_session.py
```

### LRJ one-off reviews

Shared offline benchmark:

```bash
python3 analysis/run_lrj_offline_benchmark.py
```

Stepwise jaw + EEG benchmark:

```bash
python3 analysis/run_stepwise_benchmark.py
```

Ben:

```bash
python3 analysis/ben-lrj-review/run_ben_lrj_review.py --csv "../OPENBCI_runs/Ben/Ben-LRJ(1-6)-4:9.csv"
```

Yaniv:

```bash
python3 analysis/yaniv-lrj-review/run_yaniv_lrj_review.py --csv "../OPENBCI_runs/Yaniv/Yaniv-LRJ(6)-4:7.csv"
```

## Key Artifacts

These files are the main reusable outputs right now:

- `analysis/outputs/realtime_clench_model.pkl`
  - current jaw artifact used by replay/live detector code and the hybrid tester
- `analysis/outputs/clean_left_right_window_model.pkl`
  - clean cross-session window-direction benchmark artifact
  - currently too weak to treat as a final production direction stage
- `analysis/outputs/lrj_benchmark_summary.md`
  - shared Ben/Yaniv offline LRJ benchmark summary
- `analysis/outputs/lrj_interval_dataset.csv`
  - canonical LRJ interval dataset with labels, expected counts, observed counts, and timing
- `analysis/outputs/stepwise_benchmark/`
  - stage-gated jaw + EEG benchmark outputs driven by the explicit file-contract registry
- `analysis/outputs/hybrid_bci_tester/`
  - structured run logs for replay/live hybrid GUI sessions
- `analysis/lr_event_validation/outputs/`
  - event-to-marker alignment audit on Yaniv EEG_LR
- `analysis/lr_event_classifier/outputs/`
  - event-level `LEFT` vs `RIGHT` feature tables and cross-session results
- `analysis/ben-lrj-review/outputs/`
  - Ben LRJ structural audit and diagnostic count review outputs
- `analysis/yaniv-lrj-review/outputs/`
  - Yaniv LRJ structural audit and diagnostic count review outputs

## Project Layout

- `analysis/`
  - active EEG and jaw modeling, audits, one-off dataset reviews, and the hybrid tester
- `bci_pipeline/`
  - small runtime decoding helpers used by the analysis-side tester
- `emg_pipeline/`
  - older EMG offline pipeline package
- `experiments/`
  - older, more isolated EMG experiments and debug viewers
- `scripts/`
  - older EMG-oriented CLI helpers
- `results/`
  - older offline EMG results

## Important Conventions

- Most OpenBCI analysis code assumes `250 Hz` unless a script explicitly says otherwise.
- Marker mappings are trusted for Yaniv `EEG_LR`:
  - `1 = LEFT start`
  - `2 = LEFT stop`
  - `3 = RIGHT start`
  - `4 = RIGHT stop`
- Marker mappings for the Ben/Yaniv LRJ one-off reviews are different:
  - `1 = LEFT`
  - `2 = RIGHT`
  - `3 = JAW`
  - each label uses same-code start/end pairs and an expected count ramp `1..6`
- For clean model claims, prefer the two high-trust Yaniv sessions first.
- Do not silently promote the Dalin LRJ run into a trusted labeled dataset.
- Do not silently treat the LRJ benchmark track as a live/runtime artifact path.
- Do not retrain models just because artifacts exist nearby; check the local README first to see whether an artifact is benchmark-only or intended for reuse.

## Legacy Work

The original top-level EMG pipeline is still here and still runnable:

```bash
python3 scripts/run_pipeline.py --csv <PATH_TO_CSV> --outdir results/experiment1
```

That path is useful for historical comparison, but it is not the center of the current EEG/jaw control work. If a future prompt is about the active BCI control path, start in `analysis/`, not in the older EMG folders.
