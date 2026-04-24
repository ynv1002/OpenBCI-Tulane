# Professor Evidence Map

This file freezes the current evidence chain for the professor-facing branch.

The goal is not to claim that the repo has one universal BCI model.
The goal is to show, clearly, what the project built, what worked better, what stayed weak, and which files support each statement.

## 1. Main Story To Support

Recommended top-line narrative:

- We built a hybrid BCI system with separate `jaw` and `left/right` control branches.
- The system includes a guided live GUI that starts with baseline and testing, then transitions into a game.
- The jaw branch is the stronger and more reusable branch.
- The left/right branch is weaker and should be presented as protocol-specific and session-calibrated rather than universally solved.
- We explored several model families and benchmarking paths, including ones that did not improve the final story.

## 2. Claim: We Built A Hybrid BCI GUI/Game System

Primary code evidence:

- `gui_game_code/run_bci_tracking_game.py`
- `gui_game_code/bci_tracking_game.py`
- `gui_game_code/bci_game_runtime.py`
- `gui_game_code/bci_session_flow.py`
- `gui_game_code/bci_tracking_game.md`

What these files support:

- the current deliverable is the tracking-game GUI, not the older EMG-only viewer
- the system has `LIVE` and `REPLAY` modes
- the live flow includes participant setup, `45 second baseline`, guided collection, session adaptation, `Session Ready`, review, and gameplay

Strongest references:

- `gui_game_code/bci_tracking_game.md`
- `gui_game_code/bci_tracking_game.py`

Key details:

- `gui_game_code/bci_tracking_game.md` explicitly documents:
  - `45 second baseline`
  - guided collection
  - session-local adaptation
  - `Session Ready`
  - `Review + Start Game`
  - gameplay review
- `gui_game_code/bci_tracking_game.py` contains the actual UI/session-phase logic for baseline, adaptation, readiness, and game start.

## 3. Claim: This Is A Hybrid System, Not One All-Encompassing Model

Primary evidence:

- `gui_game_code/bci_game_runtime.py`
- `gui_game_code/hybrid_bci_tester.py`
- `report/data_modeling_report_draft.md`

What these files support:

- the runtime loads separate jaw and direction artifacts
- the hand branch and jaw branch are kept separate
- the current project should be described as a multi-branch hybrid system rather than one universal decoder

Key details:

- `gui_game_code/bci_game_runtime.py` loads:
  - `models/realtime_clench_model.pkl`
  - `models/clean_left_right_window_model.pkl`
- `report/data_modeling_report_draft.md` already frames the work as separate scientific questions for jaw and `LEFT/RIGHT`.

## 4. Claim: Jaw Is The Stronger Branch

Primary result evidence:

- `results/stepwise_benchmark/summary.md`
- `results/stepwise_benchmark/jaw_stage1/summary.md`
- `results/stepwise_benchmark/jaw_stage2/summary.md`
- `results/jaw_results.md`
- `analysis/realtime_clench_detector.py`

What these files support:

- jaw event detection clears the first stage cleanly
- jaw `HOLD` vs `REPEATED` classification is strong in the current benchmark
- the jaw artifact is reused by the live/runtime code

Key numbers:

- `jaw_stage1/summary.md`
  - best strategy: `binary_clench_threshold`
  - test weighted event-F1: `0.767`
  - precision: `0.793`
  - recall: `0.742`
- `jaw_stage2/summary.md`
  - gate pass: `True`
  - best macro-F1: `1.000`

Runtime evidence:

- `analysis/realtime_clench_detector.py`
- `gui_game_code/bci_game_runtime.py`

Important nuance:

- the jaw replay evaluation uses approximate onset neighborhoods from derived event labels
- this is strong evidence, but it is not the same thing as exact physiological onset ground truth

## 5. Claim: Jaw Is The Best Candidate For Cross-Session / Cross-Context Reuse

Primary evidence:

- `models/realtime_clench_model.pkl`
- `results/jaw_results.md`
- `analysis/realtime_clench_detector.py`
- `gui_game_code/bci_game_runtime.py`
- `gui_game_code/bci_tracking_game.md`

What these files support:

- the jaw branch has a saved artifact that is reused across replay/live/runtime code
- the game/runtime branch adapts thresholds and gating, but reuses the jaw model rather than retraining a new jaw model every run

Recommended wording:

- say the jaw branch is the most reusable branch in the current system
- do not say it is a universally proven production decoder

## 6. Claim: Left/Right Is Weaker And Should Be Treated As Dataset-Specific

Primary evidence:

- `results/stepwise_benchmark/summary.md`
- `results/stepwise_benchmark/eeg_stage12_main/summary.md`
- `results/stepwise_benchmark/eeg_stage12_stress/summary.md`
- `results/stepwise_benchmark/eeg_stage3/summary.md`
- `results/left_right_lr_results.md`
- `results/left_right_windowed_results.md`
- `analysis/run_eeg_direction_clean_cross_session.py`
- `gui_game_code/bci_game_runtime.py`
- `gui_game_code/bci_session_flow.py`

What these files support:

- left/right event alignment is usable, but weaker than jaw
- stress-session performance is worse than main-session performance
- the clean hand benchmark is weak
- the runtime depends on baseline and session-local calibration/adaptation rather than one universally reliable left/right model

Key numbers:

- `eeg_stage12_main/summary.md`
  - inside-block fraction: `0.962`
- `eeg_stage12_stress/summary.md`
  - inside-block fraction: `0.808`
- `left_right_lr_results.md`
  - strict main LR pooled accuracy / macro-F1: `0.525 / 0.524`
  - expanded all-LR pooled accuracy / macro-F1: `0.613 / 0.613`
- `left_right_windowed_results.md`
  - winning pooled macro-F1: `0.480`
  - recommendation: `not strong enough to reuse directly`

Recommended wording:

- say the left/right branch shows a real but fragile offline signal
- say it remains session-sensitive and weaker than jaw
- do not say we have a fully generalized left/right control model

## 7. Claim: Hand/Left-Right Is Not One Universal Model

Primary evidence:

- `analysis/run_eeg_direction_clean_cross_session.py`
- `gui_game_code/bci_game_runtime.py`
- `gui_game_code/bci_session_flow.py`
- `gui_game_code/bci_tracking_game.md`

What these files support:

- the clean hand benchmark explicitly uses `45 seconds` of calibration
- the game/runtime applies session-local adaptation after baseline and guided collection
- v1 adaptation is threshold/gating adaptation, not full retraining

Recommended wording:

- say the system uses a stored hand artifact plus session calibration/adaptation
- say the hand branch is run-to-run sensitive
- avoid saying the whole project uses one all-encompassing model

## 8. Claim: We Explored Multiple Modeling Paths, Including Failed Iterations

Primary evidence:

- `results/left_right_lr_results.md`
- `results/spectral_results.md`
- `results/csp_results.md`
- `results/eegnet_results.md`
- `results/left_right_windowed_results.md`
- `results/left_right_lrj_results.md`

What these files support:

- the project did not just try one model
- several feature/model families were explored
- many of those variants did not improve the practical story enough to become the chosen runtime direction

Useful result readouts:

- time-domain combo baseline
  - `0.628` pooled accuracy / macro-F1 in `spectral_results.md`
- spectral baselines
  - around `0.508`
- CSP baselines
  - around `0.546`
- EEGNet
  - around `0.629 / 0.620`
- clean window cross-session LR benchmark
  - around `0.484 / 0.480`
- shared LRJ benchmark
  - movement interval macro-F1 best around `0.642`
  - joint interval exact-match remains low around `0.167`

Important caution:

- some of these comparisons use different evaluation sets and should not be collapsed into one pooled headline score
- they are best used as evidence of explored directions and relative strength, not as one single scorecard

## 9. Claim: Offline Vs Live Is A Real Distinction In This Repo

Primary evidence:

- `gui_game_code/bci_tracking_game.md`
- `gui_game_code/run_bci_tracking_game.py`
- `results/stepwise_benchmark/summary.md`
- `live_runs/examples/20260415_103538_live_cyton_Testing-1/`
- `live_runs/examples/20260415_103646_live_cyton_Testing-2/`
- `live_runs/examples/20260415_104152_live_cyton_3/`

What these files support:

- the repo has an offline benchmark layer and a separate live demo layer
- live runs create full forensic logging bundles
- the live story should be shown as real runtime evidence, not substituted with offline benchmarks

Live run bundle evidence:

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

Important caution:

- three representative live runs are included rather than narrowing the package to one showcase run
- one later live folder appears empty and should not be used as showcase evidence

## 10. Claim: The Dataset Story Is Structured, Not Ad Hoc

Primary evidence:

- `analysis/stepwise_protocol_registry.py`
- `results/stepwise_benchmark/protocol_registry.csv`
- `report/dataset_table.md`

What these files support:

- the project has an explicit file-contract registry
- files are separated by protocol family and trust tier
- this supports a more data-science-style methods section

Recommended wording:

- jaw, LR, LRJ, and diagnostic files are intentionally separated
- the repo should not be described as one pooled dataset

## 11. Claims We Should Avoid

Do not claim:

- that the repo has one single all-encompassing model trained over all protocols
- that left/right is solved for reliable general live control
- that LR, LRJ, jaw HR, and diagnostic Dalin files all belong to the same training story
- that live success is already fully established just because live logs exist

Safer alternatives:

- hybrid system with separate branches
- jaw is currently stronger
- hand branch remains weaker and session-sensitive
- offline benchmarks are clear; live evidence exists and is being curated

## 12. Best Current Evidence Spine For The Clean Branch

If we keep the branch narrow, the best evidence spine is:

- `gui_game_code/bci_tracking_game.py`
- `gui_game_code/bci_game_runtime.py`
- `gui_game_code/bci_session_flow.py`
- `gui_game_code/run_bci_tracking_game.py`
- `analysis/stepwise_protocol_registry.py`
- `analysis/run_stepwise_benchmark.py`
- `results/stepwise_benchmark/summary.md`
- `results/jaw_results.md`
- `results/left_right_lr_results.md`
- `results/left_right_lrj_results.md`
- `results/left_right_windowed_results.md`
- selected live run folders from `live_runs/examples/`

This is enough to support:

- system architecture
- dataset logic
- chosen runtime branches
- stronger jaw result
- weaker hand result
- failed iterations
- live vs offline distinction
