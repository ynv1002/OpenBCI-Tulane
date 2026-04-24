# Data and Modeling Report Draft

## 1. What This Project Is Trying To Do

The active goal of this project is to build a hybrid BCI control path with four runtime states:

- `LEFT`
- `RIGHT`
- `JAW`
- `REST`

At a very basic level, the project is asking two different scientific questions:

1. Can we detect jaw activity reliably enough to use it as a control signal?
2. Can we distinguish `LEFT` from `RIGHT` intention or movement reliably enough across sessions to use it as a second control signal?

Those two questions should stay separate in both the analysis and the report. Right now, the jaw branch is materially stronger than the `LEFT` versus `RIGHT` branch, so the report should not present the overall system as equally mature across all classes.

## 2. What Data We Actually Have

The repository currently treats the data in four tiers.

### 2.1 High-trust data

- Yaniv `EEG_LR` runs:
  - `LR-2-27-26-(01).csv`
  - `LR-3-15-26-(04).csv`
- Yaniv `EMG_JvsN` runs for jaw modeling

These are the files the current code treats as the main source for reusable model claims.

### 2.2 Stress-test data

- Yaniv `EEG_LR`:
  - `LR-3-8-26-(02).csv`
  - `LR-3-8-26-(03).csv`

These are useful for robustness checks, but the project intentionally does not use them for first-pass clean performance claims.

### 2.3 Offline benchmark data

- Ben `Ben-LRJ(1-6)-4:9.csv`
- Yaniv `Yaniv-LRJ(6)-4:7.csv`

These form the shared LRJ benchmark track. They are useful for offline interval and count-constrained evaluation, but they are not yet treated as default live-runtime training data.

### 2.4 Diagnostic-only data

- Dalin `LR(6)` benchmark
- Dalin LRJ debugging file

These are useful for tuning or postmortem reasoning, but they should not be treated as clean training data in the report.

## 3. How The Raw Signals Become Model Inputs

Across the active pipeline, the basic preprocessing logic is:

1. Load the OpenBCI tab-delimited CSV.
2. Assign standard channel names (`Channel_1` through `Channel_8`).
3. Estimate sampling from timestamps when available, otherwise fall back to `250 Hz`.
4. Median-center each channel within a session.
5. Convert counts to microvolts using `0.02235 uV/count`.
6. Apply a `60 Hz` notch filter.
7. Apply family-specific bandpass filtering:
   - `1-40 Hz` for `LEFT/RIGHT`
   - `20-100 Hz` for jaw

The code also computes a simple channel-quality screen:

- `safe`
- `questionable`
- `unsafe`

Channels can be excluded if they show rail saturation, near-constant values, or other obvious failures. This is a good practice and should be described clearly in the report because it prevents obviously broken channels from contaminating training.

## 4. What Counts As A Label In This Project

This is one of the most important parts of the report.

The models are not trained on one single kind of label. They use several different label sources:

### 4.1 Coarse marker-defined blocks

For `LEFT/RIGHT` runs:

- `1 -> 2` defines a `LEFT` block
- `3 -> 4` defines a `RIGHT` block

For jaw runs, the marker pairs define coarse `HOLD` versus `REPEATED` segments.

### 4.2 Derived event labels

The event-level `LEFT/RIGHT` branch does not directly classify whole blocks. Instead, it first detects peaks inside marker-defined blocks and then labels those detected events according to the block they fall inside.

This matters because the final event labels are partly derived from:

- marker timing
- signal processing
- event detection rules

So the report should state clearly that event-level labels are not purely hand-annotated ground truth.

### 4.3 Derived jaw state labels

The jaw branch builds four state labels:

- `INACTIVE`
- `ONSET`
- `ACTIVE`
- `OFFSET`

Repeated jaw events are detected using smoothed envelope peaks inside coarse jaw segments, and some segments can fall back to a coarse single-event approximation if no internal peaks pass threshold.

That means the jaw labels are also partly algorithm-derived rather than independently annotated.

## 5. The Three Main Modeling Branches

### 5.1 Jaw click detection

This is currently the strongest branch.

What it does:

- builds event windows from jaw-labeled sessions
- trains logistic-regression models on those windows
- evaluates replay behavior using click-trigger rules and approximate onset matching

Current best replay result:

- best strategy: `binary_clench_threshold`
- test weighted event-F1: about `0.767`
- test weighted precision: about `0.793`
- test weighted recall: about `0.742`

Interpretation:

- This is promising enough to describe as the strongest reusable signal in the repo.
- The report should still be careful and say that replay scoring uses approximate onset neighborhoods rather than exact physiological onset annotations.

### 5.2 Event-level `LEFT` vs `RIGHT`

This branch:

- validates detected events inside trusted marker-defined blocks
- extracts a `0.50 s` event-centered window
- uses conservative time-domain features
- evaluates cross-session generalization across the two high-trust Yaniv runs

Current baseline result:

- total labeled events: `375`
- `183 LEFT`, `192 RIGHT`
- best pooled model: Logistic Regression
- pooled accuracy: about `0.525`
- pooled macro-F1: about `0.524`

Second-pass result:

- adding mu/beta spectral power and asymmetry did not materially improve performance

Interpretation:

- This branch is not yet strong enough for a confident control claim.
- The report should describe it as a diagnostic modeling path, not a solved classifier.

### 5.3 Clean window-based `LEFT` vs `RIGHT`

This branch:

- uses only the two highest-trust Yaniv sessions
- calibrates on the first `45 s` of each session
- uses `1.0 s` windows with `0.5` overlap
- uses combined time and spectral features with asymmetry
- evaluates leave-one-session-out cross-session transfer

Current result:

- winning model: Logistic Regression
- pooled accuracy: about `0.484`
- pooled macro-F1: about `0.480`

Interpretation:

- This is effectively near chance for a balanced binary task.
- The clean benchmark is useful as an honest baseline, but it is not evidence of a deployable `LEFT/RIGHT` decoder.

## 6. What The Validation Numbers Mean

Before training the event-level `LEFT/RIGHT` classifier, the repo checks whether detected events are actually landing inside the marker-defined blocks.

Current overall alignment result:

- `708 / 802 = 88.3%` of kept events land inside trusted blocks

This is a meaningful quality-control step and one of the stronger parts of the current methodology. It shows that the event detector is not completely drifting away from the marker structure.

However, `88.3%` is not perfect alignment. The report should acknowledge that some detected events still fall outside trusted blocks, especially in the noisier March 8 sessions.

## 7. What The Project Is Doing Right

There is a lot here that is methodologically sound.

### 7.1 It separates trust tiers

The project explicitly distinguishes:

- high-trust files
- stress-test files
- offline-only benchmark files
- diagnostic-only files

That is excellent practice and should absolutely stay in the report.

### 7.2 It prefers cross-session evaluation

For the main `LEFT/RIGHT` analyses, the strongest reported numbers are cross-session rather than within-session. That is the correct direction if the real goal is reuse across runs.

### 7.3 It screens bad channels

The channel-quality logic is simple, but it is better than silently training on saturated or dead channels.

### 7.4 It does not overclaim the weak direction model in code

The code already treats the clean `LEFT/RIGHT` artifact as a benchmark rather than a trusted production model. That is honest and scientifically appropriate.

### 7.5 It keeps the jaw and hand problems partly separated

This is important because the jaw branch is currently much stronger than the hand-direction branch.

## 8. Where We Need To Be Careful

This is the part we should be especially explicit about in the final report.

### 8.1 The clean `LEFT/RIGHT` dataset is very small

The main clean cross-session claims are based on only two high-trust sessions. That is enough for a first benchmark, but not enough for a strong generalization claim.

### 8.2 Some labels are algorithm-derived

The event-level labels and jaw onset references are not all independently annotated by a human. They are partly constructed from marker timing plus signal rules. That is acceptable for exploratory work, but it should be described honestly.

### 8.3 The jaw evaluation is replay-based, not full live validation

Replay performance is useful, but it is not the same as a live closed-loop test on a real board with real user timing variability.

### 8.4 Absolute dataset paths reduce reproducibility

Much of the pipeline uses hard-coded absolute paths to local folders outside the repository. That is fine for one-machine development, but it weakens reproducibility and will make the report harder for someone else to audit or rerun.

### 8.5 We should avoid system-level claims that the weakest branch does not support

Because the current `LEFT/RIGHT` performance is near chance, the report should not imply that the full four-state system is already reliable end to end.

## 8A. Threats To Validity

This study has several important limitations that should temper strong claims. First, the clean cross-session `LEFT/RIGHT` analysis is based on only two high-trust sessions, which limits how confidently we can generalize the reported performance. Second, several labels are partly derived from marker timing and signal-processing rules rather than fully independent manual annotation, so some evaluation targets are only approximate ground truth. Third, the strongest jaw results are replay-based rather than full live closed-loop validation on a real board, which means replay success may overestimate real-world runtime performance. Fourth, much of the pipeline depends on absolute external dataset paths outside the repository, which weakens reproducibility and makes independent reruns or handoff more difficult unless the same folder structure is available.

## 9. Bottom-Line Assessment

If the question is "Are we going about this in a basically correct way?", the answer is:

- `Yes` for the overall structure of the experimentation.
- `Mostly yes` for data hygiene and split discipline.
- `Not yet` for claiming strong `LEFT/RIGHT` decoding performance.

More specifically:

- the trust map is good
- the separation between clean, stress, and diagnostic data is good
- cross-session testing is good
- channel quality screening is good
- the jaw branch is genuinely promising
- the `LEFT/RIGHT` branch is still weak and should be presented as unresolved

That is not a failure. It is actually the kind of honest result that makes a technical report stronger.

## 10. Recommended Report Framing

The safest framing for the report is:

1. We built a structured analysis pipeline for a hybrid BCI problem.
2. We separated data sources by trust level and evaluated reusable versus diagnostic paths.
3. Jaw click detection currently shows the strongest reusable signal.
4. Event alignment for `LEFT/RIGHT` is good enough to support experimentation.
5. Cross-session `LEFT/RIGHT` decoding remains weak and is still an open modeling problem.

That framing is defensible and matches the current evidence.

## 11. What We Should Add Next

If we continue this report together, the next high-value additions are:

1. A one-page dataset table listing every file, subject, protocol family, trust tier, and intended use.
2. A one-page modeling table listing each model, input unit, feature set, split rule, and primary metric.
3. A short "threats to validity" section covering small sample size, derived labels, replay versus live testing, and external-path reproducibility.
4. A final conclusions section that makes strong claims only for the jaw branch and treats `LEFT/RIGHT` as ongoing work.

## 12. Practical Next Step

The next thing we should do together is convert this draft into a polished report with:

- a formal introduction
- a dataset table
- a methods section
- a results section
- a limitations section
- a conclusion

The structure is ready. The main work left is turning the current internal notes into clean report language.
