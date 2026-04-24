# Results

This folder holds the curated results summaries for the final branch.

Best files to open first:

1. `stepwise_benchmark/summary.md`
2. `jaw_results.md`
3. `left_right_lr_results.md`
4. `left_right_lrj_results.md`
5. `left_right_windowed_results.md`

Current files:

- `stepwise_benchmark/`
- `jaw_results.md`
- `left_right_lr_results.md`
- `left_right_relevant_files.csv`
- `left_right_lrj_results.md`
- `left_right_windowed_results.md`
- `spectral_results.md`
- `csp_results.md`
- `eegnet_results.md`

Interpretation:

- `stepwise_benchmark/`
  - main offline hybrid benchmark spine
- `jaw_results.md`
  - strongest jaw replay result
- `left_right_lr_results.md`
  - current LR coverage and scope decision
- `left_right_lrj_results.md`
  - related LRJ benchmark summary with the extra jaw event included
- `left_right_windowed_results.md`
  - honest weak-window benchmark for left/right
- `spectral_results.md`, `csp_results.md`, `eegnet_results.md`
  - failed-iteration / alternative-model evidence

Current headline interpretation:

- jaw is the strongest branch
- left/right shows a real but weaker offline signal
- LRJ is related supporting hybrid evidence, but it is not identical to the main LR scorecard
- several alternative model families were tested and did not become the chosen runtime direction
