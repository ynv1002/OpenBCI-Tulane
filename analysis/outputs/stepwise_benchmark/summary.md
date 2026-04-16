# Stepwise Jaw + EEG Benchmark

- Registry file count: `11`
- Diagnostic-only files: `Dalin-LRJ(1,2)-4:7.csv, Dalin-LR(6)-4:7.csv`

## Jaw Branch

- Stage 1 pass: `True`
- Best trigger strategy: `binary_clench_threshold` with test weighted event-F1 `0.767`
- Stage 2 pass: `True`
- Stage 3 pass: `False`
- LRJ jaw pooled constrained exact-count accuracy: `0.583`
- LRJ jaw pooled constrained MAE: `0.750`

## Hand Branch

- LRJ hand decode pass: `False`
- LRJ hand pooled constrained exact-count accuracy: `0.458`
- LRJ hand pooled constrained MAE: `0.875`
- Stage 1/2 pass: `False`
- Main inside-block fraction: `0.962`
- Stress inside-block fraction: `0.808`
- Stage 3 pass: `False`
- EEG Stage 3 status: skipped because `the hand LRJ constrained decode did not clear its exact-count/MAE gate`

## Registry Preview

| key | subject | filename | csv_path | protocol_family | trust_tier | processing_family | marker_semantics | supported_targets |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ben_lrj | Ben | Ben-LRJ(1-6)-4:9.csv | /Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Ben/Ben-LRJ(1-6)-4:9.csv | LRJ | main | lrj | 1=LEFT, 2=RIGHT, 3=JAW; same-code start/end pairs; expected within-label counts 1..6 | interval, event, side, jaw_event, exact_count |
| yaniv_lrj | Yaniv | Yaniv-LRJ(6)-4:7.csv | /Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/Yaniv-LRJ(6)-4:7.csv | LRJ | main | lrj | 1=LEFT, 2=RIGHT, 3=JAW; same-code start/end pairs; expected within-label counts 1..6 | interval, event, side, jaw_event, exact_count |
| yaniv_hr_2026_02_27 | Yaniv | HR-2-27-26-(01).csv | /Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/HR-2-27-26-(01).csv | HR | main | jaw | Legacy 1->2 movement1 and 3->4 movement2 markers; movement labels map to HOLD vs REPEATED at the coarse-segment level | jaw_event, jaw_type |
| yaniv_hr_2026_03_08 | Yaniv | HR-3-8-26-(02).csv | /Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/HR-3-8-26-(02).csv | HR | main | jaw | Legacy 1->2 movement1 and 3->4 movement2 markers; movement labels map to HOLD vs REPEATED at the coarse-segment level | jaw_event, jaw_type |
| yaniv_hr_2026_03_15 | Yaniv | HR-3-15-26-(03).csv | /Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/HR-3-15-26-(03).csv | HR | main | jaw | Legacy 1->2 movement1 and 3->4 movement2 markers; movement labels map to HOLD vs REPEATED at the coarse-segment level | jaw_event, jaw_type |
| yaniv_lr_2026_02_27 | Yaniv | LR-2-27-26-(01).csv | /Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/LR-2-27-26-(01).csv | LR | main | left_right | 1->2=LEFT block, 3->4=RIGHT block | block, event, side |
| yaniv_lr_2026_03_15 | Yaniv | LR-3-15-26-(04).csv | /Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/LR-3-15-26-(04).csv | LR | main | left_right | 1->2=LEFT block, 3->4=RIGHT block | block, event, side |
| yaniv_lr_2026_03_08_a | Yaniv | LR-3-8-26-(02).csv | /Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/LR-3-8-26-(02).csv | LR | stress | left_right | 1->2=LEFT block, 3->4=RIGHT block | block, event, side |
| yaniv_lr_2026_03_08_b | Yaniv | LR-3-8-26-(03).csv | /Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Yaniv/LR-3-8-26-(03).csv | LR | stress | left_right | 1->2=LEFT block, 3->4=RIGHT block | block, event, side |
| dalin_lrj_diagnostic | Dalin | Dalin-LRJ(1,2)-4:7.csv | /Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Dalin/Dalin-LRJ(1,2)-4:7.csv | diagnostic | diagnostic | diagnostic | Diagnostic-only LRJ one-off; do not promote into the first benchmark scorecard | diagnostic |
| dalin_lr6_diagnostic | Dalin | Dalin-LR(6)-4:7.csv | /Users/yanivnaggar/Desktop/Spring 2026/IS/OPENBCI_runs/Dalin/Dalin-LR(6)-4:7.csv | diagnostic | diagnostic | diagnostic | Diagnostic-only LR(6) benchmark; useful for frozen counter sanity checks only | diagnostic |