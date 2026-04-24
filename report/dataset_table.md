# Dataset Table

This table follows the manual file-contract registry as the source of truth for the active report path. It excludes the older legacy EMG-only pipeline files and focuses on the datasets that the current EEG/jaw analysis stack treats as in-scope.

| file | subject | protocol family | trust tier | intended use |
| --- | --- | --- | --- | --- |
| `LR-2-27-26-(01).csv` | Yaniv | `LR` | `main` | Clean `LEFT/RIGHT` event validation, event-level `LEFT/RIGHT` modeling, clean cross-session EEG direction benchmark, and replay testing in the hybrid path |
| `LR-3-15-26-(04).csv` | Yaniv | `LR` | `main` | Clean `LEFT/RIGHT` event validation, event-level `LEFT/RIGHT` modeling, clean cross-session EEG direction benchmark, and replay testing in the hybrid path |
| `HR-2-27-26-(01).csv` | Yaniv | `HR` | `main` | Jaw event labeling, jaw model training, jaw replay evaluation, and current reusable jaw detector development |
| `HR-3-8-26-(02).csv` | Yaniv | `HR` | `main` | Jaw event labeling, jaw model training, jaw replay evaluation, and current reusable jaw detector development |
| `HR-3-15-26-(03).csv` | Yaniv | `HR` | `main` | Jaw event labeling, jaw model training, jaw replay evaluation, and current reusable jaw detector development |
| `Ben-LRJ(1-6)-4:9.csv` | Ben | `LRJ` | `main` | Shared offline LRJ benchmark, interval/count-constrained decode analysis, and subject-specific LRJ review wrapper |
| `Yaniv-LRJ(6)-4:7.csv` | Yaniv | `LRJ` | `main` | Shared offline LRJ benchmark, interval/count-constrained decode analysis, and subject-specific LRJ review wrapper |
| `LR-3-8-26-(02).csv` | Yaniv | `LR` | `stress` | Robustness and transfer stress testing for `LEFT/RIGHT`; not a first-pass clean-training file |
| `LR-3-8-26-(03).csv` | Yaniv | `LR` | `stress` | Robustness and transfer stress testing for `LEFT/RIGHT`; not a first-pass clean-training file |
| `Dalin-LR(6)-4:7.csv` | Dalin | `diagnostic` | `diagnostic` | Frozen event-counter sanity checks and count-benchmark reference only; not a clean training file |
| `Dalin-LRJ(1,2)-4:7.csv` | Dalin | `diagnostic` | `diagnostic` | LRJ debugging and postmortem reasoning only; do not promote into the benchmark scorecard or training set |

## Notes

- `main` means the file is part of the active report-scope dataset registry, not that every `main` file is equally mature for live reuse.
- Within the `LR` family, the two Yaniv files from February 27 and March 15 are the cleanest files for first-pass `LEFT/RIGHT` claims.
- The `stress` files are useful for robustness checks, but they should not anchor the main performance claims.
- The `diagnostic` files are intentionally preserved for tuning, sanity checks, and debugging, not for clean model training.
