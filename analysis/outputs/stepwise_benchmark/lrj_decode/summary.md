# Shared LRJ Decode

- Gate pass: `False`

- LRJ files: `Ben-LRJ(1-6)-4:9.csv, Yaniv-LRJ(6)-4:7.csv`
- Jaw pooled constrained exact-count accuracy: `0.583`
- Jaw pooled constrained MAE: `0.750`
- Left/right pooled constrained exact-count accuracy: `0.458`
- Left/right pooled constrained MAE: `0.875`

## Decode Summary

| filename | label | trial_count | unconstrained_exact_accuracy | constrained_exact_accuracy | unconstrained_mae | constrained_mae |
| --- | --- | --- | --- | --- | --- | --- |
| Ben-LRJ(1-6)-4:9.csv | LEFT | 6 | 0.5 | 0.5 | 1.0 | 1.0 |
| Ben-LRJ(1-6)-4:9.csv | RIGHT | 6 | 0.6666666666666666 | 0.6666666666666666 | 0.6666666666666666 | 0.6666666666666666 |
| Ben-LRJ(1-6)-4:9.csv | JAW | 6 | 0.5 | 0.5 | 1.1666666666666667 | 1.1666666666666667 |
| Yaniv-LRJ(6)-4:7.csv | LEFT | 6 | 0.3333333333333333 | 0.3333333333333333 | 1.1666666666666667 | 1.1666666666666667 |
| Yaniv-LRJ(6)-4:7.csv | RIGHT | 6 | 0.3333333333333333 | 0.3333333333333333 | 0.6666666666666666 | 0.6666666666666666 |
| Yaniv-LRJ(6)-4:7.csv | JAW | 6 | 0.6666666666666666 | 0.6666666666666666 | 0.3333333333333333 | 0.3333333333333333 |
| __pooled__ | LEFT | 12 | 0.4166666666666667 | 0.4166666666666667 | 1.0833333333333333 | 1.0833333333333333 |
| __pooled__ | RIGHT | 12 | 0.5 | 0.5 | 0.6666666666666666 | 0.6666666666666666 |
| __pooled__ | JAW | 12 | 0.5833333333333334 | 0.5833333333333334 | 0.75 | 0.75 |