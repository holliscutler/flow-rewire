# Replication on fresh seeds 5–9

Commit and runner: `MANIFEST.json`, `run.sh`. Raw records: `mnist1d.jsonl`. Every setting was
frozen before this run: perturbation-flow pruning with `PruneConfig` defaults, and each
method's learning rate as selected on seed 0. Nothing was re-tuned.

| seed | perturbation flow | random | magnitude | modular | dense |
| --- | --- | --- | --- | --- | --- |
| 5 | 0.590 | 0.567 | 0.571 | 0.531 | 0.497 |
| 6 | 0.574 | 0.594 | 0.532 | 0.559 | 0.541 |
| 7 | 0.576 | 0.559 | 0.540 | 0.541 | 0.531 |
| 8 | 0.574 | 0.593 | 0.544 | 0.563 | 0.497 |
| 9 | 0.621 | 0.561 | 0.616 | 0.527 | 0.529 |
| mean | 0.587 | 0.575 | 0.561 | 0.544 | 0.519 |

Modular (Z = 16) seeds 5–9 were added afterwards with the same frozen settings (LR 1e-2).

| perturbation-flow pruning − | seeds 5–9 | all seeds 0–9 |
| --- | --- | --- |
| magnitude pruning | +2.6, 5/5, p = 0.016 | +2.3, 9/10, p < 0.001 |
| random mask | +1.2, 3/5, p = 0.46 | +1.9, 8/10, p = 0.039 |
| dense | +6.8, 5/5, p = 0.005 | +6.4, 10/10, p < 0.001 |

Two-sided paired t-tests; the statistics are computed by `flow_rewire.figures`.

Measured GPU energy (same session, kJ per training run / mJ per inference sample):
- dense: 10.29 / 2.08
- random: 10.00 / 2.00
- magnitude: 10.26 / 1.99
- perturbation: 8.29 / 1.55

The mask is applied to a dense matrix, so the computation is identical across methods.
These differences reflect GPU power state, not sparsity (see `docs/writeup.md`, Section 6).
