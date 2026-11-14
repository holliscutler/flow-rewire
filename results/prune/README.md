# Gradual pruning from dense: flow vs. magnitude — result

Commit and commands: `MANIFEST.json`, `run.sh`. Raw records: `mnist1d.jsonl` (selected LR),
`lr_select.jsonl` (seed 0, LR grid). Protocol identical to `results/baseline`; final edge
count matched per seed to the random baseline (verified for all 5 seeds).

| mask | level | N | edges | coupling ops / inf. | LR | seeds | final acc. | best acc. | train acc. | train energy (kJ) | inference energy (mJ / sample) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense | 0 | 2128 | 4,526,256 | 4.53e+07 | 0.001 | 5 | 0.537 ± 0.018 | 0.542 ± 0.016 | 0.599 ± 0.008 | 9.90 ± 0.12 | 1.984 ± 0.019 |
| flow_prune | 0.9375 | 2128 | 282,912 | 2.83e+06 | 0.003 | 5 | 0.584 ± 0.020 | 0.587 ± 0.017 | 0.698 ± 0.016 | 8.01 ± 0.31 | 1.511 ± 0.030 |
| magnitude_prune | 0.9375 | 2128 | 282,912 | 2.83e+06 | 0.003 | 5 | 0.578 ± 0.017 | 0.583 ± 0.016 | 0.700 ± 0.008 | 8.20 ± 1.14 | 1.731 ± 0.224 |
| modular | 16 | 2128 | 280,912 | 2.81e+06 | 0.01 | 5 | 0.550 ± 0.027 | 0.557 ± 0.027 | 0.617 ± 0.016 | 8.71 ± 0.52 | 1.762 ± 0.047 |
| random | 0.9375 | 2128 | 282,912 | 2.83e+06 | 0.01 | 5 | 0.571 ± 0.017 | 0.576 ± 0.025 | 0.674 ± 0.016 | 9.72 ± 0.08 | 1.904 ± 0.007 |

## Paired differences (same seed, same final edge count; points of test accuracy)

| comparison | per seed (0–4) | mean ± SE | t (df = 4) | wins |
| --- | --- | --- | --- | --- |
| flow − random | +3.3, +1.3, −0.5, −0.5, +3.0 | +1.32 ± 0.82 | 1.61 | 3/5 |
| magnitude − random | +2.0, +0.7, −0.1, −0.1, +1.4 | +0.78 ± 0.41 | 1.88 | 3/5 |
| flow − magnitude | +1.3, +0.6, −0.4, −0.4, +1.6 | +0.54 ± 0.42 | 1.30 | 3/5 |

## Findings

- Ranking at matched edges: flow (0.584) > magnitude (0.578) > random (0.571) > modular (0.550)
  > dense (0.537). Flow pruning is +4.7 points over dense with 16x fewer coupling operations.
- Neither pruning method is significantly better than random at 5 seeds (two-sided p > 0.1),
  and flow vs. magnitude is not resolved. The direction is consistent: on every seed, flow's
  difference to random has the same sign as magnitude's and is larger in magnitude.
- Both pruned models train better than the static random mask (train acc. 0.70 vs. 0.67).
- Both select LR 3e-3 (dense optimum 1e-3, static sparse 1e-2).

## Cost

- Inference: 2.83e6 coupling operations, identical to random.
- Training: 1.15e7 coupling operations per sample on average (dense start), 4x the static
  mask and 4x below dense.
- Measured GPU energy (same session for flow and magnitude): 8.0 vs. 8.2 kJ per training run,
  including flow's measurement overhead. Inference GPU energy differences are measurement noise
  (identical computation).
