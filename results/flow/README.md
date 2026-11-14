# Flow rewiring v1 (naive) — result

Commit and commands: `MANIFEST.json`. Raw records: `mnist1d.jsonl` (selected LR 1e-2),
`lr_select.jsonl` (seed 0, LR grid). Protocol identical to `results/baseline`.

| method (N = 2128, 282,912 edges) | test acc. (5 seeds) | train acc. |
| --- | --- | --- |
| random (static, baseline) | 0.571 ± 0.017 | 0.674 ± 0.016 |
| flow v1 | 0.562 ± 0.013 | 0.697 ± 0.010 |

Paired difference (flow − random, same initial mask per seed): **−0.008 ± 0.008 (SE)**,
t = −1.06. No significant effect.

## Diagnosis

- The topology locks in at the first event: 72–73% of edges swapped at step 250,
  0% at the last event, on every seed. Rewiring is effectively one-shot, not continual.
- Cause: dormant edges get no gradient, so their K stays at initialisation (~N^-1/2) while
  active K grows. Potential flow |K sin Δθ| of dormant edges is therefore systematically
  smaller, and they cannot compete. Conductivity is conflated with the learned weight,
  unlike Tero's Q = D·Δp.
- Higher train accuracy without test gain: a different topology, not a better one.

## Note on measured energy

GPU energy for flow (8.2 kJ, 1.51 mJ/sample) is lower than random (9.7 kJ, 1.90 mJ/sample)
although the simulated computation is identical (dense matmul). The runs were made in
different sessions; measured GPU energy drifts between sessions (thermal/power state) and
should only be compared within a session. Coupling operations are identical (2.83e6).
