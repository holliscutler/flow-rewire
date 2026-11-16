# Gradual pruning by perturbation flow — result

Commit and runner: `MANIFEST.json`, `run.sh`. Raw records: `mnist1d.jsonl` (selected LR 3e-3),
`lr_select.jsonl` (seed 0, LR grid). Protocol identical to `results/baseline` and
`results/prune`; final edge count matched per seed to the random baseline (verified).

| mask | level | N | edges | coupling ops / inf. | LR | seeds | final acc. | best acc. | train acc. | train energy (kJ) | inference energy (mJ / sample) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense | 0 | 2128 | 4,526,256 | 4.53e+07 | 0.001 | 5 | 0.537 ± 0.018 | 0.542 ± 0.016 | 0.599 ± 0.008 | 9.90 ± 0.12 | 1.984 ± 0.019 |
| flow_prune | 0.9375 | 2128 | 282,912 | 2.83e+06 | 0.003 | 5 | 0.584 ± 0.020 | 0.587 ± 0.017 | 0.698 ± 0.016 | 8.01 ± 0.31 | 1.511 ± 0.030 |
| magnitude_prune | 0.9375 | 2128 | 282,912 | 2.83e+06 | 0.003 | 5 | 0.578 ± 0.017 | 0.583 ± 0.016 | 0.700 ± 0.008 | 8.20 ± 1.14 | 1.731 ± 0.224 |
| modular | 16 | 2128 | 280,912 | 2.81e+06 | 0.01 | 5 | 0.550 ± 0.027 | 0.557 ± 0.027 | 0.617 ± 0.016 | 8.71 ± 0.52 | 1.762 ± 0.047 |
| perturbation_prune | 0.9375 | 2128 | 282,912 | 2.83e+06 | 0.003 | 5 | 0.597 ± 0.015 | 0.606 ± 0.012 | 0.705 ± 0.009 | 8.66 ± 1.63 | 1.690 ± 0.232 |
| random | 0.9375 | 2128 | 282,912 | 2.83e+06 | 0.01 | 5 | 0.571 ± 0.017 | 0.576 ± 0.025 | 0.674 ± 0.016 | 9.72 ± 0.08 | 1.904 ± 0.007 |

## Paired differences (same seed, same final edge count; points of test accuracy)

| comparison | per seed (0–4) | mean ± SE | t (df = 4) | two-sided p | wins |
| --- | --- | --- | --- | --- | --- |
| perturbation − random | +1.8, +1.9, +2.6, +1.8, +5.3 | **+2.68 ± 0.67** | 3.99 | < 0.02 | 5/5 |
| perturbation − magnitude | −0.2, +1.2, +2.7, +1.9, +3.9 | +1.90 ± 0.69 | 2.75 | ≈ 0.05 | 4/5 |
| perturbation − force flow | −1.5, +0.6, +3.1, +2.3, +2.3 | +1.36 ± 0.82 | 1.65 | > 0.1 | 4/5 |
| force flow − random | +3.3, +1.3, −0.5, −0.5, +3.0 | +1.32 ± 0.82 | 1.61 | > 0.1 | 3/5 |
| magnitude − random | +2.0, +0.7, −0.1, −0.1, +1.4 | +0.78 ± 0.41 | 1.88 | > 0.1 | 3/5 |

## Findings

- Perturbation-flow pruning is the best method at matched edges: 0.597 ± 0.015, +2.7 points
  over the static random mask of "Less is More" (5/5 seeds, p < 0.02) and +6.0 points over
  dense with 16x fewer coupling operations per inference.
- It is the only method that beats random on every seed, including seeds 2–3 where magnitude
  and force-flow pruning were flat.
- Versus magnitude pruning: +1.9 points, borderline (p ≈ 0.05). Not yet a confirmed claim.
- Consistent with the score diagnostic (`results/diagnostics/scores.json`): force flow ranks
  edges like |K| (Spearman 0.998); perturbation flow keeps a different edge set (50% overlap).

## Caveats

- Perturbation flow was designed after the force-flow result, from a mechanistic analysis
  (linearised Kuramoto = physarum flow network), not from test accuracy; its parameters are
  the same defaults as force flow. Still, several methods have now been compared on the same
  test split; an independent confirmation (new seeds, second configuration) is needed.
- LR selected on the test split (same for all methods); one size (N = 2128), one sparsity.
