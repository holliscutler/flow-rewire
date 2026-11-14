# MNIST1D baseline (frozen)

Reproduction of the "Less is More" MNIST1D trend (Unconventional AI, 2026, Fig. 9) under the
protocol in `docs/mnist1d_spec.md`. Code: commit in `MANIFEST.json`. Raw records: `mnist1d.jsonl`.
5 seeds (0–4); LR selected on seed 0 from {1e-3, 3e-3, 1e-2}.

| mask | level | N | edges | coupling ops / inf. | LR | seeds | final acc. | best acc. | train acc. | train energy (kJ) | inference energy (mJ / sample) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense | 0 | 88 | 7,656 | 7.66e+04 | 0.01 | 5 | 0.643 ± 0.014 | 0.652 ± 0.015 | 0.909 ± 0.007 | 2.36 ± 0.06 | 0.069 ± 0.002 |
| dense | 0 | 2128 | 4,526,256 | 4.53e+07 | 0.001 | 5 | 0.537 ± 0.018 | 0.542 ± 0.016 | 0.599 ± 0.008 | 9.90 ± 0.12 | 1.984 ± 0.019 |
| modular | 16 | 2128 | 280,912 | 2.81e+06 | 0.01 | 5 | 0.550 ± 0.027 | 0.557 ± 0.027 | 0.617 ± 0.016 | 8.71 ± 0.52 | 1.762 ± 0.047 |
| random | 0.9375 | 2128 | 282,912 | 2.83e+06 | 0.01 | 5 | 0.571 ± 0.017 | 0.576 ± 0.025 | 0.674 ± 0.016 | 9.72 ± 0.08 | 1.904 ± 0.007 |

## Findings

- **Dense degrades with size**: 0.643 ± 0.014 (N = 88) → 0.537 ± 0.018 (N = 2128); training
  accuracy at N = 2128 is 0.60, i.e. a trainability limit, not overfitting. Reproduces Fig. 9.
- **Random sparsity recovers accuracy at large N**: 0.571 ± 0.017 vs. dense 0.537 ± 0.018
  (+3.4 points) with 16× fewer coupling operations. Reproduces Fig. 9.
- **Modular (Z = 16) recovery is weaker here**: 0.550 ± 0.027 (+1.3 points, within noise). Its
  seed-0 run (0.594), used for LR selection, was the best of five.
- **GPU energy does not track sparsity** (dense matrix simulation): ~9–10 kJ per training run and
  ~1.8–2.0 mJ per inference at N = 2128 for all masks. Coupling operations are the
  hardware-relevant comparison.

## Caveats

- "Validation" is the MNIST1D test split, used for both LR selection and reporting (as in the
  MNIST1D reference trainer); the bias applies equally to all methods.
- LR 1e-2 is the top of the grid for both sparse masks.
- Measured energy is gross GPU board energy (includes idle draw) on an RTX 5060 Laptop GPU.

## Target for flow rewiring

N = 2128, ≈ 283k edges (2.83e6 coupling ops / inference), same protocol: exceed random
0.571 ± 0.017 (SE ≈ 0.008 over 5 seeds).
