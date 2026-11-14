# MNIST1D protocol: what "Less is More" specifies

Source: "Less is More: Scaling Dynamics with Sparsity" (Unconventional AI, Aug 2026),
Sec. 5, Figs. 9 and a2, appendix "Methodology for MNIST1D Experiments".
No code for these experiments is released (checked: `unconv-ai/Un-0` main and all
branches contain no sparsity or MNIST1D code, as of commit `75243ca`).

## Specified

| Item | Value | Where |
| --- | --- | --- |
| Input oscillators | 40, initial phases from a learned linear encoder of the signal | appendix |
| Readout oscillators | 40, disjoint from inputs | appendix |
| Decoder | sin/cos of readout phases → linear → softmax | appendix |
| Coupling | one matrix over all 80 + H oscillators, P ≈ (80 + H)² | appendix |
| Encoder/decoder | fixed size across scaling; scaling adds hidden oscillators only | appendix |
| Random sparsity | each coupling deleted with prob. s; s ∈ {0, 0.5, 0.75, …, 0.992} (s = 1 − 2⁻ᵏ) | Sec. 1, Fig. 9 colorbar |
| Modular sparsity | Z modules, all-to-all within, none between; equal readout oscillators per module; Z ≤ 40 | Sec. 1, appendix |
| Size bins (N) | 80–88, 96–112, 144–208, 272–336, 464–592, 848–1104, 1616–2128 | Fig. 9 x-axis |
| Metric | validation accuracy, mean of 5 seeds | Fig. 9, appendix |
| Tuning | hyperparameters swept per configuration | Sec. 5 text |

## Target values (read from Fig. 9, ±0.005)

| N bin | dense val. acc. | best sparse (envelope) |
| --- | --- | --- |
| 80–88 | ≈ 0.578 | ≈ 0.58 |
| 144–208 | ≈ 0.556 | ≈ 0.575 |
| 272–336 | ≈ 0.530 | ≈ 0.57 |
| 464–592 | ≈ 0.519 | ≈ 0.565 |
| 848–1104 | ≈ 0.509 | ≈ 0.57 |
| 1616–2128 | < 0.50 | ≈ 0.565 |

Trend: dense accuracy falls monotonically with N; the sparse envelope stays flat at
≈ 0.565–0.58.

## Not specified: inferred values

Rule: model and optimizer details follow the released Un-0 CIFAR-10 recipe (the only
documented Kuramoto training recipe from the same authors); data and training-loop
details follow the MNIST1D reference trainer (`greydanus/mnist1d`, commit `e2f536f`,
`notebooks/train.py`).

| Item | Value | Source |
| --- | --- | --- |
| Non-input initial phases | uniform on [−π, π) | Un-0 (`_sample_initial_state`) |
| K, ω initialization | K ~ N(0, 1/N), ω ~ N(0, 1) (standard parameterization) | Un-0 CIFAR-10 |
| Dynamics | T = 1, 10 Euler steps | Un-0 CIFAR-10 |
| Readout relativization | mean-relative | Un-0 CIFAR-10 |
| Optimizer | DecoupledAdamW, betas (0.9, 0.95), wd 1e-3, clip 1.0 | Un-0 CIFAR-10 |
| Schedule | 10% linear warmup, linear decay to 0 | Un-0 CIFAR-10 |
| Dataset | default MNIST1D, 4000 train / 1000 test | MNIST1D reference |
| Batch, steps | 100, 8000 | MNIST1D reference |
| "Validation" accuracy | test split, every 250 steps (reference has no val split) | MNIST1D reference |
| LR grid | {1e-3, 3e-3, 1e-2}: spans Un-0 (1.4–2.7e-3) and MNIST1D (1e-2) | both |
