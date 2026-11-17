# Physarum flow pruning for Kuramoto oscillator networks

## Summary

"Less is More" (Unconventional AI, 2026) shows that large Kuramoto oscillator networks
lose accuracy through synchronization collapse, and that **static** sparse masks (random
or modular) recover it. Those masks are chosen without reference to the dynamics.

This project chooses the mask *from* the dynamics. Linearizing Kuramoto dynamics around
the mean trajectory gives an exact physarum flow network: couplings become conductances
and phase perturbations become pressures. Starting dense, edges are pruned gradually by
the physarum conductivity rule dD/dt = |Q| − γD (Tero et al., 2007; Cutler & Li,
ALIFE 2026).

The test is MNIST1D at N = 2128 oscillators, with every sparse network at the same edge
count (6.25%) and the protocol of "Less is More". Over 10 paired seeds, perturbation-flow
pruning gives:

- **vs. gradual magnitude pruning:** +2.3 points, better on 9/10 seeds, p < 0.001.
  Magnitude pruning is the standard method. It does no better than a random mask here
  (−0.3 points, 5/10 seeds).
- **vs. the random mask of "Less is More":** +1.9 points, better on 8/10 seeds, p = 0.039.
- **vs. dense:** +6.4 points, better on 10/10 seeds, with 16× fewer coupling operations per
  inference.

Seeds 5–9 were run after the method and every setting were frozen. They replicate the
gain over magnitude pruning (5/5 seeds) but not, on their own, the gain over the random
mask ([Section 4](#4-replication)).

## 1. Idea: Kuramoto dynamics is a flow network

Un-0 integrates

  θ̇ᵢ = ωᵢ + Σⱼ Kᵢⱼ sin(θⱼ − θᵢ).

Let θ̄ be the batch-mean (circular) phase trajectory and δθ = θ − θ̄ one input's
deviation from it. To first order,

  δθ̇ᵢ = Σⱼ Cᵢⱼ (δθⱼ − δθᵢ),  with Cᵢⱼ = Kᵢⱼ cos(θ̄ⱼ − θ̄ᵢ).

This is Kirchhoff flow on a graph, which is the physarum model term for term:

| Physarum (Tero et al., 2007) | Linearized Kuramoto |
| --- | --- |
| conductivity Dᵢⱼ | Cᵢⱼ = Kᵢⱼ cos(θ̄ⱼ − θ̄ᵢ) |
| pressure pᵢ | δθᵢ, the input-driven phase perturbation |
| flux Qᵢⱼ = Dᵢⱼ (pⱼ − pᵢ) | Qᵢⱼ = Cᵢⱼ (δθⱼ − δθᵢ) |
| sources | input oscillators (set by the encoder) |

An edge's **perturbation flow** |Qᵢⱼ| measures how much input-dependent signal it
carries. That is the quantity a classifier needs to route from input to readout.

A first, naive mapping used the nonlinear coupling force, |Kᵢⱼ sin(θⱼ − θᵢ)|, as the flow.
It ranks edges almost exactly like |Kᵢⱼ| ([Section 5](#5-why-the-perturbation-form-matters))
and is kept as an ablation.

## 2. Method

Training follows the baseline recipe, except that it starts dense. Edges are removed
permanently, and the final edge count is matched per seed to that seed's random-baseline
mask.

- **Schedule.** One pruning event every 250 steps, from step 250 until 75% of training
  (24 events). The active edge count follows the cubic schedule of Zhu & Gupta (2017).
- **Measurement.** At each event, one training batch of 100 is integrated with the current
  (pruned) couplings, and non-input initial phases are shared across the batch. As a
  result, δθ reflects the inputs only. |Qᵢⱼ| is averaged over the batch and the 11 time
  points (`flow.abs_perturbation_flow`).
- **Conductivity.** Dᵢⱼ ← (1 − γ) Dᵢⱼ + Qᵢⱼ with γ = 0.5, the discrete physarum rule.
  At each event, the lowest-D active edges are removed.

Configuration: `PruneConfig` defaults in `src/flow_rewire/pruning.py`. These defaults were
set before any result and are shared by all pruning scores.

## 3. Results

### Setup

| | |
| --- | --- |
| Task | MNIST1D, default split: 4000 train / 1000 test |
| Model | Un-0 Kuramoto classifier: 40 input, 2048 hidden, 40 readout oscillators (N = 2128). 10 Euler steps, T = 1, sin/cos readout. |
| Training | Un-0 optimizer: DecoupledAdamW, betas (0.9, 0.95), weight decay 1e-3, warmup then linear decay, gradient clip 1.0. MNIST1D reference loop: batch 100, 8000 steps. |
| Learning rate | Selected on seed 0 from {1e-3, 3e-3, 1e-2}, separately for each method, then fixed for all seeds |
| Sparsity | 93.75% (s = 1 − 2⁻⁴), 282,912 of 4.5M couplings. Modular uses Z = 16 (280,912). |
| Seeds | 0–9, paired: the same seed gives the same initialization and data order for every method |

Unspecified details were inferred from Un-0 and the MNIST1D reference trainer. Each one
is listed with its source in [`mnist1d_spec.md`](mnist1d_spec.md).

**Pipeline check.** The same codebase reproduces the released Un-0 CIFAR-10 checkpoint at
FID 11.145 (published 11.16). The MNIST1D baselines reproduce the qualitative trend of
"Less is More" Fig. 9: dense accuracy falls with N, and random sparsity recovers it at
N = 2128.

### Accuracy

![Test accuracy per method](figures/accuracy.png)

| method | seeds | test acc. | train acc. | LR |
| --- | --- | --- | --- | --- |
| **Perturbation-flow pruning** | 10 | **0.592 ± 0.018** | 0.707 ± 0.013 | 3e-3 |
| Force-flow pruning (ablation) | 5 | 0.584 ± 0.020 | 0.698 ± 0.016 | 3e-3 |
| Random mask ("Less is More") | 10 | 0.573 ± 0.016 | 0.675 ± 0.013 | 1e-2 |
| Gradual magnitude pruning | 10 | 0.570 ± 0.027 | 0.694 ± 0.015 | 3e-3 |
| Modular mask ("Less is More") | 5 | 0.550 ± 0.027 | 0.617 ± 0.016 | 1e-2 |
| Dense | 10 | 0.528 ± 0.021 | 0.598 ± 0.009 | 1e-3 |

Values are mean ± std over seeds, using final accuracy at step 8000. Force-flow pruning
and the modular mask were not part of the replication, so they have seeds 0–4 only.

### Paired comparison

![Paired differences](figures/paired.png)

| perturbation-flow pruning − | seeds | mean (points) | 95% CI | p | wins |
| --- | --- | --- | --- | --- | --- |
| Magnitude pruning | 10 | **+2.3** | [+1.2, +3.3] | **< 0.001** | **9/10** |
| Random mask | 10 | **+1.9** | [+0.1, +3.8] | **0.039** | **8/10** |
| Force-flow pruning | 5 | +1.4 | [−0.9, +3.7] | 0.17 | 4/5 |
| Dense | 10 | +6.4 | [+4.7, +8.2] | < 0.001 | 10/10 |

The p-values come from a two-sided paired t-test.

- Magnitude pruning minus random is −0.3 points (5/10 seeds, p = 0.77). Choosing edges by
  coupling strength gains nothing over choosing them at random. Choosing them by
  perturbation flow does.
- Pruned models train better than the random mask (train acc. 0.69–0.71 vs. 0.68). Only
  perturbation flow turns this into better test accuracy.

## 4. Replication

After the 5-seed result, everything was frozen: method, defaults and each method's
selected learning rate. Seeds 5–9 were then run fresh (`results/replicate/`).

| perturbation-flow pruning − | seeds 0–4 | seeds 5–9 (fresh) |
| --- | --- | --- |
| Magnitude pruning | +1.9, 4/5, p = 0.051 | **+2.6, 5/5, p = 0.016** |
| Random mask | +2.7, 5/5, p = 0.016 | +1.2, 3/5, p = 0.46 |
| Dense | +6.1, 5/5, p = 0.005 | +6.8, 5/5, p = 0.005 |

- **Vs. magnitude pruning:** the gain replicated and grew.
- **Vs. the random mask:** the gain is smaller on the fresh seeds and not significant on
  its own. The random mask lost on seeds 5, 7 and 9 and won on seeds 6 and 8 (both about
  0.59). The 10-seed estimate (+1.9, p = 0.039) is the one to quote.

## 5. Why the perturbation form matters

Each score was computed on the same partially trained dense model, and the top 6.25% of
edges kept under each (`results/diagnostics/scores.json`):

| | Spearman ρ with \|K\| | kept-set overlap with \|K\| | in-phase edges kept* |
| --- | --- | --- | --- |
| Force flow ⟨\|K sin Δθ\|⟩ | 0.998 | 92% | 5.8% |
| Perturbation flow | 0.815 | 50% | 8.1% |

\*Edges with cos Δθ̄ > 0.9. Uniform keeping would retain 6.25% of them.

- **Force flow is magnitude pruning in disguise.** sin Δθ is roughly independent of the
  edge, so |K| dominates the score.
- **Perturbation flow selects a different half of the network.** It deliberately keeps
  in-phase edges, which have high conductance cos Δθ̄ and transmit input-driven
  perturbations. The naive view would prune them as "synchronized".

## 6. Cost

![Accuracy vs. coupling operations](figures/cost.png)

| | dense | random mask | perturbation-flow pruning |
| --- | --- | --- | --- |
| Coupling ops / inference | 4.53e7 | 2.83e6 | 2.83e6 |
| Training coupling ops / sample | 4.53e7 | 2.83e6 | 1.15e7 |

- **Inference.** The pruned network has the same cost as the random mask: 16× below
  dense.
- **Training.** Because it starts dense, training costs 4× the static mask and 4× below
  dense. A sparse-start variant with regrowth would remove this overhead.

**Measured GPU energy** was recorded for every run (NVML power sampled every 50 ms and
integrated, RTX 5060 Laptop GPU, 40 W limit). It does not measure the benefit of sparsity
here. This implementation applies the mask to a dense matrix, so every method performs
the same floating-point operations per Euler step. Measured differences therefore reflect
GPU power state, not the network.

For example, in the same session the random and perturbation-pruned networks measured
2.00 and 1.55 mJ per inference sample, for identical computation. Realizing the
coupling-operation savings needs sparse kernels or hardware where each coupling is a
physical element. Coupling operations are the hardware metric used here.

## 7. Limitations

- **One configuration.** N = 2128, s = 0.9375, MNIST1D only. A second size or sparsity
  is the most informative next experiment.
- **Test-split selection.** The learning rate is selected on the test split, as in the
  MNIST1D reference, which has no validation split. This applies equally to all methods.
- **Sequential design.** Perturbation flow was designed after the force-flow result. The
  design came from the analysis in Section 5, not from test accuracy, and it reuses the
  same defaults. The fresh-seed replication (Section 4) addresses this.
- **Small dense networks.** A small dense network (N = 88) reaches 0.641 in this setup.
  The claim concerns the large-N regime, where "Less is More" documents collapse and
  sparsity is required.

## Reproduce

```bash
uv sync --group dev --group eval
uv run pytest
results/perturb/run.sh                         # perturbation-flow pruning, seeds 0-4
results/replicate/run.sh                       # frozen replication, seeds 5-9 (~70 min)
uv run python -m flow_rewire.figures           # figures in docs/figures
```

Each `results/*/MANIFEST.json` records the commit, protocol and commands used for that run.

## References

- Unconventional AI. *Introducing Un-0: Generating Images with Coupled Oscillators.* 2026.
- Unconventional AI. *Less is More: Scaling Dynamics with Sparsity.* 2026.
- Cutler, H., Li, C.-H. G. *Topology That Learns.* ALIFE 2026.
- Tero, A., Kobayashi, R., Nakagaki, T. A mathematical model for adaptive transport network
  in path finding by true slime mold. *J. Theor. Biol.* 244(4), 2007.
- Zhu, M., Gupta, S. To prune, or not to prune. arXiv:1710.01878, 2017.
- Greydanus, S. Scaling down deep learning with MNIST-1D. 2020.
