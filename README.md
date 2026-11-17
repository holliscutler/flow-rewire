# flow-rewire

Physarum flow pruning of Kuramoto oscillator couplings, built on Unconventional AI's
[Un-0](https://github.com/unconv-ai/Un-0) release and the protocol of "Less is More".

**Full writeup: [docs/writeup.md](docs/writeup.md).**

## Result

Linearized Kuramoto dynamics is an exact physarum flow network:

- Conductance: Kᵢⱼ cos Δθ̄
- Pressure: the input-driven phase perturbation δθ

Starting dense, edges are pruned by the physarum conductivity rule driven by this
perturbation flow. On MNIST1D at N = 2128, all sparse networks keep 6.25% of the edges.

| method | test acc. (10 seeds) | paired gain of this work |
| --- | --- | --- |
| **Perturbation-flow pruning (this work)** | **0.592 ± 0.018** | |
| Random mask ("Less is More") | 0.573 ± 0.016 | +1.9 points, 8/10 seeds, p = 0.039 |
| Gradual magnitude pruning | 0.570 ± 0.027 | +2.3 points, 9/10 seeds, p < 0.001 |
| Dense | 0.528 ± 0.021 | +6.4 points, 10/10 seeds |

- **Cost:** the pruned network uses 16× fewer coupling operations per inference than dense.
- **Replication:** seeds 5–9 were run after the method and all settings were frozen.
  They replicate the gain over magnitude pruning (5/5 seeds) but not, on their own, the
  gain over the random mask (3/5 seeds).

![Test accuracy per method](docs/figures/accuracy.png)

## Layout

| Path | Contents |
| --- | --- |
| `src/flow_rewire/mnist1d_model.py` | Kuramoto classifier, Un-0 dynamics with a coupling mask |
| `src/flow_rewire/mnist1d_train.py` | Training protocol and CLI |
| `src/flow_rewire/flow.py` | Force flow and perturbation flow per edge |
| `src/flow_rewire/pruning.py` | Gradual pruning by physarum conductivity or magnitude |
| `src/flow_rewire/energy.py` | Coupling-operation proxy and measured GPU energy |
| `src/flow_rewire/figures.py` | Writeup figures from the result records |
| `docs/mnist1d_spec.md` | Protocol: specified vs. inferred, with sources |
| `results/<run>/` | Raw records, `MANIFEST.json` (commit, commands) and README per run |

The `results/` runs:

- `baseline`: dense, random and modular masks (frozen)
- `prune`: force-flow vs. magnitude pruning
- `perturb`: perturbation-flow pruning
- `replicate`: frozen replication, seeds 5–9

Un-0 is pinned to upstream commit `75243ca` in `pyproject.toml`.

## Usage

```bash
uv sync --group dev --group eval
uv run pytest                                          # unit tests
uv run python -m flow_rewire.mnist1d_train --hidden 2048 --sparsity 0.9375 \
    --masks perturbation_prune --seeds 0 --lrs 3e-3    # one run, ~4 min on an 8 GB laptop GPU
uv run python -m flow_rewire.figures                   # regenerate figures
```

## Earlier exploration: one-shot pruning of the CIFAR-10 checkpoint

The pipeline reproduces the released `cifar10/n1024` checkpoint at clean-FID **11.145**
(published 11.16; 50k samples, seed 42).

Post-hoc pruning without retraining was tested at 10k samples. Force flow ranked first:

| s | force flow | magnitude | random |
| --- | --- | --- | --- |
| 0.90 | 22.22 | 22.84 | 24.20 |
| 0.98 | 24.26 | 24.63 | 26.07 |

The margin over magnitude came from a single seed. Training with masks at CIFAR scale
does not fit in 8 GB, so the study moved to the MNIST1D setting of "Less is More"
(`flow_rewire.evaluate`, `flow_rewire.sweep`).

## References

- Unconventional AI. *Introducing Un-0: Generating Images with Coupled Oscillators.* 2026.
- Unconventional AI. *Less is More: Scaling Dynamics with Sparsity.* 2026.
- Cutler, H., Li, C.-H. G. *Topology That Learns.* ALIFE 2026.
- Tero, A., Kobayashi, R., Nakagaki, T. *J. Theor. Biol.* 244(4), 2007.
