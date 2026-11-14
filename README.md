# flow-rewire

Physarum flow-based sparsification of Kuramoto oscillator couplings, built on
Unconventional AI's [Un-0](https://github.com/unconv-ai/Un-0) release.

## Question

"Less is More" (Unconventional AI, 2026) shows that sparse coupling improves Un-0,
using **static** masks (random or modular) chosen without reference to the dynamics.
It attributes the benefit to avoiding synchronization collapse.

The Kuramoto coupling term already has the form of a physarum flow:

| Physarum (Tero et al., 2007) | Kuramoto (Un-0) |
| --- | --- |
| Q_ij = D_ij (p_i − p_j) / L_ij | K_ij · sin(θ_j − θ_i) |
| conductivity D_ij | coupling K_ij |
| pressure difference | phase difference |
| dD/dt = f(\|Q\|) − γD | keep edges with high ⟨\|K_ij sin(θ_j − θ_i)\|⟩ |

Edges between phase-locked oscillators (Δθ ≈ 0 or π) carry no flow. A flow-based
criterion therefore removes exactly the edges that only maintain synchrony, using a
quantity the physical system computes locally.

**Hypothesis.** At matched sparsity, pruning by trajectory-averaged edge flow degrades
FID less than random or magnitude pruning.

## Method

1. Load a released checkpoint (`cifar10/n1024`) through Un-0's own API.
2. Integrate its generation dynamics and record ⟨|Q_ij|⟩ per edge (`flow_rewire.flow`).
3. Keep the top (1 − s) fraction of edges by score: random, |K|, or flow (`flow_rewire.masks`).
4. Score with Un-0's own clean-FID pipeline (`un0.metrics.compute_fid`), with no retraining.

Un-0 is pinned to upstream commit `75243ca` in `pyproject.toml`.

## Usage

```bash
uv sync --group dev --group eval
uv run pytest                                   # unit tests, ~10 s
uv run python -m flow_rewire.evaluate --method flow --sparsity 0.9 --num-samples 2000
uv run python -m flow_rewire.sweep --sparsities 0.9 0.98 --num-samples 10000
```

## Results

**Baseline reproduction.** `cifar10/n1024`, 50k samples, seed 42: clean-FID **11.145**
(published: 11.16).

**One-shot pruning, no retraining** (10k samples, seed 42; 10k FID is biased upward
relative to 50k, so compare within this table only):

| method | s = 0.90 | s = 0.98 |
| --- | --- | --- |
| dense (s = 0) | 12.82 | 12.82 |
| random | 24.20 | 26.07 |
| magnitude \|K\| | 22.84 | 24.63 |
| flow ⟨\|Q\|⟩ | **22.22** | **24.26** |

Flow ranks first at both sparsities, but the margin over magnitude (0.4–0.6 FID) comes
from a single seed and is not yet significant. The global order parameter R(T) stays
≈ 0.03 (≈ 1/√N) for every configuration, so global synchrony does not discriminate
here; cluster and anti-phase locking require a different metric.

## References

- Unconventional AI. *Introducing Un-0: Generating Images with Coupled Oscillators.* 2026.
- Unconventional AI. *Less is More: Scaling Dynamics with Sparsity.* 2026.
- Cutler, H., Li, C.-H. G. *Topology That Learns.* ALIFE 2026.
- Tero, A., Kobayashi, R., Nakagaki, T. *J. Theor. Biol.* 244(4), 2007.
