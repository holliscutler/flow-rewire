"""Gradual pruning from a dense Kuramoto coupling matrix: physarum flow vs. magnitude.

Training starts dense. Every ``interval`` steps from ``interval`` until
``stop_fraction * total_steps``, the active edge count follows the cubic schedule of
Zhu & Gupta (2017) from all edges down to ``final_edges``; at each event the
lowest-scoring active edges are removed permanently.

Scores:
  * ``flow``: physarum conductivity (Tero et al., 2007; Cutler & Li, ALIFE 2026, Eq. 8)
        D_ij <- (1 - gamma) D_ij + Q_ij,   Q_ij = < |K_ij sin(theta_j - theta_i)| >
    measured on the current (pruned) dynamics, so flow re-routed through surviving edges
    after each prune is reflected in the next measurement. Edges between phase-locked
    oscillators (delta theta ~ 0 or pi) carry no flow and are pruned first.
  * ``perturbation``: the same conductivity rule driven by perturbation flow
        Q_ij = < |K_ij cos(theta_bar_j - theta_bar_i) (dtheta_j - dtheta_i)| >,
    the exact physarum form of linearised Kuramoto dynamics (see flow.abs_perturbation_flow).
    Measured with non-input phases shared across the batch, so dtheta is input-driven:
    inputs act as sources and perturbations propagate along edges. Phase-locked edges
    (cos ~ 1) have maximal conductance and are kept, unlike ``flow``.
  * ``magnitude``: |K_ij| (gradual magnitude pruning; the control for the |K| factor in Q).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from flow_rewire.flow import abs_edge_flow, abs_perturbation_flow
from flow_rewire.masks import topk_mask_count
from flow_rewire.mnist1d_model import KuramotoClassifier

SCORES = ("flow", "perturbation", "magnitude")
FLOW_FNS = {"flow": abs_edge_flow, "perturbation": abs_perturbation_flow}


@dataclass(frozen=True)
class PruneConfig:
    """Pruning schedule and conductivity dynamics."""

    score: str = "flow"
    interval: int = 250  # training steps between pruning events
    stop_fraction: float = 0.75  # final sparsity reached at this fraction of training
    gamma: float = 0.5  # flow conductivity decay per event: D <- (1 - gamma) D + Q
    chunk: int = 10  # batch elements per flow-computation chunk (memory: chunk * N^2)


def cubic_edge_schedule(step: int, *, start_edges: int, final_edges: int, t0: int, t1: int) -> int:
    """Active edge count at ``step`` under the cubic schedule (Zhu & Gupta, 2017)."""
    if step <= t0:
        return start_edges
    if step >= t1:
        return final_edges
    remaining = (1.0 - (step - t0) / (t1 - t0)) ** 3
    return final_edges + round((start_edges - final_edges) * remaining)


class GradualPruner:
    """Permanently prune ``model.mask`` to ``final_edges`` by the configured score."""

    def __init__(
        self,
        model: KuramotoClassifier,
        *,
        final_edges: int,
        total_steps: int,
        config: PruneConfig,
    ) -> None:
        """Record the starting edge count and the schedule end points."""
        if config.score not in SCORES:
            raise ValueError(f"score must be one of {SCORES}, got {config.score!r}.")
        self.model = model
        self.config = config
        self.start_edges = model.num_edges
        self.final_edges = int(final_edges)
        self.t1 = int(config.stop_fraction * total_steps)
        self.D = torch.zeros(model.n, model.n, device=model.K.device)
        self.events = 0

    def due(self, step: int) -> bool:
        """Whether a pruning event happens after training step ``step``."""
        return step % self.config.interval == 0 and step <= self.t1

    @torch.no_grad()
    def flow(self, x: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Batch- and trajectory-mean flow (``config.score``) on active edges."""
        coupling = self.model.coupling
        shared = self.config.score == "perturbation"  # input-driven variation only
        states = self.model.trajectory(x, generator, shared_noise=shared)
        flow_fn = FLOW_FNS[self.config.score]
        total = sum(flow_fn(theta, coupling, chunk=self.config.chunk) for theta in states)
        return total / (x.shape[0] * len(states))

    @torch.no_grad()
    def prune(self, step: int, x: Tensor, generator: torch.Generator | None = None) -> int:
        """Run one pruning event after ``step``; return the new active edge count."""
        if self.config.score in FLOW_FNS:
            self.D.mul_(1.0 - self.config.gamma).add_(self.flow(x, generator))
            score = self.D
        else:
            score = self.model.K.abs()
        target = cubic_edge_schedule(
            step,
            start_edges=self.start_edges,
            final_edges=self.final_edges,
            t0=0,
            t1=self.t1,
        )
        active_score = score.masked_fill(~self.model.mask, float("-inf"))
        self.model.mask.copy_(topk_mask_count(active_score, target))
        self.events += 1
        return target

    def summary(self) -> dict:
        """Run-level statistics for the result record."""
        return {"prune_events": self.events, "prune_score": self.config.score}
