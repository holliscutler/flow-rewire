"""Physarum flow rewiring of a sparse Kuramoto coupling mask during training.

The coupling term of oscillator i, sum_j K_ij sin(theta_j - theta_i), is read as a
physarum flow network: K_ij is the edge conductance, the phase difference is the
potential difference, and

    Q_ij = < |K_ij sin(theta_j - theta_i)| >      (batch and trajectory average)

is the flow carried by edge i <- j. Every edge carries a conductivity D_ij updated by
the discrete physarum rule (Cutler & Li, ALIFE 2026, Eq. 8, with alpha = 1):

    D_ij <- (1 - gamma) D_ij + Q_ij

and the active mask is the top-B edges by D, with B fixed (the edge budget). Dormant
(masked) edges are not deleted: they keep their K_ij value and their *potential* flow
is computed from the observed phase difference, so a dormant edge between oscillators
that persistently differ in phase can regain conductivity and reactivate. Edges between
phase-locked oscillators (delta theta ~ 0 or pi) carry no flow and decay.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from flow_rewire.flow import abs_edge_flow
from flow_rewire.masks import topk_mask_count
from flow_rewire.mnist1d_model import KuramotoClassifier


@dataclass(frozen=True)
class PhysarumConfig:
    """Rewiring schedule and conductivity dynamics."""

    interval: int = 250  # training steps between rewiring events
    stop_fraction: float = 0.75  # no rewiring after this fraction of training
    gamma: float = 0.5  # conductivity decay per event: D <- (1 - gamma) D + Q
    chunk: int = 10  # batch elements per flow-computation chunk (memory: chunk * N^2)


@dataclass
class RewireStats:
    """Per-run rewiring statistics."""

    events: int = 0
    swapped_per_event: list[float] = field(default_factory=list)  # fraction of budget


class PhysarumRewirer:
    """Maintain edge conductivities and rewire ``model.mask`` under a fixed edge budget."""

    def __init__(self, model: KuramotoClassifier, total_steps: int, config: PhysarumConfig):
        """Initialise conductivities to zero; the budget is the model's current edge count."""
        self.model = model
        self.config = config
        self.budget = model.num_edges
        self.last_step = int(config.stop_fraction * total_steps)
        self.D = torch.zeros(model.n, model.n, device=model.K.device)
        self.initial_mask = model.mask.clone()
        self.stats = RewireStats()

    def due(self, step: int) -> bool:
        """Whether a rewiring event happens after training step ``step``."""
        return step % self.config.interval == 0 and step <= self.last_step

    @torch.no_grad()
    def flow(self, x: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Mean |K_ij sin(theta_j - theta_i)| over batch and trajectory, for all edges.

        Phases come from the current (masked) dynamics; K is the raw coupling, so active
        edges get their actual flow and dormant edges their potential flow.
        """
        K = self.model.K * ~torch.eye(self.model.n, dtype=torch.bool, device=x.device)
        states = self.model.trajectory(x, generator)
        total = sum(abs_edge_flow(theta, K, chunk=self.config.chunk) for theta in states)
        return total / (x.shape[0] * len(states))

    @torch.no_grad()
    def rewire(self, x: Tensor, generator: torch.Generator | None = None) -> float:
        """Update conductivities from one batch and rewire; return fraction of budget swapped."""
        self.D.mul_(1.0 - self.config.gamma).add_(self.flow(x, generator))
        new_mask = topk_mask_count(self.D, self.budget)
        swapped = (new_mask & ~self.model.mask).sum().item() / self.budget
        self.model.mask.copy_(new_mask)
        self.stats.events += 1
        self.stats.swapped_per_event.append(swapped)
        return swapped

    def summary(self) -> dict:
        """Run-level statistics for the result record."""
        final_vs_initial = (self.model.mask & ~self.initial_mask).sum().item() / self.budget
        return {
            "rewire_events": self.stats.events,
            "swapped_first_event": self.stats.swapped_per_event[0] if self.stats.events else 0.0,
            "swapped_last_event": self.stats.swapped_per_event[-1] if self.stats.events else 0.0,
            "final_edges_changed": final_vs_initial,
        }
