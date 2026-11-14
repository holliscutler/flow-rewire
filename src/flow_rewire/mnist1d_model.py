"""Kuramoto MNIST1D classifier following "Less is More" (Unconventional AI, 2026), Sec. 5.

Architecture, as specified in the post's appendix:
  * 40 input oscillators whose initial phases come from a linear encoder of the signal;
  * H hidden oscillators;
  * 40 readout oscillators, disjoint from the inputs;
  * one coupling matrix K over all N = 80 + H oscillators (dense: N^2 parameters),
    optionally restricted by a fixed binary mask;
  * linear decoder on (sin, cos) of the readout phases -> 10 logits.

Details the post leaves unspecified are taken from the released Un-0 CIFAR-10 recipe
(see docs/mnist1d_spec.md): standard parameterization (K ~ N(0, 1/N)), omega ~ N(0, 1),
T = 1 with 10 Euler steps, mean-relative readout, and non-input phases drawn uniformly
from [-pi, pi) as Un-0 draws all initial phases.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from un0.model import ReadoutTransform, _kuramoto_velocity

N_IN = 40
N_OUT = 40
NUM_CLASSES = 10


class KuramotoClassifier(nn.Module):
    """Linear encoder -> Kuramoto dynamics -> sin/cos linear decoder."""

    def __init__(
        self,
        n_hidden: int,
        *,
        mask: Tensor | None = None,
        num_steps: int = 10,
        integration_time: float = 1.0,
        signal_dim: int = 40,
    ) -> None:
        """Build the model; ``mask`` is a bool ``(N, N)`` keep-mask over K (None = dense)."""
        super().__init__()
        self.n_hidden = int(n_hidden)
        self.n = N_IN + self.n_hidden + N_OUT
        self.num_steps = int(num_steps)
        self.dt = integration_time / num_steps

        self.encoder = nn.Linear(signal_dim, N_IN)
        self.omega = nn.Parameter(torch.randn(1, self.n))
        self.K = nn.Parameter(torch.randn(self.n, self.n) * self.n**-0.5)
        keep = torch.ones(self.n, self.n, dtype=torch.bool) if mask is None else mask.bool()
        if keep.shape != (self.n, self.n):
            raise ValueError(f"mask shape {tuple(keep.shape)} != {(self.n, self.n)}.")
        keep = keep & ~torch.eye(self.n, dtype=torch.bool)
        self.register_buffer("mask", keep)
        self.readout = ReadoutTransform(encoding="sin_cos", relativization="mean_relative")
        self.decoder = nn.Linear(2 * N_OUT, NUM_CLASSES)

    @property
    def coupling(self) -> Tensor:
        """Effective coupling: masked, zero diagonal."""
        return self.K * self.mask

    @property
    def num_edges(self) -> int:
        """Number of active couplings."""
        return int(self.mask.sum())

    def initial_state(self, x: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Phases at t = 0: encoded inputs; all others uniform on [-pi, pi) (Un-0 prior)."""
        noise = torch.rand(x.shape[0], self.n, device=x.device, generator=generator)
        theta = noise * (2.0 * torch.pi) - torch.pi
        return torch.cat([self.encoder(x), theta[:, N_IN:]], dim=1)

    def trajectory(self, x: Tensor, generator: torch.Generator | None = None) -> list[Tensor]:
        """All phase states ``theta(t_k)``, k = 0..num_steps (Euler)."""
        K = self.coupling
        theta = self.initial_state(x, generator)
        states = [theta]
        for _ in range(self.num_steps):
            theta = theta + self.dt * _kuramoto_velocity(theta, self.omega, K)
            states.append(theta)
        return states

    def forward(self, x: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Class logits for signals ``x`` of shape ``(batch, signal_dim)``."""
        theta = self.trajectory(x, generator)[-1]
        return self.decoder(self.readout(theta[:, -N_OUT:]))
