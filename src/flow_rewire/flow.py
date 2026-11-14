"""Per-edge flow statistics for Un-0 Kuramoto dynamics.

The Kuramoto coupling term of oscillator i is  sum_j K_ij sin(theta_j - theta_i).
Read as a physarum network (Tero et al., 2007), K_ij is the edge conductivity,
sin(theta_j - theta_i) the potential difference, and their product the flow

    Q_ij(t) = K_ij * sin(theta_j(t) - theta_i(t)).

This module integrates a released Un-0 model and accumulates the time- and
batch-averaged edge flow  <|Q_ij|>  along its own generation trajectories.
Edges between phase-locked oscillators (delta theta ~ 0 or pi) carry ~zero flow.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


def unwrap(module: nn.Module) -> nn.Module:
    """Return the eager module behind a ``torch.compile`` wrapper."""
    return getattr(module, "_orig_mod", module)


def effective_coupling(dynamics: nn.Module) -> Tensor:
    """Main-pool coupling exactly as used in the forward pass (zero diagonal, scaled)."""
    dyn = unwrap(dynamics)
    K = dyn.K
    return (K - torch.diag_embed(K.diagonal())) * dyn._K_scale


def abs_edge_flow(theta: Tensor, coupling: Tensor, chunk: int = 32) -> Tensor:
    """Batch-summed absolute edge flow  sum_b |K_ij sin(theta_bj - theta_bi)|.

    Args:
        theta: Main-pool phases, shape ``(batch, n)``.
        coupling: Effective coupling, shape ``(n, n)``; row i receives from column j.
        chunk: Batch elements processed at once (memory is ``chunk * n * n``).

    Returns:
        Tensor of shape ``(n, n)``.

    """
    out = torch.zeros_like(coupling)
    abs_k = coupling.abs()
    for start in range(0, theta.shape[0], chunk):
        th = theta[start : start + chunk]
        delta = th.unsqueeze(1) - th.unsqueeze(2)  # [b, i, j] = theta_j - theta_i
        out += (delta.sin().abs() * abs_k).sum(dim=0)
    return out


@dataclass
class FlowStats:
    """Trajectory-averaged statistics of one flow measurement."""

    flow: Tensor  # (n, n) mean |Q_ij| over batch and time steps
    order_parameter: Tensor  # (num_steps + 1,) mean Kuramoto R(t) over batch
    final_state: Tensor  # (batch, n + n_cond) phases at t = T


@torch.no_grad()
def measure_flow(
    model: nn.Module,
    class_id: Tensor,
    *,
    generator: torch.Generator | None = None,
    chunk: int = 32,
) -> FlowStats:
    """Integrate ``model`` with its own Euler schedule and accumulate edge flow.

    Reproduces ``ConditionalImplicitKuramotoGenerator.forward`` (eval mode, no class
    dropout) step by step so that flow is measured on the exact trajectories the
    model generates from. Only the Euler solver is supported (the released CIFAR-10
    checkpoints use 10 Euler steps).
    """
    if model.solver != "euler":
        raise ValueError(f"Only the euler solver is supported, got {model.solver!r}.")
    dyn = unwrap(model.dynamics)
    param = next(model.parameters())
    state = model._sample_initial_state(
        class_id.shape[0], device=param.device, dtype=param.dtype, generator=generator
    )
    drive = dyn.K_drive[class_id]
    coupling = effective_coupling(dyn)
    dt = model.integration_time / model.num_steps

    flow = torch.zeros_like(coupling)
    order = []
    for step in range(model.num_steps + 1):
        theta = state[:, : dyn.n]
        flow += abs_edge_flow(theta, coupling, chunk=chunk)
        order.append(torch.exp(1j * theta.to(torch.float32)).mean(dim=1).abs().mean())
        if step < model.num_steps:
            state = state + dt * dyn(state, None, drive)

    flow /= class_id.shape[0] * (model.num_steps + 1)
    return FlowStats(flow=flow, order_parameter=torch.stack(order), final_state=state)
