"""Edge scores and sparsity masks for the main-pool coupling matrix K.

All masks keep the top ``(1 - sparsity)`` fraction of off-diagonal edges by score.
Only ``K`` is pruned; the class-drive and conditioning couplings are left intact,
matching the scope of the static masks in "Less is More" (Unconventional AI, 2026).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from flow_rewire.flow import unwrap


def topk_mask(score: Tensor, sparsity: float) -> Tensor:
    """Binary mask keeping the highest-scoring ``(1 - sparsity)`` off-diagonal edges."""
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}.")
    n = score.shape[0]
    off_diag = ~torch.eye(n, dtype=torch.bool, device=score.device)
    num_keep = round((1.0 - sparsity) * n * (n - 1))
    values = score.masked_fill(~off_diag, float("-inf")).flatten()
    keep = torch.zeros_like(values, dtype=torch.bool)
    keep[torch.topk(values, num_keep).indices] = True
    return keep.view(n, n)


def random_score(n: int, *, generator: torch.Generator | None = None) -> Tensor:
    """Uniform random scores: the static random-sparsity baseline."""
    return torch.rand(n, n, generator=generator)


def magnitude_score(K: Tensor) -> Tensor:
    """|K_ij|: the standard magnitude-pruning baseline."""
    return K.abs()


def random_mask(n: int, sparsity: float, *, generator: torch.Generator | None = None) -> Tensor:
    """Static random mask: each coupling is deleted independently with probability ``s``."""
    return torch.rand(n, n, generator=generator) >= sparsity


def modular_mask(group_sizes: list[int], num_modules: int) -> Tensor:
    """Static modular mask: all-to-all within modules, no coupling between modules.

    Each group of oscillators (e.g. inputs, hidden, readout) is dealt round-robin across
    modules, so every module receives an equal share of each group. This mirrors the
    "Less is More" requirement of an equal number of readout oscillators per module.
    """
    module_id = torch.cat([torch.arange(size) % num_modules for size in group_sizes])
    return module_id[:, None] == module_id[None, :]


@torch.no_grad()
def apply_mask(model: nn.Module, mask: Tensor) -> None:
    """Zero pruned entries of the main-pool coupling ``K`` in place."""
    K = unwrap(model.dynamics).K
    K.mul_(mask.to(device=K.device, dtype=K.dtype))
