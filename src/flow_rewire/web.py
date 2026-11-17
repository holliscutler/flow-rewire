"""Spider-web coupling topology for the Kuramoto classifier.

Adapted from the spider-web reservoir (github.com/holliscutler/spiderweb_reservoir,
commit c93e4e9): oscillators sit on concentric rings, input oscillators on the rim,
readout oscillators at the hub, and each oscillator couples only to its geometric
neighbours, plus small-world rewiring (Watts & Strogatz).

Choices made here (the reservoir's 201-neuron web does not fix them for N = 2128):
  * Rings are 1 unit apart and ring k (radius k) holds round(2 pi k) oscillators, so the
    density is uniform and a single neighbourhood radius gives near-equal degrees.
    The outermost ring is partly filled, evenly spaced.
  * Readout: the 40 innermost positions. Input: 40 evenly spaced positions on the rim.
  * Edge budget: exactly ``num_edges`` (matched per seed to the random mask). The
    shortest admissible edges are kept, i.e. one global neighbourhood radius.
  * ``inward``: oscillator i receives from j only if j is on the same ring or further
    out (signals flow rim -> hub, as in the reservoir). ``twoway``: any neighbour.
  * Rewiring: each edge, with probability ``rewire_p``, keeps its source and moves to a
    random new target (the reservoir's rule). No anchor thread: rewiring already
    creates outward links, and the classifier needs no recurrent echo.

Masks follow the model convention: ``mask[i, j]`` means i receives from j (K_ij).
"""

from __future__ import annotations

import math
from collections import deque

import torch
from torch import Tensor

from flow_rewire.mnist1d_model import N_IN, N_OUT

DIRECTIONS = ("inward", "twoway")
REWIRE_P = 0.1  # spider-web reservoir default


def ring_positions(n: int) -> tuple[Tensor, Tensor]:
    """Uniform-density concentric rings: (xy positions (n, 2), ring index (n,)), inner first."""
    xy, ring = [], []
    k = 1
    while len(xy) < n:
        full = round(2 * math.pi * k)
        count = min(full, n - len(xy))  # last ring partly filled, evenly spaced
        for s in range(count):
            angle = 2 * math.pi * s / count
            xy.append((k * math.cos(angle), k * math.sin(angle)))
            ring.append(k)
        k += 1
    return torch.tensor(xy, dtype=torch.float64), torch.tensor(ring)


def web_layout(n_hidden: int) -> tuple[Tensor, Tensor]:
    """Positions and ring index per model oscillator (order: input, hidden, readout)."""
    n = N_IN + n_hidden + N_OUT
    xy, ring = ring_positions(n)
    rim = (ring == ring.max()).nonzero().squeeze(1)
    pick = torch.linspace(0, len(rim), N_IN + 1)[:-1].long()
    inputs = rim[pick]
    readout = torch.arange(N_OUT)  # innermost positions
    taken = torch.zeros(n, dtype=torch.bool)
    taken[inputs] = taken[readout] = True
    hidden = (~taken).nonzero().squeeze(1)
    order = torch.cat([inputs, hidden, readout])
    return xy[order], ring[order]


def web_mask(
    n_hidden: int,
    num_edges: int,
    direction: str,
    *,
    rewire_p: float = REWIRE_P,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Keep-mask with exactly ``num_edges`` off-diagonal edges, local plus rewired."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}.")
    xy, ring = web_layout(n_hidden)
    n = xy.shape[0]
    dist = torch.cdist(xy, xy)
    admissible = ~torch.eye(n, dtype=torch.bool)
    if direction == "inward":
        admissible &= ring[None, :] >= ring[:, None]  # source j on same ring or further out
    # Shortest admissible edges; ties broken by index so that twoway stays symmetric.
    i, j = admissible.nonzero(as_tuple=True)
    pair = torch.minimum(i, j) * n + torch.maximum(i, j)
    by_pair = pair.argsort(stable=True)
    d = dist[i, j].round(decimals=9)[by_pair]
    keep = by_pair[d.argsort(stable=True)][:num_edges]
    mask = torch.zeros(n, n, dtype=torch.bool)
    mask[i[keep], j[keep]] = True
    return rewire(mask, rewire_p, generator)


def rewire(mask: Tensor, p: float, generator: torch.Generator | None = None) -> Tensor:
    """Each edge j -> i, with probability p, keeps source j and gets a random new target."""
    mask = mask.clone()
    n = mask.shape[0]
    targets, sources = mask.nonzero(as_tuple=True)
    moved = torch.rand(len(targets), generator=generator) < p
    for i, j in zip(targets[moved].tolist(), sources[moved].tolist(), strict=True):
        for _ in range(100):
            c = int(torch.randint(n, (1,), generator=generator))
            if c != j and not mask[c, j]:
                mask[i, j] = False
                mask[c, j] = True
                break
    return mask


def hops_to_readout(mask: Tensor) -> int | None:
    """Graph hops for input signals to reach every readout oscillator (None if never).

    One Euler step moves information one hop, so this must stay below the 10 steps.
    """
    n = mask.shape[0]
    out_edges = [[] for _ in range(n)]  # j -> receivers i
    for i, j in mask.nonzero().tolist():
        out_edges[j].append(i)
    dist = [-1] * n
    queue = deque(range(N_IN))
    for s in range(N_IN):
        dist[s] = 0
    while queue:
        j = queue.popleft()
        for i in out_edges[j]:
            if dist[i] < 0:
                dist[i] = dist[j] + 1
                queue.append(i)
    readout = dist[n - N_OUT :]
    return None if min(readout) < 0 else max(readout)


def wire_length(mask: Tensor, n_hidden: int) -> float:
    """Total Euclidean length of all couplings on the web layout (ring spacing = 1)."""
    xy, _ = web_layout(n_hidden)
    return float(torch.cdist(xy, xy)[mask].sum())
