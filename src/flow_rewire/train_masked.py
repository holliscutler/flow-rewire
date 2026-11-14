"""Train Un-0 on CIFAR-10 with a fixed sparsity mask on the main-pool coupling K.

Runs Un-0's own ``train_cifar10.train`` unchanged; the only intervention is that the
model builder is wrapped so that ``K`` is parametrized as ``K_raw * M`` for a fixed
binary mask ``M``. Masked edges are zero in every forward pass and receive zero
gradient, matching the static-mask protocol of "Less is More" (Unconventional AI, 2026).

Example:
    uv run python -m flow_rewire.train_masked --mask results/masks/flow_s0.9.pt \
        --n-oscillators 1024 --lr 2.683e-3 --epochs 1 --checkpoint-dir results/raw/run

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn.utils import parametrize
from un0 import train_cifar10

from flow_rewire.flow import unwrap


class FixedMask(nn.Module):
    """Parametrization ``K -> K * mask`` with a non-trainable buffer mask."""

    def __init__(self, mask: Tensor) -> None:
        """Store the mask as a float buffer."""
        super().__init__()
        self.register_buffer("mask", mask.to(torch.float32))

    def forward(self, K: Tensor) -> Tensor:
        """Apply the mask."""
        return K * self.mask.to(K.dtype)


def attach_mask(model: nn.Module, mask: Tensor) -> None:
    """Parametrize the main-pool coupling of ``model`` with a fixed mask."""
    dyn = unwrap(model.dynamics)
    if mask.shape != dyn.K.shape:
        raise ValueError(f"Mask shape {tuple(mask.shape)} != K shape {tuple(dyn.K.shape)}.")
    parametrize.register_parametrization(dyn, "K", FixedMask(mask.to(dyn.K.device)))


def main() -> None:
    """Parse ``--mask``, forward every other flag to Un-0's training CLI."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--mask", type=Path, default=None, help="Bool (n, n) mask; omit = dense.")
    ours, rest = pre.parse_known_args()

    if ours.mask is not None:
        mask = torch.load(ours.mask, map_location="cpu", weights_only=True)
        builder = train_cifar10.build_cifar10_model

        def masked_builder(**kwargs):
            model = builder(**kwargs)
            attach_mask(model, mask)
            return model

        train_cifar10.build_cifar10_model = masked_builder

    sys.argv = [sys.argv[0], *rest]
    train_cifar10.main()


if __name__ == "__main__":
    main()
