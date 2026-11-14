"""Train Kuramoto MNIST1D classifiers across hidden sizes, coupling masks and learning rates.

Protocol sources (see docs/mnist1d_spec.md):
  * Optimizer/schedule from the Un-0 CIFAR-10 release: DecoupledAdamW, betas (0.9, 0.95),
    weight decay 1e-3, 10% linear warmup then linear decay to 0, grad-norm clip 1.0.
  * Data/batch/steps/evaluation from the MNIST1D reference trainer: default dataset
    (4000 train / 1000 test), batch 100, 8000 steps, accuracy on the test split every
    250 steps (the reference has no separate validation split).

Each JSON line reports final and best accuracy for one (mask, level, H, seed, lr).

Example (dense trend check):
    uv run python -m flow_rewire.mnist1d_train --hidden 8 512 --masks dense
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import time
from pathlib import Path

import torch
from mnist1d.data import get_dataset_args, make_dataset
from torch import Tensor, nn
from un0.common import make_scheduler
from un0.decoupled_adamw import DecoupledAdamW

from flow_rewire.masks import modular_mask, random_mask
from flow_rewire.mnist1d_model import N_IN, N_OUT, KuramotoClassifier

BATCH_SIZE = 100  # MNIST1D reference
TOTAL_STEPS = 8000  # MNIST1D reference
EVAL_EVERY = 250  # MNIST1D reference
BETAS = (0.9, 0.95)  # Un-0 CIFAR-10
WEIGHT_DECAY = 1e-3  # Un-0 CIFAR-10
WARMUP_FRACTION = 0.1  # Un-0 CIFAR-10
GRAD_CLIP_NORM = 1.0  # Un-0 CIFAR-10


@functools.cache
def load_mnist1d(device: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Default MNIST1D dataset (generator seed 42): 4000 train / 1000 test, length 40."""
    data = make_dataset(get_dataset_args())
    as_t = functools.partial(torch.as_tensor, device=device)
    return (
        as_t(data["x"], dtype=torch.float32),
        as_t(data["y"], dtype=torch.long),
        as_t(data["x_test"], dtype=torch.float32),
        as_t(data["y_test"], dtype=torch.long),
    )


def build_mask(kind: str, n_hidden: int, level: float, seed: int) -> Tensor | None:
    """Keep-mask for one configuration; ``level`` is sparsity (random) or Z (modular)."""
    n = N_IN + n_hidden + N_OUT
    if kind == "dense":
        return None
    if kind == "random":
        return random_mask(n, level, generator=torch.Generator().manual_seed(10_000 + seed))
    if kind == "modular":
        return modular_mask([N_IN, n_hidden, N_OUT], int(level))
    raise ValueError(f"Unknown mask kind {kind!r}.")


@torch.no_grad()
def accuracy(model: KuramotoClassifier, x: Tensor, y: Tensor) -> float:
    """Accuracy with a fixed initial-phase draw, so evaluations are comparable."""
    gen = torch.Generator(device=x.device).manual_seed(0)
    model.eval()
    acc = (model(x, gen).argmax(dim=1) == y).float().mean().item()
    model.train()
    return acc


def train_one(*, n_hidden: int, mask: Tensor | None, seed: int, lr: float, device: str) -> dict:
    """Train one classifier and return its accuracy curve summary."""
    x, y, x_test, y_test = load_mnist1d(device)
    torch.manual_seed(seed)
    model = KuramotoClassifier(n_hidden, mask=mask).to(device)
    opt = DecoupledAdamW(model.parameters(), lr=lr, betas=BETAS, weight_decay=WEIGHT_DECAY)
    sched = make_scheduler(opt, total_steps=TOTAL_STEPS, warmup_fraction=WARMUP_FRACTION)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator(device=device).manual_seed(seed)

    curve = []
    start = time.perf_counter()
    for step in range(1, TOTAL_STEPS + 1):
        idx = torch.randint(0, x.shape[0], (BATCH_SIZE,), device=device, generator=gen)
        loss = loss_fn(model(x[idx], gen), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        opt.step()
        sched.step()
        if step % EVAL_EVERY == 0:
            curve.append(accuracy(model, x_test, y_test))
    return {
        "acc_final": curve[-1],
        "acc_best": max(curve),
        "train_acc": accuracy(model, x, y),
        "n_oscillators": model.n,
        "edges": model.num_edges,
        "seconds": round(time.perf_counter() - start, 1),
    }


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--hidden", nargs="+", type=int, default=[8, 512])
    p.add_argument("--masks", nargs="+", default=["dense"], choices=["dense", "random", "modular"])
    p.add_argument("--sparsity", nargs="+", type=float, default=[0.5, 0.875])
    p.add_argument("--modules", nargs="+", type=int, default=[4, 16])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--lrs", nargs="+", type=float, default=[1e-3, 3e-3, 1e-2])
    p.add_argument("--output", type=Path, default=Path("results/mnist1d.jsonl"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    levels = {"dense": [0.0], "random": args.sparsity, "modular": args.modules}
    for n_hidden, kind in itertools.product(args.hidden, args.masks):
        for level, seed, lr in itertools.product(levels[kind], args.seeds, args.lrs):
            mask = build_mask(kind, n_hidden, level, seed)
            record = {"mask": kind, "level": level, "n_hidden": n_hidden, "seed": seed, "lr": lr}
            record |= train_one(n_hidden=n_hidden, mask=mask, seed=seed, lr=lr, device=device)
            print(json.dumps(record), flush=True)
            with args.output.open("a") as f:
                f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
