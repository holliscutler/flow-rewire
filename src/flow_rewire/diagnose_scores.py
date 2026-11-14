"""Compare edge scores (magnitude, force flow, perturbation flow) on a partly trained dense model.

Trains a dense N = 2128 classifier briefly with the baseline recipe, then scores every edge
on one training batch and reports rank correlations, overlap of the kept edge sets at the
target density, and how each score treats in-phase (cos dtheta ~ 1) edges.

Example (~1 min):
    uv run python -m flow_rewire.diagnose_scores --steps 2000

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from flow_rewire import mnist1d_train as tr
from flow_rewire.flow import abs_edge_flow, abs_perturbation_flow
from flow_rewire.masks import topk_mask_count
from flow_rewire.mnist1d_model import KuramotoClassifier


@torch.enable_grad()
def train_dense(n_hidden: int, steps: int, lr: float, seed: int, device: str) -> KuramotoClassifier:
    """Short dense training run with the baseline optimiser settings."""
    x, y, _, _ = tr.load_mnist1d(device)
    torch.manual_seed(seed)
    model = KuramotoClassifier(n_hidden).to(device)
    opt = tr.DecoupledAdamW(model.parameters(), lr=lr, betas=tr.BETAS, weight_decay=tr.WEIGHT_DECAY)
    gen = torch.Generator(device=device).manual_seed(seed)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        idx = torch.randint(0, x.shape[0], (tr.BATCH_SIZE,), device=device, generator=gen)
        loss = loss_fn(model(x[idx], gen), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tr.GRAD_CLIP_NORM)
        opt.step()
    return model


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation of two flat tensors."""
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()))


@torch.no_grad()
def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep", type=float, default=0.0625, help="kept edge fraction")
    p.add_argument("--output", type=Path, default=Path("results/diagnostics/scores.json"))
    args = p.parse_args()
    device = "cuda"

    model = train_dense(args.hidden, args.steps, args.lr, args.seed, device)
    x = tr.load_mnist1d(device)[0][:100]
    gen = lambda: torch.Generator(device=device).manual_seed(123)  # noqa: E731
    K = model.coupling

    force_states = model.trajectory(x, gen())
    pert_states = model.trajectory(x, gen(), shared_noise=True)
    scores = {
        "magnitude": K.abs(),
        "force_flow": sum(abs_edge_flow(t, K, chunk=10) for t in force_states),
        "perturbation_flow": sum(abs_perturbation_flow(t, K, chunk=10) for t in pert_states),
    }
    # In-phase-ness of each edge: trajectory-mean cos(theta_bar_j - theta_bar_i).
    cos_bar = sum(
        torch.cos(m[None, :] - m[:, None])
        for m in (torch.angle(torch.exp(1j * t).mean(0)) for t in pert_states)
    ) / len(pert_states)

    off = ~torch.eye(model.n, dtype=torch.bool, device=device)
    flat = {k: v[off] for k, v in scores.items()}
    num_keep = round(args.keep * int(off.sum()))
    kept = {k: topk_mask_count(v, num_keep)[off] for k, v in scores.items()}
    locked = cos_bar[off] > 0.9  # strongly in-phase edges

    names = list(scores)
    report = {
        "config": vars(args) | {"output": str(args.output)},
        "spearman": {
            f"{a}~{b}": spearman(flat[a], flat[b])
            for i, a in enumerate(names)
            for b in names[i + 1 :]
        },
        "kept_overlap": {
            f"{a}&{b}": float((kept[a] & kept[b]).sum() / num_keep)
            for i, a in enumerate(names)
            for b in names[i + 1 :]
        },
        "in_phase_edges_fraction": float(locked.float().mean()),
        "in_phase_edges_kept_fraction": {k: float(kept[k][locked].float().mean()) for k in names},
        "all_edges_kept_fraction": args.keep,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
