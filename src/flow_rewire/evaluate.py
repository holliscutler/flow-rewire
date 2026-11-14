"""Prune a released Un-0 checkpoint and score it with Un-0's own clean-FID pipeline.

Example (smoke test, ~1 min):
    uv run python -m flow_rewire.evaluate --method flow --sparsity 0.9 --num-samples 2000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from un0.common import disable_torchscript_gpu_fuser_on_blackwell, seed_everything
from un0.data import NUM_CLASSES
from un0.metrics import compute_fid
from un0.model import ConditionalImplicitKuramotoGenerator

from flow_rewire.flow import effective_coupling, measure_flow
from flow_rewire.masks import apply_mask, magnitude_score, random_score, topk_mask

METHODS = ("dense", "random", "magnitude", "flow")


def edge_score(
    model: ConditionalImplicitKuramotoGenerator,
    method: str,
    *,
    flow_samples: int,
    seed: int,
) -> torch.Tensor:
    """Score every main-pool edge with the requested criterion."""
    K = effective_coupling(model.dynamics)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if method == "random":
        return random_score(K.shape[0], generator=gen).to(K.device)
    if method == "magnitude":
        return magnitude_score(K)
    if method == "flow":
        dev_gen = torch.Generator(device=K.device).manual_seed(seed)
        class_id = torch.arange(flow_samples, device=K.device) % NUM_CLASSES
        return measure_flow(model, class_id, generator=dev_gen).flow
    raise ValueError(f"Unknown method {method!r}; choose from {METHODS}.")


def run(args: argparse.Namespace) -> dict:
    """Build, prune, and score one configuration. Returns a JSON-serialisable record."""
    seed_everything(args.seed)
    disable_torchscript_gpu_fuser_on_blackwell()
    model = ConditionalImplicitKuramotoGenerator.from_pretrained(args.pretrained).eval()

    start = time.perf_counter()
    if args.method != "dense":
        score = edge_score(model, args.method, flow_samples=args.flow_samples, seed=args.seed)
        apply_mask(model, topk_mask(score, args.sparsity))
    prune_seconds = time.perf_counter() - start

    seed_everything(args.seed)  # identical sampling noise across methods
    fid = compute_fid(
        model,
        num_samples=args.num_samples,
        num_classes=NUM_CLASSES,
        batch_size=args.batch_size,
        device=next(model.parameters()).device,
    )
    return {
        "pretrained": args.pretrained,
        "method": args.method,
        "sparsity": 0.0 if args.method == "dense" else args.sparsity,
        "fid": fid,
        "num_samples": args.num_samples,
        "flow_samples": args.flow_samples,
        "seed": args.seed,
        "prune_seconds": round(prune_seconds, 2),
        "total_seconds": round(time.perf_counter() - start, 2),
    }


def build_parser() -> argparse.ArgumentParser:
    """CLI parser."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--pretrained", default="cifar10/n1024")
    p.add_argument("--method", choices=METHODS, default="dense")
    p.add_argument("--sparsity", type=float, default=0.75)
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--flow-samples", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=None, help="Append the record to this JSONL.")
    return p


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    record = run(args)
    print(json.dumps(record))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
