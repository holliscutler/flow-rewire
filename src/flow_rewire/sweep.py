"""Sweep pruning methods x sparsities on one released checkpoint, in one process.

The model is loaded (and compiled) once; ``K`` is restored before each configuration.
Each record also logs the final-time Kuramoto order parameter R(T), a direct check of
the synchronization hypothesis.

Example (~1 min per configuration at 10k samples on an RTX 5060):
    uv run python -m flow_rewire.sweep --sparsities 0.9 0.98 --num-samples 10000
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

from flow_rewire.evaluate import edge_score
from flow_rewire.flow import measure_flow, unwrap
from flow_rewire.masks import apply_mask, topk_mask


def final_order_parameter(model: ConditionalImplicitKuramotoGenerator, seed: int) -> float:
    """Mean |<exp(i theta)>| of the main pool at t = T over 500 class-balanced samples."""
    device = next(model.parameters()).device
    class_id = torch.arange(500, device=device) % NUM_CLASSES
    gen = torch.Generator(device=device).manual_seed(seed + 1)
    return float(measure_flow(model, class_id, generator=gen).order_parameter[-1])


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--pretrained", default="cifar10/n1024")
    p.add_argument("--methods", nargs="+", default=["random", "magnitude", "flow"])
    p.add_argument("--sparsities", nargs="+", type=float, default=[0.5, 0.75, 0.9, 0.95, 0.98])
    p.add_argument("--num-samples", type=int, default=10000)
    p.add_argument("--flow-samples", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=Path("results/sweep.jsonl"))
    args = p.parse_args()

    disable_torchscript_gpu_fuser_on_blackwell()
    model = ConditionalImplicitKuramotoGenerator.from_pretrained(args.pretrained).eval()
    K = unwrap(model.dynamics).K
    K_dense = K.detach().clone()
    device = K.device
    args.output.parent.mkdir(parents=True, exist_ok=True)

    configs = [("dense", 0.0)] + [(m, s) for s in args.sparsities for m in args.methods]
    for method, sparsity in configs:
        start = time.perf_counter()
        with torch.no_grad():
            K.copy_(K_dense)
        if method != "dense":
            score = edge_score(model, method, flow_samples=args.flow_samples, seed=args.seed)
            apply_mask(model, topk_mask(score, sparsity))
        seed_everything(args.seed)
        fid = compute_fid(
            model,
            num_samples=args.num_samples,
            num_classes=NUM_CLASSES,
            batch_size=args.batch_size,
            device=device,
        )
        record = {
            "pretrained": args.pretrained,
            "method": method,
            "sparsity": sparsity,
            "fid": fid,
            "order_parameter_T": final_order_parameter(model, args.seed),
            "num_samples": args.num_samples,
            "seed": args.seed,
            "seconds": round(time.perf_counter() - start, 1),
        }
        print(json.dumps(record), flush=True)
        with args.output.open("a") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
