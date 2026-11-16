#!/usr/bin/env bash
# Replication: fresh seeds 5-9 at the LRs already selected on seed 0. Nothing is re-tuned.
set -euo pipefail
cd "$(dirname "$0")/../.."
T="uv run python -m flow_rewire.mnist1d_train --hidden 2048 --sparsity 0.9375 --seeds 5 6 7 8 9"
O="--output results/replicate/mnist1d.jsonl"
$T --masks perturbation_prune --lrs 3e-3 $O
$T --masks random --lrs 1e-2 $O
$T --masks magnitude_prune --lrs 3e-3 $O
$T --masks dense --lrs 1e-3 $O
