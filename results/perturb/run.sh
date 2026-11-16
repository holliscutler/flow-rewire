#!/usr/bin/env bash
# Stage 1: LR selection on seed 0; stage 2: seeds 1-4 at the selected LR (by final accuracy).
set -euo pipefail
cd "$(dirname "$0")/../.."
T="uv run python -m flow_rewire.mnist1d_train --hidden 2048 --sparsity 0.9375 --masks perturbation_prune"
$T --seeds 0 --lrs 1e-3 3e-3 1e-2 --output results/perturb/lr_select.jsonl
lr=$(python3 -c "
import json
rs=[json.loads(l) for l in open('results/perturb/lr_select.jsonl')]
best=max(rs, key=lambda r: r['acc_final']); print(best['lr'])
open('results/perturb/mnist1d.jsonl','a').write(json.dumps(best)+'\n')")
echo "perturbation_prune selected lr=$lr"
$T --seeds 1 2 3 4 --lrs $lr --output results/perturb/mnist1d.jsonl
