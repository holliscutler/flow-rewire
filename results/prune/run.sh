#!/usr/bin/env bash
# Stage 1: LR selection on seed 0; stage 2: seeds 1-4 at the selected LR (by final accuracy).
set -euo pipefail
cd "$(dirname "$0")/../.."
T="uv run python -m flow_rewire.mnist1d_train --hidden 2048 --sparsity 0.9375"
$T --masks flow_prune magnitude_prune --seeds 0 --lrs 1e-3 3e-3 1e-2 --output results/prune/lr_select.jsonl
for m in flow_prune magnitude_prune; do
  lr=$(python3 -c "
import json
rs=[json.loads(l) for l in open('results/prune/lr_select.jsonl')]
rs=[r for r in rs if r['mask']=='$m']
best=max(rs, key=lambda r: r['acc_final']); print(best['lr'])
import sys; open('results/prune/mnist1d.jsonl','a').write(json.dumps(best)+'\n')")
  echo "$m selected lr=$lr"
  $T --masks $m --seeds 1 2 3 4 --lrs $lr --output results/prune/mnist1d.jsonl
done
