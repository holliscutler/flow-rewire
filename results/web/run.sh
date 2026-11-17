#!/usr/bin/env bash
# Spider-web masks (src/flow_rewire/web.py), inward and two-way, 10% rewiring, edges matched
# per seed to the random mask. Same protocol as every other method: LR selected on seed 0
# from {1e-3, 3e-3, 1e-2} by final accuracy, then seeds 1-9 at the selected LR.
set -euo pipefail
cd "$(dirname "$0")/../.."
for kind in web_inward web_twoway; do
  T="uv run python -m flow_rewire.mnist1d_train --hidden 2048 --sparsity 0.9375 --masks $kind"
  $T --seeds 0 --lrs 1e-3 3e-3 1e-2 --output results/web/lr_select.jsonl
  lr=$(python3 -c "
import json
rs=[json.loads(l) for l in open('results/web/lr_select.jsonl') if json.loads(l)['mask']=='$kind']
best=max(rs, key=lambda r: r['acc_final']); print(best['lr'])
open('results/web/mnist1d.jsonl','a').write(json.dumps(best)+'\n')")
  echo "$kind selected lr=$lr"
  $T --seeds 1 2 3 4 5 6 7 8 9 --lrs $lr --output results/web/mnist1d.jsonl
done
