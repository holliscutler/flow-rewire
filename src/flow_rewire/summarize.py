"""Summarize MNIST1D result files as a mean ± std markdown table.

Example:
    uv run python -m flow_rewire.summarize results/baseline/mnist1d.jsonl

"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

GROUP_KEYS = ("mask", "level", "n_hidden", "lr")


def summarize(paths: list[Path]) -> str:
    """Group records by configuration and return a markdown table."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for path in paths:
        for line in path.read_text().splitlines():
            record = json.loads(line)
            groups[tuple(record[k] for k in GROUP_KEYS)].append(record)

    rows = [
        "| mask | level | N | edges | coupling ops / inf. | LR | seeds | final acc. | "
        "best acc. | train acc. | train energy (kJ) | inference energy (mJ / sample) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key in sorted(groups, key=lambda k: (k[2], k[0], k[1], k[3])):
        recs = groups[key]

        def stat(field: str, scale: float = 1.0, fmt: str = ".3f", recs=recs) -> str:
            values = [r[field] * scale for r in recs if r.get(field) is not None]
            if not values:
                return "n/a"
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            return f"{statistics.mean(values):{fmt}} ± {sd:{fmt}}"

        mask, level, _, lr = key
        ops = recs[0].get("coupling_ops_per_inference")
        rows.append(
            f"| {mask} | {level:g} | {recs[0]['n_oscillators']} | {recs[0]['edges']:,} | "
            f"{'n/a' if ops is None else f'{ops:.2e}'} | {lr:g} | {len(recs)} | "
            f"{stat('acc_final')} | {stat('acc_best')} | {stat('train_acc')} | "
            f"{stat('train_energy_j', 1e-3, '.2f')} | "
            f"{stat('infer_energy_j_per_sample', 1e3, '.3f')} |"
        )
    return "\n".join(rows)


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("paths", nargs="+", type=Path)
    print(summarize(p.parse_args().paths))


if __name__ == "__main__":
    main()
