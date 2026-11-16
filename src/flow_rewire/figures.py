"""Figures for the writeup, built directly from the result records.

Reads every ``results/*/mnist1d.jsonl`` run at N = 2128 (hidden 2048), pools seeds per
method, and writes PNGs to ``docs/figures``. Rerun after new seeds land.

Example:
    uv run python -m flow_rewire.figures
    uv run python -m flow_rewire.figures --seeds 0 1 2 3 4   # original seeds only

"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

RESULT_FILES = ("baseline", "prune", "perturb", "replicate")
N_HIDDEN = 2048

# (record mask, display label, group); display order top to bottom.
METHODS = (
    ("perturbation_prune", "Perturbation flow (ours)", "ours"),
    ("flow_prune", "Force flow", "prune"),
    ("magnitude_prune", "Magnitude", "prune"),
    ("random", "Random mask", "baseline"),
    ("modular", "Modular mask", "baseline"),
    ("dense", "Dense", "baseline"),
)
LABEL = {m: label for m, label, _ in METHODS}

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e0"
ACCENT = "#2a78d6"
COLOR = {"ours": ACCENT, "prune": "#6f6e69", "baseline": "#a9a8a2"}
WIDTH = 7.2


def load(results: Path, seeds: set[int] | None = None) -> dict[str, dict[int, dict]]:
    """Load records at N = 2128, keyed by mask then seed (one LR per mask)."""
    out: dict[str, dict[int, dict]] = {}
    for name in RESULT_FILES:
        path = results / name / "mnist1d.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r["n_hidden"] != N_HIDDEN or r["mask"] not in LABEL:
                continue
            if seeds is None or r["seed"] in seeds:
                out.setdefault(r["mask"], {})[r["seed"]] = r
    return out


def mean_ci(values: list[float]) -> tuple[float, float]:
    """Mean and 95% t-interval half-width."""
    n = len(values)
    m = statistics.mean(values)
    if n < 2:
        return m, 0.0
    return m, stats.t.ppf(0.975, n - 1) * statistics.stdev(values) / math.sqrt(n)


def paired(data: dict, a: str, b: str) -> dict:
    """Paired per-seed difference a − b in accuracy points, with a two-sided t-test."""
    seeds = sorted(set(data[a]) & set(data[b]))
    diffs = [100 * (data[a][s]["acc_final"] - data[b][s]["acc_final"]) for s in seeds]
    m, ci = mean_ci(diffs)
    p = stats.ttest_1samp(diffs, 0.0).pvalue if len(diffs) > 1 else float("nan")
    return {
        "seeds": seeds,
        "diffs": diffs,
        "mean": m,
        "ci": ci,
        "p": p,
        "wins": sum(d > 0 for d in diffs),
    }


def style(ax: plt.Axes, grid_axis: str = "x") -> None:
    """Recessive axes: light grid, no top/right spines."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid_axis == "x":
        ax.spines["left"].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, length=0, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def title(ax: plt.Axes, text: str, sub: str) -> None:
    """Left-aligned title with a muted subtitle line, aligned to the figure edge."""
    ax.set_title(f"{text}\n", loc="left", fontsize=10.5, color=INK, fontweight="bold", pad=6)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")


def fig_accuracy(data: dict, out: Path) -> None:
    """Test accuracy per method: one dot per seed, mean with 95% CI."""
    rows = [(m, lab, g) for m, lab, g in METHODS if m in data]
    fig, ax = plt.subplots(figsize=(WIDTH, 0.48 * len(rows) + 1.3), dpi=200)
    for y, (mask, _, group) in enumerate(reversed(rows)):
        accs = [r["acc_final"] for r in data[mask].values()]
        m, ci = mean_ci(accs)
        color = COLOR[group]
        ax.scatter(accs, [y - 0.18] * len(accs), s=13, color=color, alpha=0.5, lw=0)
        ax.errorbar(m, y + 0.08, xerr=ci, fmt="none", ecolor=color, elinewidth=2)
        ax.scatter([m], [y + 0.08], s=44, color=color, edgecolor="white", lw=1.5, zorder=3)
        ax.text(m + ci + 0.002, y + 0.08, f"{m:.3f}", va="center", fontsize=8.5, color=INK)
    labels = [f"{lab}  ({len(data[m])})" for m, lab, _ in reversed(rows)]
    ax.set_yticks(range(len(rows)), labels)
    for tick, (_, _, group) in zip(ax.get_yticklabels(), reversed(rows), strict=True):
        tick.set_color(INK if group == "ours" else INK_2)
        tick.set_fontweight("bold" if group == "ours" else "normal")
    ax.set_xlabel("Test accuracy  (dots: seeds; bar: mean ± 95% CI)", color=INK_2, fontsize=9)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    style(ax)
    title(
        ax,
        "MNIST1D test accuracy, N = 2128",
        "Sparse methods all keep 282,912 couplings (6.25%); seeds in parentheses",
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_paired(data: dict, out: Path) -> list[tuple[str, dict]]:
    """Per-seed paired differences of perturbation-flow pruning against each alternative."""
    others = [m for m in ("random", "magnitude_prune", "flow_prune", "dense") if m in data]
    results = [(m, paired(data, "perturbation_prune", m)) for m in others]
    right = max(max(*r["diffs"], r["mean"] + r["ci"]) for _, r in results) + 0.6
    fig, ax = plt.subplots(figsize=(WIDTH, 0.5 * len(results) + 1.3), dpi=200)
    for y, (_, res) in enumerate(reversed(results)):
        ax.scatter(
            res["diffs"], [y - 0.16] * len(res["diffs"]), s=13, color=ACCENT, alpha=0.5, lw=0
        )
        ax.errorbar(res["mean"], y + 0.08, xerr=res["ci"], fmt="none", ecolor=ACCENT, elinewidth=2)
        ax.scatter(
            [res["mean"]], [y + 0.08], s=44, color=ACCENT, edgecolor="white", lw=1.5, zorder=3
        )
        n = len(res["diffs"])
        ax.text(
            right,
            y,
            f"{res['mean']:+.1f} pts   {res['wins']}/{n} seeds   p = {res['p']:.3f}",
            va="center",
            fontsize=8.5,
            color=INK,
        )
    ax.axvline(0, color=INK_2, lw=1)
    ax.set_yticks(range(len(results)), [f"vs {LABEL[m]}" for m, _ in reversed(results)])
    ax.set_xlabel("Accuracy gain, points  (same seed, same edge count)", color=INK_2, fontsize=9)
    ax.set_ylim(-0.6, len(results) - 0.4)
    ax.set_xlim(min(ax.get_xlim()[0], -1), right + 6.5)
    style(ax)
    title(
        ax,
        "Perturbation-flow pruning, paired by seed",
        "Dots: per-seed difference; bar: mean ± 95% CI; two-sided paired t-test",
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return results


def fig_cost(data: dict, out: Path) -> None:
    """Mean accuracy against coupling operations per inference (hardware proxy)."""
    fig, ax = plt.subplots(figsize=(WIDTH, 3.4), dpi=200)
    for mask, label, group in METHODS:
        if mask not in data:
            continue
        recs = list(data[mask].values())
        ops = recs[0]["coupling_ops_per_inference"]
        m = statistics.mean(r["acc_final"] for r in recs)
        ax.scatter([ops], [m], s=48, color=COLOR[group], edgecolor="white", lw=1.2, zorder=3)
        dense = mask == "dense"
        ax.annotate(
            label,
            (ops, m),
            xytext=(-9 if dense else 9, 0),
            textcoords="offset points",
            ha="right" if dense else "left",
            va="center",
            fontsize=8.5,
            color=INK if group == "ours" else INK_2,
            fontweight="bold" if group == "ours" else "normal",
        )
    ax.set_xscale("log")
    ax.set_xlim(1.5e6, 1.0e8)
    ax.set_xlabel(
        "Coupling operations per inference  (edges × 10 Euler steps, log scale)",
        color=INK_2,
        fontsize=9,
    )
    ax.set_ylabel("Mean test accuracy", color=INK_2, fontsize=9)
    style(ax, grid_axis="both")
    title(
        ax,
        "Accuracy vs. coupling cost",
        "Pruned networks use 16× fewer coupling operations than dense",
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--output", type=Path, default=Path("docs/figures"))
    p.add_argument("--seeds", nargs="+", type=int, default=None, help="default: all seeds")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = load(args.results, None if args.seeds is None else set(args.seeds))
    fig_accuracy(data, args.output / "accuracy.png")
    results = fig_paired(data, args.output / "paired.png")
    fig_cost(data, args.output / "cost.png")
    for mask, r in results:
        print(
            f"perturbation - {mask}: {r['mean']:+.2f} ± {r['ci']:.2f} (95% CI), "
            f"p = {r['p']:.4f}, wins {r['wins']}/{len(r['diffs'])}, seeds {r['seeds']}"
        )


if __name__ == "__main__":
    main()
