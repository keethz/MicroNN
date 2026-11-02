"""Plots for results/search_*.json (claims C1-C4).

  .venv/bin/python -m bench.plot_search results/search_synthetic_100000.json
"""
import json
import os
import sys

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reference palette, light mode (validated: bench uses 2 categorical slots).
SURFACE, INK, INK2, GRID = "#fcfcfb", "#1f1f1e", "#6b6a64", "#e6e5e0"
SERIES = {"micronn": "#2a78d6", "inmemory": "#eb6834"}
LABEL = {"micronn": "MicroNN (SQLite, disk)", "inmemory": "In-memory IVF"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "lines.linewidth": 2, "lines.markersize": 6,
})


def _label_end(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                color=INK2, fontsize=9)


def recall_latency(res, out):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for system in ("micronn", "inmemory"):
        rows = sorted((r for r in res["sweep"] if r["system"] == system and r["balance"] == 0.0),
                      key=lambda r: r["nprobe"])
        x = [r["recall"] for r in rows]
        y = [r["median_ms"] for r in rows]
        ax.plot(x, y, marker="o", color=SERIES[system], label=LABEL[system],
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        _label_end(ax, x[-1], y[-1], LABEL[system].split(" (")[0])
    ax.axvline(0.9, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.axhline(7, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.text(0.9, 0.02, " 90% recall", va="bottom", color=INK2, fontsize=9, transform=ax.get_xaxis_transform())
    ax.text(ax.get_xlim()[0], 7, "paper: 7 ms", va="bottom", color=INK2, fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("recall@100")
    ax.set_ylabel("median query latency (ms, log)")
    ax.set_title(f"C1/C3 — latency vs recall ({res['dataset']}, N={res['n']:,})", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def memory_scaling(res, out):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for system in ("micronn", "inmemory"):
        rows = sorted((m for m in res["memory"] if m["system"] == system), key=lambda m: m["n"])
        x = [m["n"] for m in rows]
        y = [m["tracemalloc_peak"] / 2 ** 20 for m in rows]
        ax.plot(x, y, marker="o", color=SERIES[system], label=LABEL[system],
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        _label_end(ax, x[-1], y[-1], f"{y[-1]:.1f} MiB")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1e3:,.0f}K"))
    ax.set_xlabel("vectors in index (N)")
    ax.set_ylabel("peak query-time memory (MiB, log)")
    ax.set_title(f"C2 — memory vs N at nprobe={res['nprobe90']}", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(right=max(m["n"] for m in res["memory"]) * 1.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def partition_sizes(res, out):
    builds = {b["balance"]: b for b in res["builds"]}
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    hi = max(max(b["sizes"]) for b in builds.values())
    bins = np.linspace(0, hi, 40)
    for ax, (bal, title) in zip(axes, ((0.0, "no penalty"), (1.0, "balance penalty λ=1"))):
        b = builds[bal]
        ax.hist(b["sizes"], bins=bins, color=SERIES["micronn"], edgecolor=SURFACE, linewidth=1)
        ax.set_title(f"{title}\nstd={b['size_std']:.0f}, max={b['size_max']}", loc="left", fontsize=10)
        ax.set_xlabel("vectors per partition")
    axes[0].set_ylabel("partitions")
    fig.suptitle(f"C4 — partition sizes (k={builds[0.0]['n_partitions']})", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main(path):
    with open(path) as f:
        res = json.load(f)
    stem = os.path.splitext(path)[0]
    recall_latency(res, stem + "_c1_c3.png")
    memory_scaling(res, stem + "_c2.png")
    partition_sizes(res, stem + "_c4.png")
    print("wrote", stem + "_{c1_c3,c2,c4}.png")


if __name__ == "__main__":
    main(sys.argv[1])
