"""Plots for claims C7 (batch_*.json) and C8 (updates_*.json).

  .venv/bin/python -m bench.plot_updates results/batch_sift_1000000.json results/updates_sift_1000000.json
"""
import json
import os
import sys

from bench.plot_search import INK2, SURFACE, plt

# Reference palette slots 1-3 (validated together in plot_filter).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def batch_plot(res, out):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for color, nprobe in zip(COLORS, sorted({r["nprobe"] for r in res["rows"]})):
        rows = [r for r in res["rows"] if r["nprobe"] == nprobe]
        seq = next(r for r in rows if r["batch"] == 0)
        pts = sorted((r for r in rows if r["batch"] > 0), key=lambda r: r["batch"])
        x = [r["batch"] for r in pts]
        y = [r["per_query_ms"] for r in pts]
        ax.plot(x, y, color=color, marker="o", markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=f"nprobe={nprobe} (recall {seq['recall']:.2f})")
        ax.axhline(seq["per_query_ms"], color=color, lw=1, ls=(0, (3, 3)))
        ax.annotate(f"−{pts[-1]['vs_sequential']:.0%}", (x[-1], y[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK2, fontsize=9)
    ax.plot([], [], color=INK2, lw=1, ls=(0, (3, 3)), label="one query at a time (same nprobe)")
    ax.set_xscale("log", base=2)
    ticks = sorted({r["batch"] for r in res["rows"] if r["batch"] > 0})
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xlim(right=ticks[-1] * 2)
    ax.set_yscale("log")
    ax.set_xlabel("batch size (queries)")
    ax.set_ylabel("average latency per query (ms, log)")
    ax.set_title(f"C7 — batched queries ({res['dataset']}, N={res['n']:,})", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def updates_plot(res, out):
    workloads = sorted({r["workload"] for r in res["runs"]}, key=["random", "skewed"].index)
    fig, axes = plt.subplots(2, len(workloads), figsize=(10, 6.4), sharex=True)
    policies = ["never", "merge", "merge+rebuild"]
    for col, w in enumerate(workloads):
        for color, p in zip(COLORS, policies):
            run = next(r for r in res["runs"] if r["workload"] == w and r["policy"] == p)
            x = [r["inserted"] / 1e3 for r in run["rows"]]
            style = dict(color=color, marker="o", markeredgecolor=SURFACE, markeredgewidth=1.5,
                         label=f"{p} ({run['throughput']:,.0f} inserts/s)")
            axes[0, col].plot(x, [r["median_ms"] for r in run["rows"]], **style)
            axes[1, col].plot(x, [r["recall"] for r in run["rows"]], **style)
        axes[0, col].set_title(f"{w} inserts", loc="left", fontsize=11)
        axes[0, col].set_yscale("log")
        axes[1, col].set_xlabel("vectors inserted after build (thousands)")
        axes[0, col].legend(frameon=False, fontsize=8, loc="upper left")
    axes[0, 0].set_ylabel("median latency (ms, log)")
    axes[1, 0].set_ylabel("recall@100")
    fig.suptitle(f"C8 — queries during inserts ({res['dataset']}, nprobe={res['nprobe']})",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main(paths):
    for path in paths:
        with open(path) as f:
            res = json.load(f)
        stem = os.path.splitext(path)[0]
        if os.path.basename(path).startswith("batch_"):
            batch_plot(res, stem + "_c7.png")
            print("wrote", stem + "_c7.png")
        else:
            updates_plot(res, stem + "_c8.png")
            print("wrote", stem + "_c8.png")


if __name__ == "__main__":
    main(sys.argv[1:])
