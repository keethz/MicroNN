"""Plots for results/filter_*.json (claims C5-C6).

  .venv/bin/python -m bench.plot_filter results/filter_sift_1000000.json
"""
import json
import os
import sys

from bench.plot_search import INK2, SURFACE, plt

# Reference palette slots 1-3 (validated: blue / orange / aqua).
PLANS = {"pre": ("#2a78d6", "Pre-filter"), "post": ("#eb6834", "Post-filter"),
         "auto": ("#1baf7a", "Optimizer (auto)")}


def selectivity_sweep(res, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    sweep = res["sweep"]
    x = [s["true_sel"] for s in sweep]
    for plan, (color, label) in PLANS.items():
        rows = [next(r for r in s["rows"] if r["plan"] == plan) for s in sweep]
        style = dict(color=color, label=label, marker="o", markeredgecolor=SURFACE, markeredgewidth=1.5)
        if plan == "auto":
            style.update(ls=(0, (4, 2)), marker="s", markersize=5)
        ax1.plot(x, [r["median_ms"] for r in rows], **style)
        ax2.plot(x, [r["recall"] for r in rows], **style)
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlabel("filter selectivity (fraction of rows matching)")
    ax1.set_yscale("log")
    ax1.set_ylabel("median latency (ms, log)")
    ax1.set_title("Latency", loc="left", fontsize=11)
    ax2.set_ylabel(f"recall@{res['k']}")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(0.9, color=INK2, lw=1, ls=(0, (3, 3)))
    ax2.text(ax2.get_xlim()[0], 0.9, " 90% recall", va="bottom", color=INK2, fontsize=9)
    ax2.set_title("Recall", loc="left", fontsize=11)
    ax1.legend(frameon=False, loc="upper left")
    fig.suptitle(f"C5/C6 — hybrid queries vs selectivity ({res['dataset']}, N={res['n']:,}, nprobe={res['nprobe']})",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def markdown_table(res) -> str:
    lines = ["| filter | true sel | est sel | q-error | pre recall / ms | post recall / ms | auto chose | best |",
             "|---|---|---|---|---|---|---|---|"]
    for s in res["sweep"] + res["filters"]:
        r = {x["plan"]: x for x in s["rows"]}
        lines.append(f"| {s['name']} | {s['true_sel']:.5f} | {s['est_sel']:.5f} | {s['q_error']:.1f} | "
                     f"{r['pre']['recall']:.2f} / {r['pre']['median_ms']:.1f} | "
                     f"{r['post']['recall']:.2f} / {r['post']['median_ms']:.1f} | "
                     f"{r['auto']['chosen']} | {s['best']} |")
    return "\n".join(lines)


def main(path):
    with open(path) as f:
        res = json.load(f)
    stem = os.path.splitext(path)[0]
    selectivity_sweep(res, stem + "_c5.png")
    with open(stem + "_table.md", "w") as f:
        f.write(markdown_table(res) + "\n")
    print("wrote", stem + "_c5.png", stem + "_table.md")


if __name__ == "__main__":
    main(sys.argv[1])
