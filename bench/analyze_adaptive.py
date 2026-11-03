"""Summarize bench.exp_adaptive: one row per strategy, plus a latency-vs-quality plot.

  .venv/bin/python -m bench.analyze_adaptive results/adaptive_sift_1000000.json
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TARGET = 0.9
# validated categorical slots (blue, orange, aqua, yellow); identity is also direct-labelled
COLORS = {"fixed": "#2a78d6", "auto independence": "#eb6834", "auto stratified": "#1baf7a",
          "adaptive": "#eda100"}
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dd"


def family(name):
    if name in ("pre", "post"):
        return "fixed"
    if name.startswith("auto independence"):
        return "auto independence"
    if name.startswith("auto stratified"):
        return "auto stratified"
    return "adaptive"


def summarize(filters, cats=None):
    rows = {}
    for fl in filters:
        if cats and fl["category"] not in cats:
            continue
        for s, v in fl["strategies"].items():
            r = rows.setdefault(s, dict(ms=[], recall=[]))
            r["ms"] += v["ms"]
            r["recall"] += v["recall"]
    out = []
    for s, r in rows.items():
        ms, rec = np.array(r["ms"]), np.array(r["recall"])
        out.append(dict(strategy=s, mean_ms=ms.mean(), p95_ms=np.percentile(ms, 95),
                        recall=rec.mean(), on_target=(rec >= TARGET).mean()))
    return out


def table(rows, title):
    lines = [f"### {title}", "", "| strategy | mean latency (ms) | p95 (ms) | mean recall@100 | queries with recall ≥ 0.9 |",
             "|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["mean_ms"]):
        lines.append(f"| {r['strategy']} | {r['mean_ms']:.1f} | {r['p95_ms']:.1f} | {r['recall']:.3f} | "
                     f"{r['on_target']:.0%} |")
    return "\n".join(lines)


def plot(rows, path, title):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for r in rows:
        f = family(r["strategy"])
        ax.scatter(r["mean_ms"], 100 * r["on_target"], s=64, color=COLORS[f], edgecolor="white",
                   linewidth=2, zorder=3)
    ax.set_xlim(0, max(r["mean_ms"] for r in rows) * 1.45)
    ax.set_ylim(0, 108)
    fig.canvas.draw()
    # Labels to the right of each point, pushed down when they would overlap a label above.
    placed = []                                   # (x_px, y_px) of labels already placed
    for r in sorted(rows, key=lambda r: -r["on_target"]):
        x, y = ax.transData.transform((r["mean_ms"], 100 * r["on_target"]))
        ly = y + 3
        for px, py in placed:
            if abs(px - x) < 160 and ly > py - 12:
                ly = min(ly, py - 12)
        placed.append((x, ly))
        dy_pt = (ly - y) * 72 / fig.dpi
        ax.annotate(r["strategy"], (r["mean_ms"], 100 * r["on_target"]), xytext=(7, dy_pt),
                    textcoords="offset points", fontsize=8, color=INK, va="bottom",
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8) if abs(ly - y - 3) > 4 else None)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, markersize=7, label=f)
               for f, c in COLORS.items()]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
    ax.set_xlabel("mean latency per query (ms) — lower is better", color=MUTED)
    ax.set_ylabel("queries with recall@100 ≥ 0.9 (%) — higher is better", color=MUTED)
    ax.tick_params(colors=MUTED)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    path = sys.argv[1]
    res = json.load(open(path))
    filters = res["filters"]
    cats = sorted({f["category"] for f in filters})
    parts = [f"# Adaptive filtered search — {res['dataset']}, N={res['n']}, nprobe={res['nprobe']}, "
             f"{res['n_query']} queries × {len(filters)} filters", ""]
    all_rows = summarize(filters)
    parts.append(table(all_rows, "All filters"))
    for c in cats:
        parts += ["", table(summarize(filters, {c}), f"{c} filters")]
    report = path.replace(".json", "_report.md")
    open(report, "w").write("\n".join(parts) + "\n")
    png = path.replace(".json", "_tradeoff.png")
    plot(all_rows, png, f"Latency vs quality, all filters ({res['dataset']}, N={res['n']:,})")
    print("\n".join(parts))
    print("wrote", report, png)


if __name__ == "__main__":
    main()
