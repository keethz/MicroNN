"""Tables and plots for results/research_*.json.

  .venv/bin/python -m bench.analyze_research results/research_sift_1000000.json
"""
import json
import os
import sys

import numpy as np

from bench.plot_search import INK2, SURFACE, plt

ESTIMATORS = ["independence", "column-group", "sample", "stratified", "oracle-global", "oracle-local"]
# Reference palette slots 1-4, then neutral inks for the two oracles (upper bounds, not series).
COLORS = {"independence": "#2a78d6", "column-group": "#eb6834", "sample": "#1baf7a",
          "stratified": "#eda100", "oracle-global": "#6b6a64", "oracle-local": "#1f1f1e"}
CATEGORIES = ["independent", "correlated", "anti-correlated", "vector-correlated"]
RECALL_TARGET = 0.9


def q_error(est, true, n):
    est, true = max(est, 1 / n), max(true, 1 / n)
    return max(est / true, true / est)


def best_plan(f, i):
    """Per query: fastest plan reaching the recall target, else the higher-recall plan."""
    pq = f["per_query"]
    ok = [p for p in ("pre", "post") if pq[p]["recall"][i] >= RECALL_TARGET]
    if ok:
        return min(ok, key=lambda p: pq[p]["ms"][i])
    return max(("pre", "post"), key=lambda p: pq[p]["recall"][i])


def score(res, est, guard, categories=None):
    lat, rec, hit, agree, regret = [], [], [], [], []
    for f in res["filters"]:
        if categories and f["category"] not in categories:
            continue
        for i, plan in enumerate(f["est"][est]["decisions"][str(guard)]):
            pq = f["per_query"]
            b = best_plan(f, i)
            lat.append(pq[plan]["ms"][i])
            rec.append(pq[plan]["recall"][i])
            hit.append(pq[plan]["recall"][i] >= RECALL_TARGET)
            agree.append(plan == b)
            regret.append(pq[plan]["ms"][i] - pq[b]["ms"][i])
    return dict(mean_ms=float(np.mean(lat)), mean_recall=float(np.mean(rec)),
                frac_target=float(np.mean(hit)), plan_accuracy=float(np.mean(agree)),
                mean_regret_ms=float(np.mean(regret)))


def report(res) -> str:
    n = res["n"]
    lines = [f"# Research extension A — {res['dataset']} N={n:,}, nprobe={res['nprobe']}, "
             f"{res['n_query']} queries × {len(res['filters'])} filters, sample budget {res['sample']:,} rows", ""]
    lines += ["## Estimator cost", "", "| estimator | build s | memory KiB |", "|---|---|---|"]
    for e in ESTIMATORS[:4]:
        lines.append(f"| {e} | {res['build_s'][e]:.1f} | {res['memory_bytes'][e] // 1024:,} |")
    lines += ["", "## Selectivity q-error (median / max) by filter category", "",
              "| estimator | " + " | ".join(CATEGORIES) + " |", "|---|" + "---|" * len(CATEGORIES)]
    for e in ESTIMATORS[:4]:
        cells = []
        for c in CATEGORIES:
            qs = [q_error(f["est"][e]["sel"], f["true_sel"], n) for f in res["filters"] if f["category"] == c]
            cells.append(f"{np.median(qs):.1f} / {np.max(qs):.1f}")
        lines.append(f"| {e} | " + " | ".join(cells) + " |")
    for guard in (1, best_guard(res)):
        lines += ["", f"## Plan choice, recall guard = {guard}·k "
                      f"(target recall@{res['k']} ≥ {RECALL_TARGET})", "",
                  "| estimator | category | plan accuracy | queries ≥ target | mean recall | mean ms | regret ms |",
                  "|---|---|---|---|---|---|---|"]
        for e in ESTIMATORS:
            for c in [None] + CATEGORIES:
                s = score(res, e, guard, [c] if c else None)
                lines.append(f"| {e} | {c or '**all**'} | {s['plan_accuracy']:.0%} | {s['frac_target']:.0%} | "
                             f"{s['mean_recall']:.3f} | {s['mean_ms']:.1f} | {s['mean_regret_ms']:+.1f} |")
    return "\n".join(lines) + "\n"


def best_guard(res):
    """Guard where the stratified estimator first meets the recall target on ≥ 95% of queries."""
    for g in res["guards"]:
        if score(res, "stratified", g)["frac_target"] >= 0.95:
            return g
    return res["guards"][-1]


def plot_qerror(res, out):
    fig, axes = plt.subplots(1, len(CATEGORIES), figsize=(11, 3.6), sharey=True)
    rng = np.random.default_rng(0)
    for ax, c in zip(axes, CATEGORIES):
        fs = [f for f in res["filters"] if f["category"] == c]
        for j, e in enumerate(ESTIMATORS[:4]):
            qs = [q_error(f["est"][e]["sel"], f["true_sel"], res["n"]) for f in fs]
            ax.scatter(j + rng.uniform(-0.15, 0.15, len(qs)), qs, s=22, color=COLORS[e],
                       edgecolor=SURFACE, linewidth=1, zorder=3)
            ax.hlines(np.median(qs), j - 0.3, j + 0.3, color=COLORS[e], lw=2, zorder=4)
        ax.set_xticks(range(4), ["indep.", "col-group", "sample", "stratified"], rotation=30)
        ax.set_yscale("log")
        ax.set_title(c, loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("selectivity q-error (log; 1 = exact)")
    axes[0].set_ylim(bottom=0.8)
    fig.suptitle("Estimation error by filter type (dots = filters, bar = median)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_tradeoff(res, out):
    panels = [("all filters", None), ("vector-correlated", ["vector-correlated"]),
              ("correlated + anti-correlated", ["correlated", "anti-correlated"])]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (title, cats) in zip(axes, panels):
        for e in ESTIMATORS:
            pts = [score(res, e, g, cats) for g in res["guards"]]
            style = dict(color=COLORS[e], marker="o", markeredgecolor=SURFACE, markeredgewidth=1.2,
                         label=e, markersize=5)
            if e.startswith("oracle"):
                style.update(ls=(0, (3, 2)), lw=1.5)
            ax.plot([p["mean_ms"] for p in pts], [p["frac_target"] for p in pts], **style)
        ax.set_xscale("log")
        ax.set_xlabel("mean query latency (ms, log)")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylim(0, 1.03)
    axes[0].set_ylabel(f"queries with recall ≥ {RECALL_TARGET}")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Optimizer quality per estimator; points along a line = recall guard "
                 f"{res['guards'][0]}k → {res['guards'][-1]}k (up and left is better)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main(path):
    with open(path) as f:
        res = json.load(f)
    stem = os.path.splitext(path)[0]
    with open(stem + "_report.md", "w") as f:
        f.write(report(res))
    plot_qerror(res, stem + "_qerror.png")
    plot_tradeoff(res, stem + "_tradeoff.png")
    print("wrote", stem + "_report.md", stem + "_qerror.png", stem + "_tradeoff.png")


if __name__ == "__main__":
    main(sys.argv[1])
