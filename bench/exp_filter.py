"""Experiments for claims C5-C6 (hybrid queries) on an existing unfiltered index.

  C5  pre-filter: 100% recall, wins at low selectivity; post-filter wins at high selectivity
  C6  the optimizer picks the right plan (vs best plan in hindsight), plus estimation accuracy
      on filter types that do / do not satisfy the independence assumption

Usage (after bench.exp_search has built data/db/sift_1000000_b0.0.sqlite):
  .venv/bin/python -m bench.exp_filter --dataset sift --n 1000000
"""
import argparse
import json
import os
import sqlite3
import time
import warnings

import numpy as np

from bench.datasets import attr_correlated, attr_zipf, ground_truth, recall_at_k
from bench.exp_search import RESULTS, db_path, load
from micronn import MicroNN
from micronn.filters import eq, lt
from micronn.index import SearchStats
from micronn.optimizer import Optimizer

warnings.filterwarnings("ignore", ".*encountered in matmul", RuntimeWarning)
K = 100
SELECTIVITIES = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0]
U_MAX = 1_000_000


def make_attributes(base: np.ndarray, seed: int = 0) -> dict:
    n = len(base)
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 100, n)
    return dict(
        u=rng.integers(0, U_MAX, n),                       # uniform: range filters set selectivity exactly
        cat=attr_zipf(n, 100, a=1.2, seed=seed + 1),       # skewed categorical
        corr=attr_correlated(base, 100, noise=0.1, seed=seed + 2),  # tied to vector location
        a=a,                                               # a, b: correlated pair (b == a 90% of the time)
        b=np.where(rng.random(n) < 0.9, a, rng.integers(0, 100, n)),
        u2=rng.integers(0, 100, n),                        # independent of everything
    )


def prepare_db(dataset: str, n: int, base: np.ndarray, attrs: dict) -> str:
    """Copy the unfiltered index and attach attribute columns."""
    src = db_path(dataset, n, 0.0)
    dst = src.replace("_b0.0.sqlite", "_filtered.sqlite")
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(dst + suffix):
            os.remove(dst + suffix)
    with sqlite3.connect(src) as s, sqlite3.connect(dst) as d:
        s.backup(d)
    db = MicroNN.open(dst)
    with db.store.transaction():
        for col in attrs:
            db.store.add_attribute_column(col, "INTEGER")
        rows = list(zip(*(attrs[c].tolist() for c in attrs)))
        db.store.insert_attributes(np.arange(len(base)).tolist(), rows)
    t = time.perf_counter()
    db.analyze()
    print(f"[analyze] {time.perf_counter() - t:.1f}s cost model={db.store.get_meta('cost_model')}", flush=True)
    db.close()
    return dst


def mask_of(f, attrs):
    m = np.ones(len(next(iter(attrs.values()))), bool)
    for p in f:
        col = attrs[p.col]
        if p.op == "=":
            m &= col == p.value
        elif p.op == "<":
            m &= col < p.value
        else:
            raise ValueError(p.op)
    return m


def run_plan(db, queries, f, plan, nprobe, gt, optimizer=None):
    saved = db.optimizer
    if optimizer is not None:
        db.optimizer = optimizer
    lat, found, chosen = [], [], []
    for x in queries:
        st = SearchStats()
        t = time.perf_counter()
        v, _ = db.search(x, K, nprobe=nprobe, filter=f, plan=plan, stats=st)
        lat.append(time.perf_counter() - t)
        found.append(np.pad(v, (0, K - len(v)), constant_values=-1))
        if st.plan is not None:
            chosen.append(st.plan.name)
    db.optimizer = saved
    lat = np.array(lat) * 1e3
    return dict(plan=plan, recall=recall_at_k(np.stack(found), gt, K),
                median_ms=float(np.median(lat)), p95_ms=float(np.percentile(lat, 95)),
                chosen=max(set(chosen), key=chosen.count) if chosen else plan,
                avg_results=float(np.mean([(r >= 0).sum() for r in found])))


def best_in_hindsight(rows, min_recall=0.9):
    ok = [r for r in rows if r["plan"] in ("pre", "post") and r["recall"] >= min_recall]
    return min(ok, key=lambda r: r["median_ms"]) if ok else max(
        (r for r in rows if r["plan"] in ("pre", "post")), key=lambda r: r["recall"])


def evaluate(db, base, queries, f, name, attrs, nprobe, cost_only):
    m = mask_of(f, attrs)
    true_sel = float(m.mean())
    est_sel = db.optimizer.est.filter_selectivity(f)
    gt = ground_truth(base, queries, K, mask=m)
    rows = [run_plan(db, queries, f, p, nprobe, gt) for p in ("pre", "post", "auto")]
    rows.append(dict(run_plan(db, queries, f, "auto", nprobe, gt, cost_only), plan="auto_cost_only"))
    best = best_in_hindsight(rows)
    floor = 1 / len(m)  # q-error with selectivities floored at one row
    q_err = max(est_sel, floor) / max(true_sel, floor)
    q_err = max(q_err, 1 / q_err)
    out = dict(name=name, true_sel=true_sel, est_sel=est_sel, q_error=q_err, rows=rows, best=best["plan"])
    auto = rows[2]
    print(f"[{name:28s}] sel={true_sel:.5f} est={est_sel:.5f} q-err={q_err:6.1f} | " +
          " ".join(f"{r['plan']}:{r['recall']:.2f}/{r['median_ms']:.1f}ms" for r in rows) +
          f" | auto chose {auto['chosen']}, best={best['plan']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift")
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--n-query", type=int, default=50)
    ap.add_argument("--nprobe", type=int, default=32)  # ~96% unfiltered recall on SIFT1M
    args = ap.parse_args()

    base, queries = load(args.dataset, args.n, args.n_query)
    attrs = make_attributes(base)
    path = prepare_db(args.dataset, args.n, base, attrs)
    db = MicroNN.open(path)
    cost_only = Optimizer(db.optimizer.est, db.optimizer.cost, recall_guard=0)
    for x in queries[:5]:  # warm up
        db.search(x, K, nprobe=args.nprobe)

    res = dict(dataset=args.dataset, n=args.n, nprobe=args.nprobe, k=K,
               cost_model=json.loads(db.store.get_meta("cost_model")), sweep=[], filters=[])
    # C5 + C6: selectivity sweep on a uniform column
    for s in SELECTIVITIES:
        res["sweep"].append(evaluate(db, base, queries, [lt("u", int(s * U_MAX))], f"u < {s:g}", attrs,
                                     args.nprobe, cost_only))
    # C6: estimation accuracy across filter types
    cat_vals = np.bincount(attrs["cat"], minlength=100)
    common, rare = int(cat_vals.argmax()), int(np.flatnonzero(cat_vals > 0)[-1])
    corr_v = int(np.bincount(attrs["corr"]).argmax())
    for name, f in [
        ("cat = common (zipf)", [eq("cat", common)]),
        ("cat = rare (zipf)", [eq("cat", rare)]),
        ("u2 = 7 AND cat = common", [eq("u2", 7), eq("cat", common)]),   # independent conjunction
        ("a = 7 AND b = 7", [eq("a", 7), eq("b", 7)]),                   # correlated conjunction
        ("a = 7 AND b = 8", [eq("a", 7), eq("b", 8)]),                   # anti-correlated
        ("corr = v (vector-correlated)", [eq("corr", corr_v)]),
    ]:
        res["filters"].append(evaluate(db, base, queries, f, name, attrs, args.nprobe, cost_only))
    db.close()
    out = os.path.join(RESULTS, f"filter_{args.dataset}_{args.n}.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
