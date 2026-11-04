"""Research extension A: does correlation-aware estimation fix the optimizer's plan choice?

Every (filter, query) pair is executed once with each plan (pre, post), recording latency and
per-query recall. Each estimator is then scored by the plan it would choose for that pair,
so all estimators are compared on identical measurements.

Estimators (equal sample budget for the two samples):
  independence   per-column stats + independence (the paper's design)
  column-group   + joint frequencies for low-cardinality column pairs
  sample         uniform row sample
  stratified     same budget, spread evenly over IVF partitions (ours); also predicts
                 matches inside the partitions a query will probe
  oracle-global  true selectivity (upper bound for any global estimator)
  oracle-local   true matches in the probed partitions (upper bound for local estimators)

  .venv/bin/python -m bench.exp_research --dataset sift --n 1000000
"""
import argparse
import json
import os
import sqlite3
import time
import warnings

import numpy as np

from bench.datasets import attr_correlated, ground_truth, recall_at_k
from bench.exp_filter import make_attributes
from bench.exp_search import RESULTS, db_path, load
from micronn import MicroNN
from micronn.estimators import ColumnGroupEstimator, SampleEstimator, StratifiedSampleEstimator
from micronn.filters import eq, lt
from micronn.optimizer import CostModel, Estimator, Optimizer

warnings.filterwarnings("ignore", ".*encountered in matmul", RuntimeWarning)
K = 100
U_MAX = 1_000_000
PAIR_CARDS = [5, 10, 20, 50]
GUARDS = [1, 2, 4, 8, 16]


def extra_attributes(base, seed=10):
    """Correlated column pairs p{m}/q{m} (q = p 90% of the time) and two more
    vector-correlated columns with 10 and 1000 values."""
    rng = np.random.default_rng(seed)
    out = {}
    for m in PAIR_CARDS:
        p = rng.integers(0, m, len(base))
        out[f"p{m}"] = p
        out[f"q{m}"] = np.where(rng.random(len(base)) < 0.9, p, rng.integers(0, m, len(base)))
    out["corr10"] = attr_correlated(base, 10, noise=0.1, seed=seed + 1)
    out["corr1000"] = attr_correlated(base, 1000, noise=0.1, seed=seed + 2)
    return out


def workload(attrs):
    cat_counts = np.bincount(attrs["cat"])
    common, rare = int(cat_counts.argmax()), int(np.flatnonzero(cat_counts > 0)[-1])
    corr100 = np.argsort(np.bincount(attrs["corr"]))[::-1]
    corr1000 = np.argsort(np.bincount(attrs["corr1000"]))[::-1]
    W = []
    for m in PAIR_CARDS:
        p, q = f"p{m}", f"q{m}"
        W += [("correlated", f"{p}=0 & {q}=0", [eq(p, 0), eq(q, 0)]),
              ("correlated", f"{p}=1 & {q}=1", [eq(p, 1), eq(q, 1)]),
              ("correlated", f"{p}=0 & {q}=0 & u<5%", [eq(p, 0), eq(q, 0), lt("u", U_MAX // 20)]),
              ("anti-correlated", f"{p}=0 & {q}=1", [eq(p, 0), eq(q, 1)]),
              ("anti-correlated", f"{p}=1 & {q}=2 & u<50%", [eq(p, 1), eq(q, 2), lt("u", U_MAX // 2)])]
    W += [("correlated", "a=7 & b=7", [eq("a", 7), eq("b", 7)]),
          ("anti-correlated", "a=7 & b=8", [eq("a", 7), eq("b", 8)]),
          ("independent", "u<2%", [lt("u", U_MAX // 50)]),
          ("independent", "u<5% & u2<20", [lt("u", U_MAX // 20), lt("u2", 20)]),
          ("independent", "u<20% & u2<50", [lt("u", U_MAX // 5), lt("u2", 50)]),
          ("independent", "u2=7 & cat=common", [eq("u2", 7), eq("cat", common)]),
          ("independent", "cat=rare & u2<50", [eq("cat", rare), lt("u2", 50)]),
          ("independent", "cat=common", [eq("cat", common)]),
          ("vector-correlated", "corr10=0", [eq("corr10", 0)]),
          ("vector-correlated", "corr10=5", [eq("corr10", 5)]),
          ("vector-correlated", "corr10=0 & u<10%", [eq("corr10", 0), lt("u", U_MAX // 10)]),
          ("vector-correlated", "corr=top1", [eq("corr", int(corr100[0]))]),
          ("vector-correlated", "corr=top2", [eq("corr", int(corr100[1]))]),
          ("vector-correlated", "corr1000=top1", [eq("corr1000", int(corr1000[0]))]),
          ("vector-correlated", "corr1000=top2", [eq("corr1000", int(corr1000[1]))])]
    return W


def mask_of(f, attrs):
    m = np.ones(len(attrs["u"]), bool)
    for p in f:
        m &= (attrs[p.col] == p.value) if p.op == "=" else (attrs[p.col] < p.value)
    return m


class OracleGlobal(Estimator):
    name = "oracle-global"

    def __init__(self, n_total, masks):
        super().__init__(n_total, {})
        self.masks = masks

    def filter_selectivity(self, f):
        return float(self.masks[_key(f)].mean())


class OracleLocal(OracleGlobal):
    name = "oracle-local"

    def __init__(self, n_total, masks, part_of):
        super().__init__(n_total, masks)
        self.part_of = part_of

    def expected_matches(self, f, probed):
        return float(self.masks[_key(f)][np.isin(self.part_of, probed)].sum())


def _key(f):
    return tuple((p.col, p.op, p.value) for p in f)


def prepare(args, base):
    attrs = make_attributes(base)
    extra = extra_attributes(base)
    src = db_path(args.dataset, args.n, 0.0).replace("_b0.0.sqlite", "_filtered.sqlite")
    dst = src.replace("_filtered", "_research")
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(dst + suffix):
            os.remove(dst + suffix)
    with sqlite3.connect(src) as s, sqlite3.connect(dst) as d:
        s.backup(d)
    db = MicroNN.open(dst)
    with db.store.transaction() as conn:
        for c in extra:
            db.store.add_attribute_column(c, "INTEGER")
        conn.executemany(f"UPDATE attributes SET {', '.join(f'{c} = ?' for c in extra)} WHERE vid = ?",
                         ((*(int(extra[c][i]) for c in extra), i) for i in range(len(base))))
    attrs.update(extra)
    return db, attrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift")
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--n-query", type=int, default=30)
    ap.add_argument("--nprobe", type=int, default=32)
    ap.add_argument("--sample", type=int, default=32_000, help="rows in each sample estimator")
    args = ap.parse_args()

    base, queries = load(args.dataset, args.n, args.n_query)
    db, attrs = prepare(args, base)
    conn, cols = db.store.conn, db.store.attribute_columns()
    cost = CostModel(**json.loads(db.store.get_meta("cost_model")))
    part_of = np.empty(args.n, dtype=np.int64)
    for vid, pid in conn.execute("SELECT vid, partition_id FROM vid_map"):
        part_of[vid] = pid

    t = time.perf_counter()
    ests = {"independence": Estimator.analyze(conn, cols)}
    build_s = {"independence": time.perf_counter() - t}
    for name, make in [
        ("column-group", lambda: ColumnGroupEstimator.analyze(conn, cols)),
        ("sample", lambda: SampleEstimator.analyze(conn, cols, size=args.sample)),
        ("stratified", lambda: StratifiedSampleEstimator.analyze(
            conn, cols, per_partition=max(1, args.sample // db.n_partitions))),
    ]:
        t = time.perf_counter()
        ests[name] = make()
        build_s[name] = time.perf_counter() - t
    # Approximate in-memory size; every estimator also carries the per-column stats.
    base_bytes = sum(16 * (len(s.freq or {}) + len(s.bounds)) for s in ests["independence"].columns.values())
    mem = {"independence": base_bytes,
           "column-group": base_bytes + ests["column-group"].memory_entries * 3 * 8,
           "sample": base_bytes + sum(a.nbytes for a in ests["sample"].sample.values()),
           "stratified": base_bytes + sum(a.nbytes for a in ests["stratified"].sample.values())
           + ests["stratified"].part.nbytes}
    print("[estimators] build s:", {k: round(v, 1) for k, v in build_s.items()},
          "memory KiB:", {k: v // 1024 for k, v in mem.items()}, flush=True)

    W = workload(attrs)
    masks = {_key(f): mask_of(f, attrs) for _, _, f in W}
    ests["oracle-global"] = OracleGlobal(args.n, masks)
    ests["oracle-local"] = OracleLocal(args.n, masks, part_of)

    for x in queries[:5]:
        db.search(x, K, nprobe=args.nprobe)
    probed = [db.probe(x, args.nprobe) for x in queries]

    filters = []
    for cat, name, f in W:
        m = masks[_key(f)]
        gt = ground_truth(base, queries, K, mask=m)
        per_q = {}
        for plan in ("pre", "post"):
            lat, rec = [], []
            for i, x in enumerate(queries):
                t = time.perf_counter()
                v, _ = db.search(x, K, nprobe=args.nprobe, filter=f, plan=plan)
                lat.append((time.perf_counter() - t) * 1e3)
                rec.append(recall_at_k(np.pad(v, (0, K - len(v)), constant_values=-1)[None], gt[i:i + 1], K))
            per_q[plan] = dict(ms=lat, recall=rec)
        est = {}
        for en, e in ests.items():
            t = time.perf_counter()
            s = e.filter_selectivity(f)
            est_us = (time.perf_counter() - t) * 1e6
            decisions = {}
            for g in GUARDS:
                opt = Optimizer(e, cost, recall_guard=g)
                decisions[g] = [opt.choose(f, K, args.nprobe, db.avg_partition_size, probed=probed[i]).name
                                for i in range(len(queries))]
            est[en] = dict(sel=s, est_us=est_us, decisions=decisions)
        filters.append(dict(category=cat, name=name, true_sel=float(m.mean()), per_query=per_q, est=est))
        pre_ms, post_ms = np.median(per_q["pre"]["ms"]), np.median(per_q["post"]["ms"])
        print(f"[{cat:17s} {name:26s}] sel={m.mean():.5f} indep={est['independence']['sel']:.5f} "
              f"strat={est['stratified']['sel']:.5f} | pre {pre_ms:6.1f}ms post {post_ms:5.1f}ms "
              f"recall {np.mean(per_q['post']['recall']):.2f}", flush=True)
    db.close()

    res = dict(dataset=args.dataset, n=args.n, nprobe=args.nprobe, k=K, n_query=args.n_query,
               sample=args.sample, guards=GUARDS, memory_bytes=mem, build_s=build_s, filters=filters)
    out = os.path.join(RESULTS, f"research_{args.dataset}_{args.n}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh)
    print("wrote", out)


if __name__ == "__main__":
    main()
