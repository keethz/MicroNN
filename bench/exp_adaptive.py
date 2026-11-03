"""Research extension B: adaptive filtered search vs the optimizer's up-front plan choice.

Extension A (exp_research.py) found that better selectivity estimates help, but even exact
estimates pick the best plan for only ~83% of queries: the problem is predicting post-filter
recall before running the query. plan="adaptive" avoids that prediction. It scans partitions
nearest-first with the filter applied, keeps going until it has found k matches (plus a
margin), and falls back to pre-filter once that costs more than pre-filter would.

Strategies compared on the same filters and queries as extension A:
  pre, post                   fixed plans
  auto (independence, g)      the paper's optimizer with the recall guard g
  auto (stratified, g)        extension A's per-partition estimator
  adaptive (expand e)         this file

Needs the *_filtered.sqlite database built by bench.exp_filter.
  .venv/bin/python -m bench.exp_adaptive --dataset sift --n 1000000
"""
import argparse
import json
import os
import time
import warnings

import numpy as np

from bench.datasets import ground_truth, recall_at_k
from bench.exp_research import _key, mask_of, prepare, workload
from bench.exp_search import RESULTS, load
from micronn.estimators import StratifiedSampleEstimator
from micronn.index import SearchStats
from micronn.optimizer import CostModel, Estimator, Optimizer

warnings.filterwarnings("ignore", ".*encountered in matmul", RuntimeWarning)
K = 100
GUARDS = [1, 4, 16]
EXPANDS = [0.5, 1.0, 2.0]


def run(db, x, f, nprobe, plan):
    st = SearchStats()
    t = time.perf_counter()
    v, _ = db.search(x, K, nprobe=nprobe, filter=f, plan=plan, stats=st)
    return v, (time.perf_counter() - t) * 1e3, st


def rec(v, gt_row):
    return recall_at_k(np.pad(v, (0, K - len(v)), constant_values=-1)[None], gt_row[None], K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift")
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--n-query", type=int, default=30)
    ap.add_argument("--nprobe", type=int, default=32)
    ap.add_argument("--sample", type=int, default=32_000)
    args = ap.parse_args()

    base, queries = load(args.dataset, args.n, args.n_query)
    db, attrs = prepare(args, base)
    conn, cols = db.store.conn, db.store.attribute_columns()
    cost = CostModel(**json.loads(db.store.get_meta("cost_model")))
    ests = {"independence": Estimator.analyze(conn, cols),
            "stratified": StratifiedSampleEstimator.analyze(
                conn, cols, per_partition=max(1, args.sample // db.n_partitions))}
    # adaptive sizes its fallback budget with the paper's (independence) estimator
    db.optimizer = Optimizer(ests["independence"], cost)

    for x in queries[:5]:
        db.search(x, K, nprobe=args.nprobe)
    probed = [db.probe(x, args.nprobe) for x in queries]

    filters = []
    for cat, name, f in workload(attrs):
        m = mask_of(f, attrs)
        gt = ground_truth(base, queries, K, mask=m)
        out = {}
        for plan in ("pre", "post"):
            r = [run(db, x, f, args.nprobe, plan) for x in queries]
            out[plan] = dict(ms=[t for _, t, _ in r], recall=[rec(v, gt[i]) for i, (v, _, _) in enumerate(r)])
        # the optimizer only picks between the two measured plans, so score it on those
        for en, e in ests.items():
            for g in GUARDS:
                opt = Optimizer(e, cost, recall_guard=g)
                pick = [opt.choose(f, K, args.nprobe, db.avg_partition_size, probed=probed[i]).name
                        for i in range(len(queries))]
                out[f"auto {en} g={g}"] = dict(
                    ms=[out[p]["ms"][i] for i, p in enumerate(pick)],
                    recall=[out[p]["recall"][i] for i, p in enumerate(pick)])
        for e in EXPANDS:
            db.adaptive_expand = e
            r = [run(db, x, f, args.nprobe, "adaptive") for x in queries]
            out[f"adaptive e={e:g}"] = dict(
                ms=[t for _, t, _ in r], recall=[rec(v, gt[i]) for i, (v, _, _) in enumerate(r)],
                partitions=[st.partitions_probed for _, _, st in r],
                fallback=[st.adaptive_fallback for _, _, st in r])
        filters.append(dict(category=cat, name=name, true_sel=float(m.mean()), strategies=out))
        a = out["adaptive e=1"]
        print(f"[{cat:17s} {name:26s}] sel={m.mean():.5f} | pre {np.median(out['pre']['ms']):6.1f}ms "
              f"post {np.median(out['post']['ms']):5.1f}ms r={np.mean(out['post']['recall']):.2f} | "
              f"adaptive {np.median(a['ms']):6.1f}ms r={np.mean(a['recall']):.2f} "
              f"parts={np.median(a['partitions']):.0f} fallback={np.mean(a['fallback']):.0%}", flush=True)
    db.close()

    res = dict(dataset=args.dataset, n=args.n, nprobe=args.nprobe, k=K, n_query=args.n_query,
               guards=GUARDS, expands=EXPANDS, filters=filters)
    out_path = os.path.join(RESULTS, f"adaptive_{args.dataset}_{args.n}.json")
    with open(out_path, "w") as fh:
        json.dump(res, fh)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
