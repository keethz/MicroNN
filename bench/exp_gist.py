"""Focused GIST1M experiment: build once and sweep recall versus nprobe.

This is intentionally smaller than exp_search: it does not run the paper's C1-C4
memory, scaling, balance, or in-memory-baseline experiments.

Example (GIST1M prefix):
  python -m bench.exp_gist --n 100000 --n-query 100
"""
import argparse
import json
import os
import time

import numpy as np

from bench.datasets import cached_ground_truth, recall_at_k
from bench.exp_search import K, NPROBES, RESULTS, db_path, load
from micronn import MicroNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000, choices=[100_000, 200_000])
    ap.add_argument("--n-query", type=int, default=100)
    ap.add_argument("--nprobe", default=", ".join(map(str, NPROBES)),
                    help="comma-separated nprobe values")
    args = ap.parse_args()
    dataset = "gist"
    base, queries = load(dataset, args.n, args.n_query)
    gt = cached_ground_truth(f"{dataset}_{args.n}_{args.n_query}", base, queries, K)

    path = db_path(dataset, args.n, 0.0)
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(path + suffix):
            os.remove(path + suffix)
    db = MicroNN.create(path, dim=base.shape[1])
    db.add(np.arange(len(base)), base)
    t = time.perf_counter()
    stats = db.build(balance=0.0)
    build_s = time.perf_counter() - t
    print(f"[build] n={args.n} dim={base.shape[1]} partitions={stats.n_partitions} seconds={build_s:.1f}", flush=True)

    rows = []
    for nprobe in (int(x.strip()) for x in args.nprobe.split(",") if x.strip()):
        if nprobe > db.n_partitions:
            continue
        for x in queries[:5]:
            db.search(x, K, nprobe=nprobe)
        lat, found = [], []
        for x in queries:
            t = time.perf_counter()
            v, _ = db.search(x, K, nprobe=nprobe)
            lat.append((time.perf_counter() - t) * 1e3)
            found.append(np.pad(v, (0, K - len(v)), constant_values=-1))
        row = dict(nprobe=nprobe, recall=recall_at_k(np.stack(found), gt, K),
                   median_ms=float(np.median(lat)), p95_ms=float(np.percentile(lat, 95)))
        rows.append(row)
        print(f"[sweep] nprobe={nprobe:3d} recall={row['recall']:.4f} median={row['median_ms']:.2f}ms p95={row['p95_ms']:.2f}ms", flush=True)
    db.close()
    out = os.path.join(RESULTS, f"gist_sweep_{args.n}.json")
    with open(out, "w") as fh:
        json.dump(dict(dataset=dataset, n=args.n, n_query=args.n_query, k=K,
                       dim=base.shape[1], build_s=build_s, sweep=rows), fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
