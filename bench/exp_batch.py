"""Experiment for claim C7: batching cuts average per-query latency (~33% at batch 1024).

  .venv/bin/python -m bench.exp_batch --dataset sift --n 1000000
"""
import argparse
import json
import os
import time
import warnings

import numpy as np

from bench.datasets import cached_ground_truth, recall_at_k
from bench.exp_search import RESULTS, db_path, load
from micronn import MicroNN
from micronn.batch import search_batch
from micronn.clustering import sqdist

warnings.filterwarnings("ignore", ".*encountered in matmul", RuntimeWarning)
K = 100
BATCHES = [1, 4, 16, 64, 256, 1024]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift")
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--n-query", type=int, default=2048)
    ap.add_argument("--nprobe", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    base, queries = load(args.dataset, args.n, args.n_query)
    gt = cached_ground_truth(f"{args.dataset}_{args.n}_{args.n_query}", base, queries, K)
    del base
    db = MicroNN.open(db_path(args.dataset, args.n, 0.0))
    for x in queries[:20]:
        db.search(x, K, nprobe=8)
    res = dict(dataset=args.dataset, n=args.n, n_query=args.n_query, rows=[])
    for nprobe in args.nprobe:
        # sequential baseline: one db.search per query
        times = []
        for _ in range(args.runs):
            t = time.perf_counter()
            found = [db.search(x, K, nprobe=nprobe)[0] for x in queries]
            times.append(time.perf_counter() - t)
        seq_ms = float(np.median(times)) / len(queries) * 1e3
        seq_rec = recall_at_k(np.stack(found), gt, K)
        res["rows"].append(dict(nprobe=nprobe, batch=0, per_query_ms=seq_ms, recall=seq_rec, scans=nprobe))
        print(f"[seq ] nprobe={nprobe} {seq_ms:.2f} ms/query recall={seq_rec:.3f}", flush=True)
        for b in BATCHES:
            times = []
            for _ in range(args.runs):
                t = time.perf_counter()
                out = []
                for s in range(0, len(queries), b):
                    out += search_batch(db, queries[s:s + b], K, nprobe)
                times.append(time.perf_counter() - t)
            per_q = float(np.median(times)) / len(queries) * 1e3
            rec = recall_at_k(np.stack([v for v, _ in out]), gt, K)
            # partition scans per query: distinct (batch, partition) pairs / queries
            scans = 0
            for s in range(0, len(queries), b):
                d = sqdist(queries[s:s + b], db._centroids)
                scans += len(np.unique(np.argsort(d, axis=1)[:, :nprobe]))
            row = dict(nprobe=nprobe, batch=b, per_query_ms=per_q, recall=rec, scans=scans / len(queries),
                       vs_sequential=1 - per_q / seq_ms)
            res["rows"].append(row)
            print(f"[batch] nprobe={nprobe} b={b:5d} {per_q:.2f} ms/query ({row['vs_sequential']:+.0%} vs seq) "
                  f"recall={rec:.3f} partition scans/query={row['scans']:.1f}", flush=True)
    db.close()
    out = os.path.join(RESULTS, f"batch_{args.dataset}_{args.n}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
