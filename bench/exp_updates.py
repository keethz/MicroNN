"""Experiment for claim C8: updates via the delta store keep queries fast and correct.

Start from an index over half the data, then insert the other half in batches under three
maintenance policies, measuring query latency/recall (against exact answers over the vectors
present at that moment) and insert throughput including maintenance time.

Workloads:
  random  inserts arrive in random order (same distribution as the index)
  skewed  inserts sorted along one direction (data drift: new data piles into few partitions)

  .venv/bin/python -m bench.exp_updates --dataset sift --n 1000000
"""
import argparse
import json
import os
import sqlite3
import time
import warnings

import numpy as np

from bench.datasets import DATA_DIR, ground_truth, recall_at_k
from bench.exp_search import RESULTS, load
from micronn import MicroNN
from micronn.index import MaintenancePolicy

warnings.filterwarnings("ignore", ".*encountered in matmul", RuntimeWarning)
K = 100
POLICIES = {
    "never": MaintenancePolicy(delta_max=0, rebuild_factor=0),
    "merge": MaintenancePolicy(delta_max=50_000, rebuild_factor=0),
    "merge+rebuild": MaintenancePolicy(delta_max=50_000, rebuild_factor=1.5),
}


def copy_db(src, dst):
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(dst + suffix):
            os.remove(dst + suffix)
    with sqlite3.connect(src) as s, sqlite3.connect(dst) as d:
        s.backup(d)


def measure(db, queries, gt, nprobe):
    lat, found = [], []
    for x in queries:
        t = time.perf_counter()
        v, _ = db.search(x, K, nprobe=nprobe)
        lat.append(time.perf_counter() - t)
        found.append(np.pad(v, (0, K - len(v)), constant_values=-1))
    lat = np.array(lat) * 1e3
    return float(np.median(lat)), float(np.percentile(lat, 95)), recall_at_k(np.stack(found), gt, K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sift")
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--n-query", type=int, default=30)
    ap.add_argument("--nprobe", type=int, default=16)
    ap.add_argument("--batch", type=int, default=20_000)
    ap.add_argument("--checkpoints", type=int, default=5)
    args = ap.parse_args()

    base, queries = load(args.dataset, args.n, args.n_query)
    n0 = args.n // 2
    rng = np.random.default_rng(0)
    perm = rng.permutation(args.n)
    initial, rest = perm[:n0], perm[n0:]
    proj = base[rest] @ rng.normal(size=base.shape[1])
    orders = {"random": rest, "skewed": rest[np.argsort(proj)]}

    os.makedirs(os.path.join(DATA_DIR, "db"), exist_ok=True)
    seed_path = os.path.join(DATA_DIR, "db", f"{args.dataset}_{args.n}_half.sqlite")
    if os.path.exists(seed_path):
        os.remove(seed_path)
    db = MicroNN.create(seed_path, dim=base.shape[1])
    db.add(initial.tolist(), base[initial])
    st = db.build()
    db.close()
    print(f"[seed] {n0} vectors, k={st.n_partitions}, build {st.seconds:.1f}s", flush=True)

    n_ins = len(rest)
    every = n_ins // args.checkpoints
    res = dict(dataset=args.dataset, n=args.n, nprobe=args.nprobe, batch=args.batch, runs=[])
    for workload, order in orders.items():
        # ground truth at each checkpoint, shared by all policies
        present = np.zeros(args.n, bool)
        present[initial] = True
        gts = {0: ground_truth(base, queries, K, mask=present.copy())}
        for c in range(1, args.checkpoints + 1):
            present[order[(c - 1) * every:c * every]] = True
            gts[c * every] = ground_truth(base, queries, K, mask=present.copy())
        for name, policy in POLICIES.items():
            path = seed_path.replace("_half", f"_upd_{workload}_{name.replace('+', '_')}")
            copy_db(seed_path, path)
            db = MicroNN.open(path)
            for x in queries[:5]:
                db.search(x, K, nprobe=args.nprobe)
            rows, ins_time, maint_time, actions = [], 0.0, 0.0, []
            med, p95, rec = measure(db, queries, gts[0], args.nprobe)
            rows.append(dict(inserted=0, delta=0, median_ms=med, p95_ms=p95, recall=rec,
                             insert_s=0.0, maint_s=0.0, partitions=db.n_partitions))
            for s in range(0, n_ins, args.batch):
                ids = order[s:s + args.batch]
                t = time.perf_counter()
                db.add(ids.tolist(), base[ids])
                ins_time += time.perf_counter() - t
                t = time.perf_counter()
                a = db.maintain(policy)
                maint_time += time.perf_counter() - t
                if a != "none":
                    actions.append((s + len(ids), a))
                done = s + len(ids)
                if done % every == 0 and done in gts:
                    med, p95, rec = measure(db, queries, gts[done], args.nprobe)
                    _, n_delta = db.store.count()
                    rows.append(dict(inserted=done, delta=n_delta, median_ms=med, p95_ms=p95, recall=rec,
                                     insert_s=ins_time, maint_s=maint_time, partitions=db.n_partitions))
                    print(f"[{workload:6s} {name:13s}] +{done:7d} delta={n_delta:6d} k={db.n_partitions:4d} "
                          f"median={med:7.1f}ms recall={rec:.3f} insert={ins_time:5.1f}s maint={maint_time:5.1f}s",
                          flush=True)
            db.close()
            res["runs"].append(dict(workload=workload, policy=name, rows=rows, actions=actions,
                                    throughput=n_ins / (ins_time + maint_time)))
            for suffix in ("", "-wal", "-shm"):
                if os.path.exists(path + suffix):
                    os.remove(path + suffix)
    out = os.path.join(RESULTS, f"updates_{args.dataset}_{args.n}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
