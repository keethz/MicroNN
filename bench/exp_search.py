"""Experiments for claims C1-C4 (unfiltered search, memory, build).

  C1  latency vs recall@100 sweeping nprobe
  C2  query-time memory: MicroNN vs in-memory IVF, as N grows
  C3  MicroNN (disk) vs in-memory IVF latency at equal recall
  C4  build memory vs N; partition-size balance with/without penalty

Usage:
  .venv/bin/python -m bench.exp_search --dataset synthetic --n 100000
  .venv/bin/python -m bench.exp_search --dataset sift --n 1000000
Memory is measured in a fresh subprocess per configuration (peak RSS growth over the
post-import baseline, plus the tracemalloc peak, which covers NumPy buffers).
"""
import argparse
import json
import multiprocessing as mp
import os
import resource
import sys
import time
import tracemalloc
import warnings

import numpy as np

from bench.datasets import DATA_DIR, cached_ground_truth, load_glove, load_sift, recall_at_k, synthetic
from micronn import InMemoryIVF, MicroNN

# NumPy 2.x + macOS Accelerate: spurious FP flags from matmul (results verified in tests).
warnings.filterwarnings("ignore", ".*encountered in matmul", RuntimeWarning)

RESULTS = os.path.join(os.path.dirname(DATA_DIR), "results")
K = 100
NPROBES = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def peak_rss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024  # macOS: bytes, Linux: KiB


def load(dataset: str, n: int, n_query: int):
    if dataset == "synthetic":
        base, q, _ = synthetic(n, dim=128, n_clusters=max(10, n // 100), n_query=n_query,
                                 seed=0, spread=0.7)
    elif dataset == "sift":
        base, q = load_sift(n, n_query)
    elif dataset == "glove":
        base, q = load_glove(n, n_query)
    else:
        raise ValueError(dataset)
    return base, q


def db_path(dataset: str, n: int, balance: float) -> str:
    os.makedirs(os.path.join(DATA_DIR, "db"), exist_ok=True)
    return os.path.join(DATA_DIR, "db", f"{dataset}_{n}_b{balance}.sqlite")


# ---------- subprocess workers (fresh interpreter => clean memory numbers) ----------
def _build_worker(dataset, n, balance, out):
    base, _ = load(dataset, n, 1)
    path = db_path(dataset, n, balance)
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(path + suffix):
            os.remove(path + suffix)
    db = MicroNN.create(path, dim=base.shape[1])
    db.add(np.arange(len(base)), base)
    del base
    _warm_blas(db.dim)
    rss0 = peak_rss_bytes()
    tracemalloc.start()
    st = db.build(balance=balance)
    tm_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    out.put(dict(n=n, balance=balance, seconds=st.seconds, n_partitions=st.n_partitions,
                 size_std=float(st.sizes.std()), size_max=int(st.sizes.max()),
                 size_mean=float(st.sizes.mean()), sizes=st.sizes.tolist(),
                 tracemalloc_peak=tm_peak, rss_growth=peak_rss_bytes() - rss0))


def _warm_blas(dim):
    """First BLAS call allocates thread buffers (~tens of MiB); keep that out of the measurement."""
    x = np.random.rand(1000, dim).astype(np.float32)
    _ = x @ x.T


def _memory_worker(system, path, queries, nprobe, out):
    _warm_blas(queries.shape[1])
    rss0 = peak_rss_bytes()
    tracemalloc.start()
    db = MicroNN.open(path)
    index = InMemoryIVF(db) if system == "inmemory" else db
    for q in queries:
        index.search(q, K, nprobe)
    tm_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    out.put(dict(system=system, tracemalloc_peak=tm_peak, rss_growth=peak_rss_bytes() - rss0,
                 index_bytes=index.nbytes if system == "inmemory" else None))


def run_in_subprocess(target, *args):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=target, args=(*args, q))
    p.start()
    res = q.get()
    p.join()
    return res


# ---------- in-process timing ----------
def time_queries(index, queries, nprobe, warmup=10):
    for x in queries[:warmup]:
        index.search(x, K, nprobe)
    lat, found = [], []
    for x in queries:
        t = time.perf_counter()
        v, _ = index.search(x, K, nprobe)
        lat.append(time.perf_counter() - t)
        found.append(np.pad(v, (0, K - len(v)), constant_values=-1))
    lat = np.array(lat) * 1e3
    return np.stack(found), dict(median_ms=float(np.median(lat)), p95_ms=float(np.percentile(lat, 95)),
                                 mean_ms=float(lat.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic", choices=["synthetic", "sift", "glove"])
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--n-query", type=int, default=200)
    ap.add_argument("--scaling", type=str, default="",
                    help="comma-separated N values for C2/C4 scaling, e.g. 25000,50000,100000,200000")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    res = dict(dataset=args.dataset, n=args.n, k=K, builds=[], sweep=[], memory=[], scaling=[])

    # C4: build with and without balance penalty
    for balance in (0.0, 1.0):
        b = run_in_subprocess(_build_worker, args.dataset, args.n, balance)
        print(f"[build] balance={balance} {b['seconds']:.1f}s k={b['n_partitions']} "
              f"size mean={b['size_mean']:.0f} std={b['size_std']:.0f} max={b['size_max']} "
              f"tracemalloc={b['tracemalloc_peak']/2**20:.1f}MiB", flush=True)
        res["builds"].append(b)

    base, queries = load(args.dataset, args.n, args.n_query)
    gt = cached_ground_truth(f"{args.dataset}_{args.n}_{args.n_query}", base, queries, K)
    del base

    # C1 + C3: latency/recall sweep, both systems, both balance settings
    for balance in (0.0, 1.0):
        db = MicroNN.open(db_path(args.dataset, args.n, balance))
        mem = InMemoryIVF(db)
        for system, index in (("micronn", db), ("inmemory", mem)):
            for nprobe in NPROBES:
                if nprobe > db.n_partitions:
                    continue
                runs = [time_queries(index, queries, nprobe) for _ in range(args.runs)]
                rec = recall_at_k(runs[0][0], gt, K)
                lat = {key: float(np.median([r[1][key] for r in runs])) for key in runs[0][1]}
                row = dict(system=system, balance=balance, nprobe=nprobe, recall=rec, **lat)
                res["sweep"].append(row)
                print(f"[sweep] {system:8s} b={balance} nprobe={nprobe:3d} recall={rec:.3f} "
                      f"median={lat['median_ms']:.2f}ms p95={lat['p95_ms']:.2f}ms", flush=True)
        db.close()

    # C2: query-time memory at an nprobe that reaches ~90% recall (MicroNN, balance=0)
    mn = [r for r in res["sweep"] if r["system"] == "micronn" and r["balance"] == 0.0]
    nprobe90 = next((r["nprobe"] for r in mn if r["recall"] >= 0.9), mn[-1]["nprobe"])
    res["nprobe90"] = nprobe90
    sizes = [int(s) for s in args.scaling.split(",") if s] or [args.n]
    for n in sizes:
        if n != args.n:
            b = run_in_subprocess(_build_worker, args.dataset, n, 0.0)
            res["scaling"].append(dict(kind="build", **{k: v for k, v in b.items() if k != "sizes"}))
        for system in ("micronn", "inmemory"):
            m = run_in_subprocess(_memory_worker, system, db_path(args.dataset, n, 0.0),
                                  queries[:50], nprobe90)
            m["n"] = n
            res["memory"].append(m)
            print(f"[memory] n={n} {system:8s} tracemalloc={m['tracemalloc_peak']/2**20:.1f}MiB "
                  f"rss_growth={m['rss_growth']/2**20:.1f}MiB", flush=True)

    out = os.path.join(RESULTS, f"search_{args.dataset}_{args.n}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
