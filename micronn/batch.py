"""Batched query execution (PLAN.md section 5, claim C7).

Queries that probe the same partition share one scan of it and one matrix multiply:
    D = |x|^2 - 2 X Q^T + |q|^2      for all queries routed to that partition.
"""
from collections import defaultdict
from typing import List, Tuple

import numpy as np

from .clustering import sqdist
from .index import MicroNN, topk
from .storage import DTYPE, blobs_to_matrix


def search_batch(db: MicroNN, queries: np.ndarray, k: int = 100,
                 nprobe: int = 8) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Same results as [db.search(q, k, nprobe) for q in queries], one scan per partition."""
    queries = np.asarray(queries, dtype=DTYPE)
    nq = len(queries)
    best_v = [np.empty(0, dtype=np.int64) for _ in range(nq)]
    best_d = [np.empty(0, dtype=DTYPE) for _ in range(nq)]
    q_sq = np.einsum("ij,ij->i", queries, queries)

    def merge(rows, qidx):
        if not rows:
            return
        v = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
        x = blobs_to_matrix([r[1] for r in rows], db.dim)
        d = (np.einsum("ij,ij->i", x, x)[:, None] - 2 * x @ queries[qidx].T + q_sq[qidx][None, :])
        for j, qi in enumerate(qidx):
            vv = np.concatenate([best_v[qi], v])
            dd = np.concatenate([best_d[qi], d[:, j]])
            top = topk(dd, k)
            best_v[qi], best_d[qi] = vv[top], dd[top]

    with db.snapshot():
        conn = db.store.conn
        if db.n_partitions:
            d = sqdist(queries, db._centroids, db._c_sq)
            groups = defaultdict(list)  # partition id -> queries that probe it
            for qi in range(nq):
                for pid in db._centroid_ids[topk(d[qi], min(nprobe, db.n_partitions))]:
                    groups[int(pid)].append(qi)
            for pid in sorted(groups):  # clustered-key order: sequential pages
                rows = conn.execute("SELECT vid, vec FROM vectors WHERE partition_id = ?", (pid,)).fetchall()
                merge(rows, np.array(groups[pid]))
        merge(conn.execute("SELECT vid, vec FROM delta").fetchall(), np.arange(nq))
    return list(zip(best_v, best_d))
