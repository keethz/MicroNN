"""MicroNN index: IVF partitions stored in SQLite (PLAN.md sections 2-3).

Lifecycle:
  db = MicroNN.create(path, dim, attributes)
  db.add(vids, vecs, attrs)     # lands in the delta store
  db.build(n_partitions)        # mini-batch k-means, writes clustered partitions
  db.search(q, k, nprobe)       # probes nearest partitions + always scans delta
  db.analyze()                  # attribute statistics for the filter optimizer
  db.search(q, k, nprobe, filter=[eq("city", "Seattle")])   # hybrid query, plan="auto"|"pre"|"post"
  db.add(...); db.delete(...); db.maintain(policy)          # updates (PLAN.md section 6)

Every search runs inside one read transaction, so it sees a single WAL snapshot even while
another connection inserts, merges or rebuilds. The `epoch` meta key changes whenever
partitions are rewritten; a reader that sees a new epoch reloads its cached centroids.
"""
import contextlib
import json
import math
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .clustering import balanced_assign, minibatch_kmeans, sqdist
from .filters import Filter, to_sql
from .optimizer import CostModel, Estimator, Optimizer, Plan, calibrate
from .storage import DELTA_PARTITION, DTYPE, Store, to_blob


@dataclass
class SearchStats:
    partitions_probed: int = 0
    vectors_scanned: int = 0
    delta_scanned: int = 0
    seconds: float = 0.0
    plan: Optional[Plan] = None


@dataclass
class MaintenancePolicy:
    """When to fold the delta store into the index.

    delta_max:       merge once the delta store holds this many vectors (0 = never merge)
    rebuild_factor:  full rebuild once the largest partition exceeds this multiple of the
                     largest partition at the last build (0 = never rebuild). Relative to the
                     built maximum, not the average, because plain k-means already leaves some
                     partitions several times the average size.
    """
    delta_max: int = 10_000
    rebuild_factor: float = 1.5


@dataclass
class BuildStats:
    n_vectors: int = 0
    n_partitions: int = 0
    seconds: float = 0.0
    sizes: np.ndarray = field(default_factory=lambda: np.empty(0))


def topk(d: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k smallest entries of a 1-D array, sorted ascending."""
    if len(d) <= k:
        return np.argsort(d)
    idx = np.argpartition(d, k - 1)[:k]
    return idx[np.argsort(d[idx])]


def _qdist(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Squared L2 from one query to each row of x, without materializing x**2."""
    return np.einsum("ij,ij->i", x, x) - 2 * (x @ q) + q @ q


def _merge_topk(v1, d1, v2, d2, k):
    v = np.concatenate([v1, v2])
    d = np.concatenate([d1, d2])
    best = topk(d, k)
    return v[best], d[best]


class MicroNN:
    def __init__(self, store: Store):
        self.store = store
        self._centroids: Optional[np.ndarray] = None
        self._centroid_ids: Optional[np.ndarray] = None
        self._c_sq: Optional[np.ndarray] = None
        self._n_indexed = 0
        self._epoch: Optional[str] = None
        self.optimizer: Optional[Optimizer] = None
        self.reload_centroids()
        self._load_optimizer()

    @classmethod
    def create(cls, path: str, dim: int, attributes: Optional[Dict[str, str]] = None,
               cache_kib: int = 2048) -> "MicroNN":
        store = Store(path, cache_kib=cache_kib)
        store.create(dim, attributes)
        return cls(store)

    @classmethod
    def open(cls, path: str, cache_kib: int = 2048) -> "MicroNN":
        return cls(Store(path, cache_kib=cache_kib))

    @property
    def dim(self) -> int:
        return self.store.dim

    @property
    def n_partitions(self) -> int:
        return 0 if self._centroids is None else len(self._centroids)

    def reload_centroids(self) -> None:
        self._epoch = self.store.get_meta("epoch")
        ids, c = self.store.read_centroids()
        if len(ids):
            self._centroid_ids, self._centroids = ids, np.array(c)
            self._c_sq = (self._centroids ** 2).sum(1)
            self._n_indexed = self.store.conn.execute("SELECT SUM(size) FROM centroids").fetchone()[0] or 0
        else:
            self._centroid_ids = self._centroids = self._c_sq = None
            self._n_indexed = 0

    @property
    def avg_partition_size(self) -> float:
        return self._n_indexed / max(self.n_partitions, 1)

    # ---------- statistics / optimizer ----------
    def _load_optimizer(self) -> None:
        stats = self.store.get_meta("attr_stats")
        if stats is None:
            self.optimizer = None
            return
        cost = self.store.get_meta("cost_model")
        self.optimizer = Optimizer(Estimator.from_json(stats),
                                   CostModel(**json.loads(cost)) if cost else None)

    def analyze(self, calibrate_costs: bool = True) -> None:
        """Collect attribute statistics (and per-row access costs) for the optimizer."""
        est = Estimator.analyze(self.store.conn, self.store.attribute_columns())
        with self.store.transaction():
            self.store.set_meta("attr_stats", est.to_json())
            if calibrate_costs and self.n_partitions:
                self.store.set_meta("cost_model", json.dumps(asdict(calibrate(self))))
        self._load_optimizer()

    # ---------- writes ----------
    def add(self, vids: Sequence[int], vecs: np.ndarray,
            attrs: Optional[Sequence[Sequence]] = None, batch_size: int = 10_000) -> None:
        vecs = np.asarray(vecs, dtype=DTYPE)
        for s in range(0, len(vids), batch_size):
            with self.store.transaction():
                self.store.insert_delta(vids[s:s + batch_size], vecs[s:s + batch_size],
                                        None if attrs is None else attrs[s:s + batch_size])

    def delete(self, vids: Sequence[int]) -> int:
        """Delete vectors (indexed or delta) and their attributes. Returns rows removed."""
        removed = 0
        with self.store.transaction() as conn:
            for vid in vids:
                row = conn.execute("SELECT partition_id FROM vid_map WHERE vid = ?", (int(vid),)).fetchone()
                if row is None:
                    continue
                if row[0] == DELTA_PARTITION:
                    conn.execute("DELETE FROM delta WHERE vid = ?", (int(vid),))
                else:
                    conn.execute("DELETE FROM vectors WHERE partition_id = ? AND vid = ?", (row[0], int(vid)))
                    conn.execute("UPDATE centroids SET size = size - 1 WHERE partition_id = ?", (row[0],))
                conn.execute("DELETE FROM vid_map WHERE vid = ?", (int(vid),))
                conn.execute("DELETE FROM attributes WHERE vid = ?", (int(vid),))
                removed += 1
        self.reload_centroids()
        return removed

    def merge_delta(self, move_centroids: bool = True) -> int:
        """Incremental merge: assign every delta vector to its nearest centroid and move it
        into that partition, in one transaction. With move_centroids, each centroid becomes
        the running mean of its old members and the new ones. Returns vectors merged."""
        if self._centroids is None:
            raise RuntimeError("build() before merge_delta()")
        with self.store.transaction() as conn:
            if self.store.get_meta("epoch") != self._epoch:
                self.reload_centroids()  # another connection rebuilt since we cached centroids
            vids, x = self.store.read_delta()
            if len(vids) == 0:
                return 0
            a = self._centroid_ids[sqdist(x, self._centroids, self._c_sq).argmin(1)]
            conn.executemany("INSERT INTO vectors VALUES(?, ?, ?)",
                             ((int(p), int(v), to_blob(r)) for p, v, r in zip(a, vids, x)))
            conn.executemany("UPDATE vid_map SET partition_id = ? WHERE vid = ?",
                             ((int(p), int(v)) for p, v in zip(a, vids)))
            conn.execute("DELETE FROM delta")
            sizes = dict(conn.execute("SELECT partition_id, size FROM centroids"))
            for pid in np.unique(a):
                members = x[a == pid]
                old = sizes[int(pid)]
                i = int(np.flatnonzero(self._centroid_ids == pid)[0])
                c = self._centroids[i]
                if move_centroids:
                    c = (c * old + members.sum(0)) / (old + len(members))
                conn.execute("UPDATE centroids SET size = ?, vec = ? WHERE partition_id = ?",
                             (old + len(members), to_blob(c), int(pid)))
            self._bump_epoch()
        self.reload_centroids()
        return len(vids)

    def maintain(self, policy: MaintenancePolicy) -> str:
        """Apply the policy. Returns "none", "merge" or "rebuild"."""
        _, n_delta = self.store.count()
        action = "none"
        if policy.delta_max and n_delta >= policy.delta_max:
            self.merge_delta()
            action = "merge"
        if policy.rebuild_factor and self._centroids is not None:
            biggest = self.store.conn.execute("SELECT MAX(size) FROM centroids").fetchone()[0] or 0
            built_max = float(self.store.get_meta("build_max_size") or biggest)
            if biggest > policy.rebuild_factor * built_max:
                self.build(n_partitions=None)
                action = "rebuild"
        return action

    def _bump_epoch(self) -> None:
        self.store.set_meta("epoch", int(self.store.get_meta("epoch") or 0) + 1)

    def build(self, n_partitions: Optional[int] = None, n_epochs: int = 3, balance: float = 0.0,
              batch_size: int = 10_000, seed: int = 0) -> BuildStats:
        """(Re)cluster every vector (indexed + delta) and rewrite the partitions.

        Memory is bounded by batch_size and n_partitions, not N: training streams from SQLite,
        and the final assignment streams from a second WAL-snapshot connection while this
        connection writes the new partitions.
        """
        t0 = time.perf_counter()
        n_idx, n_delta = self.store.count()
        n = n_idx + n_delta
        k = n_partitions or max(1, int(round(math.sqrt(n))))

        def stream():
            return self.store.iter_all_vectors(batch_size)

        centroids, n = minibatch_kmeans(stream, k, n_epochs=n_epochs, balance=balance, seed=seed)
        c_sq = (centroids ** 2).sum(1)
        sizes = np.zeros(k)

        reader = sqlite3.connect(self.store.path)
        reader.execute("BEGIN")  # pin the snapshot we are about to rewrite
        cur = reader.execute("SELECT vid, vec FROM vectors UNION ALL SELECT vid, vec FROM delta")
        with self.store.transaction() as conn:
            conn.execute("DELETE FROM vectors")
            conn.execute("DELETE FROM delta")
            seen = 0
            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                vids = np.array([r[0] for r in rows], dtype=np.int64)
                x = np.frombuffer(b"".join(r[1] for r in rows), dtype=DTYPE).reshape(len(rows), -1)
                seen += len(x)
                a = balanced_assign(sqdist(x, centroids, c_sq), sizes, expected=seen / k, lam=balance)
                conn.executemany("INSERT INTO vectors VALUES(?, ?, ?)",
                                 ((int(p), int(v), b) for p, v, b in zip(a, vids, (r[1] for r in rows))))
                conn.executemany("INSERT OR REPLACE INTO vid_map VALUES(?, ?)",
                                 ((int(v), int(p)) for v, p in zip(vids, a)))
            self.store.write_centroids(centroids, sizes.astype(int))
            self.store.set_meta("n_partitions", k)
            self.store.set_meta("build_max_size", int(sizes.max()))
            self._bump_epoch()
        reader.execute("COMMIT")
        reader.close()
        self.reload_centroids()
        return BuildStats(n_vectors=n, n_partitions=k, seconds=time.perf_counter() - t0,
                          sizes=sizes.astype(int))

    # ---------- reads ----------
    def probe(self, q: np.ndarray, nprobe: int) -> np.ndarray:
        """Partition ids of the nprobe centroids nearest to q."""
        if self._centroids is None:
            return np.empty(0, dtype=np.int64)
        d = sqdist(q[None, :], self._centroids, self._c_sq)[0]
        return self._centroid_ids[topk(d, min(nprobe, len(d)))]

    def search(self, q: np.ndarray, k: int = 100, nprobe: int = 8,
               filter: Optional[Filter] = None, plan: str = "auto",
               stats: Optional[SearchStats] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Approximate top-k by squared L2. Returns (vids, distances), nearest first.

        With a filter, plan is "pre" (filter via attribute index, then exact distances:
        100% recall), "post" (scan the nprobe nearest partitions joined with the filter), or
        "auto" (the optimizer picks; requires analyze())."""
        t0 = time.perf_counter()
        q = np.asarray(q, dtype=DTYPE)
        st = stats if stats is not None else SearchStats()
        with self.snapshot():
            res = self._search(q, k, nprobe, filter, plan, st)
        st.seconds += time.perf_counter() - t0
        return res

    @contextlib.contextmanager
    def snapshot(self):
        """Run reads in one transaction (one WAL snapshot); reload centroids if another
        connection rewrote the partitions since we last looked."""
        conn = self.store.conn
        own = not conn.in_transaction
        if own:
            conn.execute("BEGIN")
        try:
            if self.store.get_meta("epoch") != self._epoch:
                self.reload_centroids()
            yield
        finally:
            if own:
                conn.execute("COMMIT")

    def _search(self, q, k, nprobe, filter, plan, st):
        if filter:
            if plan == "auto":
                if self.optimizer is None:
                    raise RuntimeError("call analyze() before plan='auto'")
                st.plan = self.optimizer.choose(filter, k, nprobe, self.avg_partition_size,
                                                probed=self.probe(q, nprobe))
                plan = st.plan.name
            if plan == "pre":
                res = self._search_pre(q, k, filter, st)
            elif plan == "post":
                res = self._search_ivf(q, k, nprobe, filter, st)
            else:
                raise ValueError(plan)
        else:
            res = self._search_ivf(q, k, nprobe, None, st)
        return res

    def _search_ivf(self, q, k, nprobe, f: Optional[Filter], st: SearchStats):
        # Stream one partition at a time into a running top-k, so peak memory is one
        # partition + k rows regardless of nprobe or N.
        pids = self.probe(q, nprobe)
        if f:
            # CROSS JOIN pins the join order: scan the partition, then check each row's
            # attributes -- a true post-filter, not an attribute-index lookup.
            where, params = to_sql(f, "a", self.store.attribute_columns())
            part_sql = ("SELECT v.vid, v.vec FROM vectors v CROSS JOIN attributes a ON a.vid = v.vid "
                        f"WHERE v.partition_id = ? AND {where}")
            delta_sql = f"SELECT d.vid, d.vec FROM delta d CROSS JOIN attributes a ON a.vid = d.vid WHERE {where}"
        else:
            params = []
            part_sql = "SELECT vid, vec FROM vectors WHERE partition_id = ?"
            delta_sql = "SELECT vid, vec FROM delta"
        best_v = np.empty(0, dtype=np.int64)
        best_d = np.empty(0, dtype=DTYPE)
        conn = self.store.conn
        for pid in pids:
            rows = conn.execute(part_sql, [int(pid), *params]).fetchall()
            best_v, best_d, n = self._merge_rows(q, k, rows, best_v, best_d)
            st.vectors_scanned += n
        rows = conn.execute(delta_sql, params).fetchall()
        best_v, best_d, n = self._merge_rows(q, k, rows, best_v, best_d)
        st.delta_scanned += n
        st.partitions_probed += len(pids)
        return best_v, best_d

    def _search_pre(self, q, k, f: Filter, st: SearchStats, chunk: int = 4096):
        where, params = to_sql(f, "a", self.store.attribute_columns())
        sql = ("SELECT v.vid, v.vec FROM attributes a "
               "JOIN vid_map m ON m.vid = a.vid "
               "JOIN vectors v ON v.partition_id = m.partition_id AND v.vid = m.vid "
               f"WHERE {where} "
               "UNION ALL "
               f"SELECT d.vid, d.vec FROM attributes a JOIN delta d ON d.vid = a.vid WHERE {where}")
        cur = self.store.conn.execute(sql, params + params)
        best_v = np.empty(0, dtype=np.int64)
        best_d = np.empty(0, dtype=DTYPE)
        while True:
            rows = cur.fetchmany(chunk)
            if not rows:
                break
            best_v, best_d, n = self._merge_rows(q, k, rows, best_v, best_d)
            st.vectors_scanned += n
        return best_v, best_d

    def _merge_rows(self, q, k, rows, best_v, best_d):
        if not rows:
            return best_v, best_d, 0
        v = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
        x = np.frombuffer(b"".join(r[1] for r in rows), dtype=DTYPE).reshape(len(rows), -1)
        best_v, best_d = _merge_topk(best_v, best_d, v, _qdist(q, x), k)
        return best_v, best_d, len(rows)

    def close(self) -> None:
        self.store.close()
