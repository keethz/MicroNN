"""Baselines for the validation experiments (PLAN.md section 7)."""
from typing import Tuple

import numpy as np

from .clustering import sqdist
from .index import MicroNN, topk
from .storage import DTYPE


class InMemoryIVF:
    """Same IVF algorithm as MicroNN with every partition held in RAM as NumPy arrays.

    Built from an existing MicroNN index so both systems share identical centroids and
    assignments; any latency/memory difference is then due to storage alone (claims C2, C3).
    """

    def __init__(self, db: MicroNN):
        self._centroid_ids = db._centroid_ids
        self.centroids = db._centroids.copy()
        self._c_sq = (self.centroids ** 2).sum(1)
        self.vids, self.vecs = {}, {}
        for pid in self._centroid_ids:
            v, x = db.store.read_partitions([pid])
            self.vids[int(pid)], self.vecs[int(pid)] = v, x.copy()
        self.delta_vids, self.delta_vecs = (a.copy() for a in db.store.read_delta())

    @property
    def nbytes(self) -> int:
        return (sum(x.nbytes + self.vids[p].nbytes for p, x in self.vecs.items())
                + self.centroids.nbytes + self.delta_vecs.nbytes)

    def search(self, q: np.ndarray, k: int = 100, nprobe: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q, dtype=DTYPE)
        d = sqdist(q[None, :], self.centroids, self._c_sq)[0]
        pids = self._centroid_ids[topk(d, min(nprobe, len(d)))]
        vids = np.concatenate([self.vids[int(p)] for p in pids] + [self.delta_vids])
        x = np.vstack([self.vecs[int(p)] for p in pids] + [self.delta_vecs])
        d = sqdist(q[None, :], x)[0]
        best = topk(d, k)
        return vids[best], d[best]


class BruteForce:
    """Exact search over an in-memory matrix."""

    def __init__(self, vids: np.ndarray, vecs: np.ndarray):
        self.vids = np.asarray(vids)
        self.vecs = np.asarray(vecs, dtype=DTYPE)
        self._sq = (self.vecs ** 2).sum(1)

    def search(self, q: np.ndarray, k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        d = sqdist(np.asarray(q, dtype=DTYPE)[None, :], self.vecs, self._sq)[0]
        best = topk(d, k)
        return self.vids[best], d[best]
