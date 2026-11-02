"""Mini-batch k-means with a cluster-size balance penalty (PLAN.md section 2).

Data is consumed as a stream of (vids, X) batches, so memory depends on the batch size,
the init sample and k -- never on the dataset size N (claim C4).
"""
from typing import Callable, Iterator, Optional, Tuple

import numpy as np

BatchStream = Callable[[], Iterator[Tuple[np.ndarray, np.ndarray]]]


def sqdist(x: np.ndarray, c: np.ndarray, c_sq: Optional[np.ndarray] = None) -> np.ndarray:
    """Squared L2 distances, shape (len(x), len(c))."""
    if c_sq is None:
        c_sq = (c ** 2).sum(1)
    d = (x ** 2).sum(1, keepdims=True) - 2 * x @ c.T + c_sq[None, :]
    return np.maximum(d, 0, out=d)


def balanced_assign(d: np.ndarray, sizes: np.ndarray, expected: float, lam: float) -> np.ndarray:
    """Assign each row to argmin_j d_ij + lam * scale * sizes_j / expected.

    `scale` is the batch's mean nearest-centroid distance, which makes lam unitless:
    lam=0 is plain k-means, lam=1 means a cluster twice the expected size costs about one
    typical distance extra. `sizes` is updated in place as rows are assigned.
    """
    if lam <= 0:
        a = d.argmin(1)
        np.add.at(sizes, a, 1)
        return a
    scale = float(d.min(1).mean()) or 1.0
    w = lam * scale / max(expected, 1.0)
    out = np.empty(len(d), dtype=np.int64)
    # Assign in small chunks so the penalty reacts to sizes as they grow.
    step = max(1, len(d) // 16)
    for s in range(0, len(d), step):
        a = (d[s:s + step] + w * sizes[None, :]).argmin(1)
        np.add.at(sizes, a, 1)
        out[s:s + step] = a
    return out


def _reservoir_sample(stream: BatchStream, size: int, rng: np.random.Generator) -> Tuple[np.ndarray, int]:
    sample, seen = None, 0
    for _, x in stream():
        if sample is None:
            sample = np.empty((size, x.shape[1]), dtype=np.float32)
        fill = min(max(size - seen, 0), len(x))
        sample[seen:seen + fill] = x[:fill]
        rest = x[fill:]
        if len(rest):
            # Algorithm R, vectorized per batch: row with global index t replaces slot j ~ U[0, t].
            t = np.arange(seen + fill, seen + len(x))
            j = (rng.random(len(rest)) * (t + 1)).astype(np.int64)
            keep = j < size
            sample[j[keep]] = rest[keep]
        seen += len(x)
    if sample is None:
        raise ValueError("empty dataset")
    return sample[:min(seen, size)], seen


def kmeans_pp(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    centers = np.empty((k, x.shape[1]), dtype=np.float32)
    centers[0] = x[rng.integers(len(x))]
    closest = sqdist(x, centers[:1])[:, 0]
    for i in range(1, k):
        p = closest / closest.sum() if closest.sum() > 0 else None
        centers[i] = x[rng.choice(len(x), p=p)]
        closest = np.minimum(closest, sqdist(x, centers[i:i + 1])[:, 0])
    return centers


def minibatch_kmeans(stream: BatchStream, k: int, n_epochs: int = 3, balance: float = 0.0,
                     init_sample: Optional[int] = None, seed: int = 0) -> Tuple[np.ndarray, int]:
    """Sculley-style mini-batch k-means. `stream()` must yield a fresh pass over the data.

    Returns (centroids (k, dim), N).
    """
    rng = np.random.default_rng(seed)
    init_sample = init_sample or max(20 * k, 10_000)
    sample, n = _reservoir_sample(stream, init_sample, rng)
    if n < k:
        raise ValueError(f"need at least k={k} vectors, got {n}")
    c = kmeans_pp(sample, k, rng)
    del sample
    v = np.zeros(k)  # per-center count of points used for updates (sets the learning rate)
    for _ in range(n_epochs):
        sizes = np.zeros(k)
        seen = 0
        last_x = None
        for _, x in stream():
            x = np.asarray(x, dtype=np.float32)
            seen += len(x)
            a = balanced_assign(sqdist(x, c), sizes, expected=seen / k, lam=balance)
            m = np.bincount(a, minlength=k)
            sums = np.zeros_like(c)
            np.add.at(sums, a, x)
            hit = m > 0
            v[hit] += m[hit]
            eta = (m[hit] / v[hit])[:, None]
            c[hit] = (1 - eta) * c[hit] + eta * (sums[hit] / m[hit][:, None])
            last_x = x
        # Re-seed centers that got no points this epoch.
        dead = np.flatnonzero(sizes == 0)
        if len(dead) and last_x is not None:
            c[dead] = last_x[rng.choice(len(last_x), len(dead), replace=len(dead) > len(last_x))]
            v[dead] = 0
    return c, n
