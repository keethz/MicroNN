"""Dataset loading, synthetic data/attributes, and exact ground truth.

SIFT1M / GloVe are not downloaded automatically. Get them from:
  SIFT1M: http://corpus-texmex.irisa.fr/  (sift.tar.gz -> sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs)
  GloVe:  https://nlp.stanford.edu/projects/glove/  (glove.6B.zip -> glove.6B.100d.txt)
and put them under data/.
"""
import hashlib
import os
from typing import Optional, Tuple

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# ---------- file formats ----------
def read_fvecs(path: str, limit: Optional[int] = None) -> np.ndarray:
    """TEXMEX .fvecs: each record is int32 dim followed by dim float32."""
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    rows = raw.reshape(-1, dim + 1)
    if limit is not None:
        rows = rows[:limit]
    return rows[:, 1:].copy().view(np.float32)


def read_ivecs(path: str, limit: Optional[int] = None) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    rows = raw.reshape(-1, dim + 1)
    if limit is not None:
        rows = rows[:limit]
    return rows[:, 1:].copy()


def read_glove(path: str, limit: Optional[int] = None) -> np.ndarray:
    vecs = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            vecs.append(np.asarray(line.rstrip().split(" ")[1:], dtype=np.float32))
    return np.vstack(vecs)


def load_sift(n_base: Optional[int] = None, n_query: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    d = os.path.join(DATA_DIR, "sift")
    return (read_fvecs(os.path.join(d, "sift_base.fvecs"), n_base),
            read_fvecs(os.path.join(d, "sift_query.fvecs"), n_query))


def load_glove(n_base: Optional[int] = None, n_query: int = 1000, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """GloVe has no query split: hold out n_query random rows as queries."""
    x = read_glove(os.path.join(DATA_DIR, "glove", "glove.6B.100d.txt"),
                   None if n_base is None else n_base + n_query)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12  # angular -> L2 on unit vectors
    perm = np.random.default_rng(seed).permutation(len(x))
    return x[perm[n_query:]], x[perm[:n_query]]


# ---------- synthetic data ----------
def synthetic(n: int, dim: int = 64, n_clusters: int = 50, n_query: int = 100,
              seed: int = 0, spread: float = 10.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gaussian mixture with unit-variance clusters whose centers have std `spread`.
    Large spread = well separated (easy); spread < 1 = overlapping (SIFT-like difficulty).
    Returns (base, queries, cluster label of each base vector)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, spread, (n_clusters, dim)).astype(np.float32)
    labels = rng.integers(0, n_clusters, n)
    base = centers[labels] + rng.normal(0, 1, (n, dim)).astype(np.float32)
    q_labels = rng.integers(0, n_clusters, n_query)
    queries = centers[q_labels] + rng.normal(0, 1, (n_query, dim)).astype(np.float32)
    return base.astype(np.float32), queries.astype(np.float32), labels


# ---------- synthetic attributes (for hybrid queries) ----------
def attr_uniform(n: int, n_values: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_values, n)


def attr_zipf(n: int, n_values: int, a: float = 1.5, seed: int = 0) -> np.ndarray:
    """Skewed categorical attribute in [0, n_values)."""
    rng = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, n_values + 1) ** a
    return rng.choice(n_values, n, p=p / p.sum())


def attr_correlated(vectors: np.ndarray, n_values: int, noise: float = 0.1, seed: int = 0) -> np.ndarray:
    """Attribute tied to vector-space location: value = bucket of the projection onto a random
    direction, with a `noise` fraction replaced by random values. Similar vectors share values,
    which breaks the optimizer's independence assumption (research extension A)."""
    rng = np.random.default_rng(seed)
    proj = vectors @ rng.normal(size=vectors.shape[1])
    edges = np.quantile(proj, np.linspace(0, 1, n_values + 1)[1:-1])
    vals = np.searchsorted(edges, proj)
    flip = rng.random(len(vals)) < noise
    vals[flip] = rng.integers(0, n_values, flip.sum())
    return vals


# ---------- ground truth ----------
def ground_truth(base: np.ndarray, queries: np.ndarray, k: int = 100,
                 mask: Optional[np.ndarray] = None, chunk: int = 50_000) -> np.ndarray:
    """Exact top-k by squared L2, scanning base in chunks to bound memory.
    `mask` (bool, len(base)) restricts the search to matching rows (filtered ground truth).
    Returns (n_query, k) int64 row indices into base, -1-padded if fewer than k match."""
    nq = len(queries)
    best_d = np.full((nq, k), np.inf, dtype=np.float32)
    best_i = np.full((nq, k), -1, dtype=np.int64)
    q_sq = (queries.astype(np.float32) ** 2).sum(1, keepdims=True)
    for start in range(0, len(base), chunk):
        b = base[start:start + chunk].astype(np.float32)
        d = q_sq - 2 * queries @ b.T + (b ** 2).sum(1)[None, :]
        if mask is not None:
            d[:, ~mask[start:start + chunk]] = np.inf
        idx = np.broadcast_to(np.arange(start, start + len(b)), d.shape)
        all_d = np.concatenate([best_d, d], axis=1)
        all_i = np.concatenate([best_i, idx], axis=1)
        top = np.argpartition(all_d, k - 1, axis=1)[:, :k] if all_d.shape[1] > k else np.argsort(all_d, axis=1)
        best_d = np.take_along_axis(all_d, top, 1)
        best_i = np.take_along_axis(all_i, top, 1)
    order = np.argsort(best_d, axis=1)
    best_d = np.take_along_axis(best_d, order, 1)
    best_i = np.take_along_axis(best_i, order, 1)
    best_i[np.isinf(best_d)] = -1
    return best_i


def recall_at_k(found: np.ndarray, truth: np.ndarray, k: int) -> float:
    """Mean |found ∩ truth| / |truth| over queries, ignoring -1 padding."""
    total = 0.0
    for f, t in zip(found, truth):
        t = set(t[:k][t[:k] >= 0].tolist())
        if t:
            total += len(t & set(f[:k].tolist())) / len(t)
        else:
            total += 1.0
    return total / len(found)


def cached_ground_truth(name: str, base: np.ndarray, queries: np.ndarray, k: int = 100,
                        mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Cache keyed by a hash of the data, so regenerated datasets never reuse stale answers."""
    h = hashlib.sha1()
    for a in (base[:: max(1, len(base) // 1000)], queries, mask):
        if a is not None:
            h.update(np.ascontiguousarray(a).tobytes())
    h.update(f"{base.shape}{queries.shape}".encode())
    os.makedirs(os.path.join(DATA_DIR, "gt"), exist_ok=True)
    path = os.path.join(DATA_DIR, "gt", f"{name}_k{k}_{h.hexdigest()[:12]}.npy")
    if os.path.exists(path):
        return np.load(path)
    gt = ground_truth(base, queries, k, mask)
    np.save(path, gt)
    return gt
