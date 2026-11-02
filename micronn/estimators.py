"""Correlation-aware selectivity estimators (research extension A, PLAN.md section 10).

The baseline `optimizer.Estimator` assumes (1) predicates on different columns are
independent and (2) matching rows are spread evenly over partitions, so a post-filter scan
of nprobe partitions finds sel * scanned matches. Both fail on correlated data. Estimators
here keep the same interface (`n_total`, `filter_selectivity`) and may add
`expected_matches(f, pids)`, which the Optimizer prefers when present.

  ColumnGroupEstimator   joint frequencies for low-cardinality column pairs (fixes 1)
  SampleEstimator        uniform row sample, filters evaluated on it (fixes 1)
  StratifiedSampleEstimator
                         fixed number of sampled rows per partition (fixes 1 and 2)
"""
import itertools
from typing import Dict, List, Optional, Sequence

import numpy as np

from .filters import Filter, Pred
from .optimizer import Estimator

MIN_HITS = 5  # below this many sample hits, a sample estimate is mostly noise


def _eval(p: Pred, col: np.ndarray) -> np.ndarray:
    if p.op == "=": return col == p.value
    if p.op == "IN": return np.isin(col, list(p.value))
    if p.op == "<": return col < p.value
    if p.op == "<=": return col <= p.value
    if p.op == ">": return col > p.value
    if p.op == ">=": return col >= p.value
    if p.op == "BETWEEN": return (col >= p.value[0]) & (col <= p.value[1])
    raise ValueError(p.op)


def _sample_mask(f: Filter, sample: Dict[str, np.ndarray]) -> np.ndarray:
    n = len(next(iter(sample.values())))
    m = np.ones(n, bool)
    for p in f:
        m &= _eval(p, sample[p.col])
    return m


def _blend(sample_sel: float, hits: float, n_eff: float, fallback: float) -> float:
    """Trust the sample when it saw enough hits. Otherwise clamp the fallback (independence)
    estimate into a rough interval implied by the hit count: at most ~(hits+3)/n (≈95% upper
    bound for a Poisson count; caps anti-correlated overestimates) and, if anything was hit,
    at least hits/(4n) (lifts correlated underestimates)."""
    if hits >= MIN_HITS:
        return sample_sel
    n_eff = max(n_eff, 1)
    lo = hits / (4 * n_eff)
    hi = (hits + 3) / n_eff
    return min(max(fallback, lo), hi)


class ColumnGroupEstimator(Estimator):
    """Independence, except that equality predicates on a column pair with stored joint
    statistics use the joint frequency (like PostgreSQL CREATE STATISTICS ... (mcv))."""

    name = "column-group"

    def __init__(self, base: Estimator, joint: Dict[tuple, Dict[tuple, int]]):
        super().__init__(base.n_total, base.columns)
        self.joint = joint

    @classmethod
    def analyze(cls, conn, columns: List[str], max_combos: int = 20_000) -> "ColumnGroupEstimator":
        base = Estimator.analyze(conn, columns)
        joint = {}
        low = [c for c in columns if base.columns[c].freq is not None]
        for a, b in itertools.combinations(low, 2):
            if base.columns[a].ndv * base.columns[b].ndv > max_combos:
                continue
            rows = conn.execute(f"SELECT {a}, {b}, COUNT(*) FROM attributes "
                                f"WHERE {a} IS NOT NULL AND {b} IS NOT NULL GROUP BY {a}, {b}")
            joint[(a, b)] = {(x, y): c for x, y, c in rows}
        return cls(base, joint)

    @property
    def memory_entries(self) -> int:
        return sum(len(t) for t in self.joint.values())

    def filter_selectivity(self, f: Filter) -> float:
        eqs = {p.col: p for p in f if p.op == "="}
        used, sel = set(), 1.0
        for (a, b), table in self.joint.items():
            if a in eqs and b in eqs and a not in used and b not in used:
                sel *= table.get((eqs[a].value, eqs[b].value), 0) / self.n_total
                used |= {a, b}
        for p in f:
            if p.col not in used:
                sel *= self.pred_selectivity(p)
        return sel


class SampleEstimator(Estimator):
    """Uniform random sample of attribute rows; conjunctions are evaluated on the sample."""

    name = "sample"

    def __init__(self, base: Estimator, sample: Dict[str, np.ndarray]):
        super().__init__(base.n_total, base.columns)
        self.sample = sample
        self.n_sample = len(next(iter(sample.values())))

    @classmethod
    def analyze(cls, conn, columns: List[str], size: int, seed: int = 0) -> "SampleEstimator":
        base = Estimator.analyze(conn, columns)
        vids = np.array([v for (v,) in conn.execute("SELECT vid FROM attributes")])
        pick = np.random.default_rng(seed).choice(vids, min(size, len(vids)), replace=False)
        return cls(base, _fetch(conn, columns, pick))

    def filter_selectivity(self, f: Filter) -> float:
        hits = float(_sample_mask(f, self.sample).sum())
        return _blend(hits / self.n_sample, hits, self.n_sample,
                      Estimator.filter_selectivity(self, f))


class StratifiedSampleEstimator(Estimator):
    """`per_partition` sampled rows from every IVF partition (plus the delta store).

    Global selectivity: each partition's hit rate weighted by its size.
    expected_matches(f, pids): sum over the probed partitions of size * hit rate -- the
    number of matches a post-filter scan of exactly those partitions will see, which
    captures filters whose matches cluster in vector space."""

    name = "stratified"

    def __init__(self, base: Estimator, sample: Dict[str, np.ndarray], part: np.ndarray,
                 sizes: Dict[int, int]):
        super().__init__(base.n_total, base.columns)
        self.sample = sample
        self.part = part
        self.sizes = sizes
        self.pids = np.array(sorted(sizes))
        self._pid_index = {p: i for i, p in enumerate(self.pids)}
        counts = np.bincount([self._pid_index[p] for p in part], minlength=len(self.pids))
        self.size_arr = np.array([sizes[p] for p in self.pids], dtype=float)
        self.sample_counts = counts.astype(float)
        self.row_part = np.array([self._pid_index[p] for p in part])

    @classmethod
    def analyze(cls, conn, columns: List[str], per_partition: int, seed: int = 0) -> "StratifiedSampleEstimator":
        base = Estimator.analyze(conn, columns)
        rng = np.random.default_rng(seed)
        sizes = dict(conn.execute("SELECT partition_id, COUNT(*) FROM vid_map GROUP BY partition_id"))
        picks, parts = [], []
        for pid in sizes:
            vids = np.array([v for (v,) in conn.execute("SELECT vid FROM vid_map WHERE partition_id = ?", (pid,))])
            take = rng.choice(vids, min(per_partition, len(vids)), replace=False)
            picks.append(take)
            parts.append(np.full(len(take), pid))
        pick = np.concatenate(picks)
        return cls(base, _fetch(conn, columns, pick), np.concatenate(parts), sizes)

    @property
    def n_sample(self) -> int:
        return len(self.part)

    def _partition_hits(self, f: Filter) -> np.ndarray:
        m = _sample_mask(f, self.sample)
        return np.bincount(self.row_part[m], minlength=len(self.pids)).astype(float)

    def filter_selectivity(self, f: Filter) -> float:
        hits = self._partition_hits(f)
        rate = np.divide(hits, self.sample_counts, out=np.zeros_like(hits), where=self.sample_counts > 0)
        est = float((rate * self.size_arr).sum()) / self.n_total
        return _blend(est, hits.sum(), self.n_sample, Estimator.filter_selectivity(self, f))

    def expected_matches(self, f: Filter, probed: Sequence[int]) -> float:
        hits = self._partition_hits(f)
        idx = [self._pid_index[int(p)] for p in probed if int(p) in self._pid_index]
        if -1 in self._pid_index:
            idx.append(self._pid_index[-1])  # the delta store is always scanned
        h, c, s = hits[idx], self.sample_counts[idx], self.size_arr[idx]
        if h.sum() >= MIN_HITS:
            return float((np.divide(h, c, out=np.zeros_like(h), where=c > 0) * s).sum())
        # Too few local hits to trust: global estimate spread evenly, capped by the local bound.
        uniform = self.filter_selectivity(f) * s.sum()
        return min(uniform, (h.sum() + 3) / max(c.sum(), 1) * s.sum())


def _fetch(conn, columns: List[str], vids: np.ndarray) -> Dict[str, np.ndarray]:
    """Attribute values for the given vids, in the same order."""
    out = {c: [] for c in columns}
    sel = ", ".join(columns)
    for s in range(0, len(vids), 900):  # stay under SQLite's bound-parameter limit
        chunk = [int(v) for v in vids[s:s + 900]]
        rows = dict((r[0], r[1:]) for r in conn.execute(
            f"SELECT vid, {sel} FROM attributes WHERE vid IN ({', '.join('?' * len(chunk))})", chunk))
        for v in chunk:
            r = rows.get(v, (None,) * len(columns))
            for c, x in zip(columns, r):
                out[c].append(np.nan if x is None else x)
    return {c: np.asarray(v) for c, v in out.items()}
