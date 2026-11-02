"""Selectivity estimation and pre- vs post-filter plan choice (PLAN.md section 4).

Baseline estimator (what the paper describes): per-column statistics combined under the
**independence assumption**, sel(A AND B) = sel(A) * sel(B). Research extension A replaces
`Estimator.filter_selectivity` with something correlation-aware.
"""
import bisect
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .filters import Filter, Pred

MAX_FREQ_VALUES = 1000  # columns with at most this many distinct values keep exact counts
N_BUCKETS = 100         # equi-depth histogram buckets otherwise


@dataclass
class ColumnStats:
    n: int                                   # non-null rows
    ndv: int
    freq: Optional[Dict] = None              # value -> count (low-cardinality columns)
    bounds: List = field(default_factory=list)  # equi-depth bucket boundaries, len B+1

    def selectivity(self, p: Pred, n_total: int) -> float:
        if n_total == 0:
            return 0.0
        if self.freq is not None:
            hit = sum(c for v, c in self.freq.items() if _match(p, v))
            return hit / n_total
        frac_nonnull = self.n / n_total
        if p.op == "=":
            return frac_nonnull / max(self.ndv, 1)
        if p.op == "IN":
            return min(1.0, frac_nonnull * len(p.value) / max(self.ndv, 1))
        lo, hi, lo_inc, hi_inc = _range(p)
        return frac_nonnull * max(0.0, self._cdf(hi, hi_inc) - self._cdf(lo, not lo_inc))

    def _cdf(self, x, inclusive: bool) -> float:
        """Fraction of non-null values <= x (or < x), linear interpolation inside a bucket."""
        b = self.bounds
        if x is None:
            return 1.0
        if x == -math.inf:
            return 0.0
        if x < b[0]:
            return 0.0
        if x > b[-1] or (x == b[-1] and inclusive):
            return 1.0
        i = bisect.bisect_right(b, x) - 1
        i = min(i, len(b) - 2)
        lo, hi = b[i], b[i + 1]
        if hi == lo:
            within = 0.0
        elif isinstance(x, (int, float)):
            within = (x - lo) / (hi - lo)
        else:
            within = 0.5  # non-numeric (TEXT) bucket: assume the middle
        return (i + within) / (len(b) - 1)


def _range(p: Pred):
    """(lo, hi, lo_inclusive, hi_inclusive) for range predicates."""
    if p.op == "<": return -math.inf, p.value, False, False
    if p.op == "<=": return -math.inf, p.value, False, True
    if p.op == ">": return p.value, None, False, True
    if p.op == ">=": return p.value, None, True, True
    if p.op == "BETWEEN": return p.value[0], p.value[1], True, True
    raise ValueError(p.op)


def _match(p: Pred, v) -> bool:
    if p.op == "=": return v == p.value
    if p.op == "IN": return v in p.value
    if p.op == "<": return v < p.value
    if p.op == "<=": return v <= p.value
    if p.op == ">": return v > p.value
    if p.op == ">=": return v >= p.value
    if p.op == "BETWEEN": return p.value[0] <= v <= p.value[1]
    raise ValueError(p.op)


class Estimator:
    """Per-column stats + independence assumption."""

    name = "independence"

    def __init__(self, n_total: int, columns: Dict[str, ColumnStats]):
        self.n_total = n_total
        self.columns = columns

    @classmethod
    def analyze(cls, conn, columns: List[str]) -> "Estimator":
        """Build statistics with one pass per column (like SQL ANALYZE)."""
        n_total = conn.execute("SELECT COUNT(*) FROM attributes").fetchone()[0]
        stats = {}
        for col in columns:
            n, ndv = conn.execute(
                f"SELECT COUNT({col}), COUNT(DISTINCT {col}) FROM attributes").fetchone()
            if ndv <= MAX_FREQ_VALUES:
                freq = dict(conn.execute(
                    f"SELECT {col}, COUNT(*) FROM attributes WHERE {col} IS NOT NULL GROUP BY {col}"))
                stats[col] = ColumnStats(n=n, ndv=ndv, freq=freq)
            else:
                # Stream values in index order and keep every (n/B)-th as a bucket boundary.
                step = max(1, n // N_BUCKETS)
                bounds, i = [], 0
                cur = conn.execute(f"SELECT {col} FROM attributes WHERE {col} IS NOT NULL ORDER BY {col}")
                last = None
                for (v,) in cur:
                    if i % step == 0:
                        bounds.append(v)
                    last, i = v, i + 1
                if bounds[-1] != last:
                    bounds.append(last)
                stats[col] = ColumnStats(n=n, ndv=ndv, bounds=bounds)
        return cls(n_total, stats)

    def pred_selectivity(self, p: Pred) -> float:
        s = self.columns.get(p.col)
        return 1.0 if s is None else s.selectivity(p, self.n_total)

    def filter_selectivity(self, f: Filter) -> float:
        sel = 1.0
        for p in f:
            sel *= self.pred_selectivity(p)
        return sel

    # persistence (stored in the meta table as JSON)
    def to_json(self) -> str:
        cols = {}
        for c, s in self.columns.items():
            d = asdict(s)
            if s.freq is not None:
                d["freq"] = [[k, v] for k, v in s.freq.items()]
            cols[c] = d
        return json.dumps(dict(n_total=self.n_total, columns=cols))

    @classmethod
    def from_json(cls, text: str) -> "Estimator":
        d = json.loads(text)
        cols = {}
        for c, s in d["columns"].items():
            if s["freq"] is not None:
                s["freq"] = {k: v for k, v in s["freq"]}
            cols[c] = ColumnStats(**s)
        return cls(d["n_total"], cols)


@dataclass
class CostModel:
    """Per-row costs (seconds). pre: indexed lookup of one matching row (attribute index ->
    vid_map -> vectors). scan: one row of a sequential partition scan with the filter join."""
    point: float = 3e-6
    scan: float = 1e-6


@dataclass
class Plan:
    name: str                  # "pre" or "post"
    est_selectivity: float
    est_cost_pre: float
    est_cost_post: float
    reason: str


class Optimizer:
    def __init__(self, estimator: Estimator, cost: Optional[CostModel] = None,
                 recall_guard: float = 1.0):
        """recall_guard: choose pre-filtering whenever the probed partitions are expected to
        contain fewer than recall_guard * k matching rows (post-filtering would then return
        fewer than k results). 0 disables the guard (pure cost-based choice)."""
        self.est = estimator
        self.cost = cost or CostModel()
        self.recall_guard = recall_guard

    def choose(self, f: Filter, k: int, nprobe: int, avg_partition: float,
               probed: Optional[Sequence[int]] = None) -> Plan:
        """probed: partition ids a post-filter would scan. Estimators that implement
        expected_matches(f, probed) then predict matches in those partitions directly
        instead of assuming matches are spread evenly (sel * scanned)."""
        sel = self.est.filter_selectivity(f)
        n = self.est.n_total
        c_pre = sel * n * self.cost.point
        scanned = min(nprobe * avg_partition, n)
        c_post = scanned * self.cost.scan
        if probed is not None and hasattr(self.est, "expected_matches"):
            expected_matches = self.est.expected_matches(f, probed)
        else:
            expected_matches = sel * scanned
        if self.recall_guard > 0 and expected_matches < self.recall_guard * k:
            return Plan("pre", sel, c_pre, c_post,
                        f"post would find ~{expected_matches:.0f} < {self.recall_guard:g}k matches")
        if c_pre <= c_post:
            return Plan("pre", sel, c_pre, c_post, "cheaper")
        return Plan("post", sel, c_pre, c_post, "cheaper")


def calibrate(db, n_samples: int = 20, seed: int = 0) -> CostModel:
    """Measure per-row costs of the two access paths on this database."""
    conn = db.store.conn
    rng = np.random.default_rng(seed)
    pids = rng.choice(db._centroid_ids, min(n_samples, db.n_partitions), replace=False)
    for pid in pids:  # warm-up pass: calibrate on a warm cache, like the queries it predicts
        conn.execute("SELECT vid, vec FROM vectors WHERE partition_id = ?", (int(pid),)).fetchall()
    t, rows = time.perf_counter(), 0
    for pid in pids:
        rows += len(conn.execute(
            "SELECT v.vid, v.vec FROM vectors v CROSS JOIN attributes a ON a.vid = v.vid "
            "WHERE v.partition_id = ?", (int(pid),)).fetchall())
    scan = (time.perf_counter() - t) / max(rows, 1)
    max_vid = conn.execute("SELECT MAX(vid) FROM vid_map").fetchone()[0] or 0
    vids = rng.integers(0, max_vid + 1, max(rows // 4, 100))
    t = time.perf_counter()
    for vid in vids:
        conn.execute("SELECT v.vec FROM vid_map m JOIN vectors v "
                     "ON v.partition_id = m.partition_id AND v.vid = m.vid WHERE m.vid = ?",
                     (int(vid),)).fetchall()
    # A pre-filter fetch is one batched join, cheaper than separate statements; the
    # statement overhead is measured separately and removed.
    t_point = (time.perf_counter() - t) / len(vids)
    t = time.perf_counter()
    for _ in range(len(vids)):
        conn.execute("SELECT 1").fetchall()
    overhead = (time.perf_counter() - t) / len(vids)
    return CostModel(point=max(t_point - overhead, scan), scan=scan)
