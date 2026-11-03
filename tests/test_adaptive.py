"""Adaptive filtered search (plan="adaptive", research extension B)."""
import numpy as np
import pytest

from bench.datasets import attr_correlated, ground_truth, recall_at_k, synthetic
from micronn import MicroNN
from micronn.filters import eq, lt
from micronn.index import SearchStats

N, K = 6000, 20


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    base, q, _ = synthetic(N, dim=16, n_clusters=60, n_query=15, seed=5)
    rng = np.random.default_rng(1)
    attrs = dict(
        u=rng.integers(0, 100_000, N),                         # uniform: range filters set selectivity
        corr=attr_correlated(base, 30, noise=0.05, seed=2),    # tied to vector location
    )
    path = str(tmp_path_factory.mktemp("a") / "db.sqlite")
    db = MicroNN.create(path, dim=16, attributes={c: "INTEGER" for c in attrs})
    db.add(np.arange(N), base, list(zip(*(attrs[c].tolist() for c in attrs))))
    db.build(n_partitions=60)
    db.analyze(calibrate_costs=False)
    yield db, base, q, attrs
    db.close()


def _mask(attrs, f):
    m = np.ones(N, bool)
    for p in f:
        m &= (attrs[p.col] == p.value) if p.op == "=" else (attrs[p.col] < p.value)
    return m


def _recall(db, base, q, attrs, f, plan, nprobe=4):
    gt = ground_truth(base, q, K, mask=_mask(attrs, f))
    found = []
    for x in q:
        v, _ = db.search(x, K, nprobe=nprobe, filter=f, plan=plan)
        found.append(np.pad(v, (0, K - len(v)), constant_values=-1))
    return recall_at_k(np.array(found), gt, K)


def test_results_satisfy_filter_and_are_sorted(setup):
    db, base, q, attrs = setup
    f = [lt("u", 20_000)]
    v, d = db.search(q[0], K, nprobe=4, filter=f, plan="adaptive")
    assert len(v) == K
    assert np.all(attrs["u"][v] < 20_000)
    assert np.all(np.diff(d) >= 0)


def test_broad_filter_scans_like_post(setup):
    """A filter matching most rows (local selectivity ~0.9) reaches k matches in the first
    partitions, so adaptive scans about nprobe / 0.9 partitions: barely more than post-filter."""
    db, _, q, _ = setup
    st = SearchStats()
    db.search(q[0], K, nprobe=4, filter=[lt("u", 90_000)], plan="adaptive", stats=st)
    assert 4 <= st.partitions_probed <= 5 and not st.adaptive_fallback


def test_vector_correlated_filter_beats_post_recall(setup):
    """Matches of a vector-correlated filter sit far from most queries: post-filter with a
    small nprobe misses them, adaptive keeps scanning until it finds them."""
    db, base, q, attrs = setup
    counts = np.bincount(attrs["corr"])
    f = [eq("corr", int(np.argsort(counts)[len(counts) // 2]))]
    r_post = _recall(db, base, q, attrs, f, "post")
    r_adapt = _recall(db, base, q, attrs, f, "adaptive")
    assert r_adapt >= 0.9
    assert r_adapt > r_post


def test_selective_filter_falls_back_to_exact_pre(setup):
    db, base, q, attrs = setup
    f = [lt("u", 200)]                  # ~0.2% of rows: scanning would cost more than pre
    st = SearchStats()
    v, _ = db.search(q[0], K, nprobe=4, filter=f, plan="adaptive", stats=st)
    v_pre, _ = db.search(q[0], K, nprobe=4, filter=f, plan="pre")
    assert st.adaptive_fallback
    assert np.array_equal(np.sort(v), np.sort(v_pre))


def test_sees_delta_store(setup, tmp_path):
    db, base, q, attrs = setup
    new_vid = N + 1
    db.add([new_vid], q[1][None, :].astype(np.float32), [(5, 0)])
    try:
        v, _ = db.search(q[1], K, nprobe=4, filter=[lt("u", 50_000)], plan="adaptive")
        assert v[0] == new_vid
    finally:
        db.delete([new_vid])
