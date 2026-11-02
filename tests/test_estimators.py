import numpy as np
import pytest

from bench.datasets import attr_correlated, synthetic
from micronn import MicroNN
from micronn.estimators import ColumnGroupEstimator, SampleEstimator, StratifiedSampleEstimator
from micronn.filters import eq, lt
from micronn.optimizer import Estimator, Optimizer

N = 20_000


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    base, q, _ = synthetic(N, dim=16, n_clusters=100, n_query=10, seed=7, spread=2.0)
    rng = np.random.default_rng(1)
    a = rng.integers(0, 10, N)
    attrs = dict(
        a=a,
        b=np.where(rng.random(N) < 0.9, a, rng.integers(0, 10, N)),  # b ≈ a
        u=rng.integers(0, 1_000_000, N),
        corr=attr_correlated(base, 20, noise=0.0, seed=3),           # tied to vector location
    )
    db = MicroNN.create(str(tmp_path_factory.mktemp("e") / "db.sqlite"), dim=16,
                        attributes={c: "INTEGER" for c in attrs})
    db.add(np.arange(N), base, list(zip(*(attrs[c].tolist() for c in attrs))))
    db.build(n_partitions=100)
    conn, cols = db.store.conn, db.store.attribute_columns()
    ests = dict(
        indep=Estimator.analyze(conn, cols),
        group=ColumnGroupEstimator.analyze(conn, cols),
        sample=SampleEstimator.analyze(conn, cols, size=3000),
        strat=StratifiedSampleEstimator.analyze(conn, cols, per_partition=30),
    )
    yield db, q, attrs, ests
    db.close()


def true_sel(f, attrs):
    m = np.ones(N, bool)
    for p in f:
        m &= (attrs[p.col] == p.value) if p.op == "=" else (attrs[p.col] < p.value)
    return m.mean(), m


def qerr(est, true, n=N):
    est, true = max(est, 1 / n), max(true, 1 / n)
    return max(est / true, true / est)


def test_correlated_pair(setup):
    _, _, attrs, ests = setup
    f = [eq("a", 3), eq("b", 3)]  # true ≈ 0.09, independence ≈ 0.01
    t, _ = true_sel(f, attrs)
    assert qerr(ests["indep"].filter_selectivity(f), t) > 5
    for name in ("group", "sample", "strat"):
        assert qerr(ests[name].filter_selectivity(f), t) < 1.3, name


def test_anti_correlated_pair_is_capped(setup):
    _, _, attrs, ests = setup
    f = [eq("a", 3), eq("b", 4)]  # true ≈ 0.001, independence ≈ 0.01
    t, _ = true_sel(f, attrs)
    assert ests["indep"].filter_selectivity(f) > 5 * t
    for name in ("group", "sample", "strat"):
        assert ests[name].filter_selectivity(f) < 3 * t + 3 / 3000, name


def test_independent_predicates_not_worse(setup):
    _, _, attrs, ests = setup
    f = [lt("u", 200_000), eq("a", 5)]
    t, _ = true_sel(f, attrs)
    for name, e in ests.items():
        assert qerr(e.filter_selectivity(f), t) < 1.5, name


def test_expected_matches_in_probed_partitions(setup):
    db, q, attrs, ests = setup
    f = [eq("corr", int(np.bincount(attrs["corr"]).argmax()))]
    sel, m = true_sel(f, attrs)
    vid_part = dict(db.store.conn.execute("SELECT vid, partition_id FROM vid_map"))
    part_of = np.array([vid_part[i] for i in range(N)])
    err_uniform, err_strat = [], []
    for x in q:
        probed = db.probe(x, 10)
        truth = m[np.isin(part_of, probed)].sum()
        scanned = np.isin(part_of, probed).sum()
        err_uniform.append(abs(sel * scanned - truth))
        err_strat.append(abs(ests["strat"].expected_matches(f, probed) - truth))
    assert np.mean(err_strat) < 0.5 * np.mean(err_uniform)


def test_optimizer_uses_expected_matches(setup):
    db, q, attrs, ests = setup
    f = [eq("corr", int(np.bincount(attrs["corr"]).argmax()))]
    sel, _ = true_sel(f, attrs)
    nprobe = 5
    k = int(0.8 * sel * nprobe * db.avg_partition_size)  # spread-evenly estimate clears the guard
    plans = {name: [Optimizer(ests[name]).choose(f, k, nprobe, db.avg_partition_size,
                                                 probed=db.probe(x, nprobe)).name for x in q]
             for name in ("indep", "strat")}
    assert set(plans["indep"]) == {"post"}
    assert "pre" in plans["strat"]  # queries far from where the matches live get pre-filter
