import numpy as np
import pytest

from bench.datasets import ground_truth, recall_at_k, synthetic
from micronn import MicroNN
from micronn.filters import between, eq, isin, lt, matches, to_sql
from micronn.optimizer import Estimator

N = 6000


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    base, q, _ = synthetic(N, dim=16, n_clusters=60, n_query=15, seed=3)
    rng = np.random.default_rng(0)
    attrs = dict(
        u=rng.integers(0, 100_000, N),            # uniform, high cardinality -> histogram
        cat=rng.integers(0, 20, N),               # low cardinality -> exact frequencies
        a=rng.integers(0, 10, N),
    )
    attrs["b"] = np.where(rng.random(N) < 0.9, attrs["a"], rng.integers(0, 10, N))  # b ≈ a
    path = str(tmp_path_factory.mktemp("f") / "db.sqlite")
    db = MicroNN.create(path, dim=16, attributes={c: "INTEGER" for c in attrs})
    rows = list(zip(*(attrs[c].tolist() for c in attrs)))
    db.add(np.arange(N), base, rows)
    db.build(n_partitions=60)
    db.analyze()
    yield db, base, q, attrs
    db.close()


def mask_of(f, attrs):
    m = np.ones(N, bool)
    for p in f:
        col = attrs[p.col]
        m &= np.array([matches([p], {p.col: x}) for x in col.tolist()])
    return m


def test_to_sql_rejects_unknown_columns_and_binds_values():
    sql, params = to_sql([eq("city", "x'); DROP TABLE vectors;--")], "a", ["city"])
    assert sql == "a.city = ?" and params == ["x'); DROP TABLE vectors;--"]
    with pytest.raises(ValueError):
        to_sql([eq("vec", 1)], "a", ["city"])


@pytest.mark.parametrize("f", [
    [lt("u", 1000)], [between("u", 20_000, 60_000)], [eq("cat", 3)], [isin("cat", [1, 2, 3])]])
def test_single_column_estimates_are_close(setup, f):
    db, _, _, attrs = setup
    true = mask_of(f, attrs).mean()
    est = db.optimizer.est.filter_selectivity(f)
    assert abs(est - true) < 0.02 + 0.2 * true


def test_independence_underestimates_correlated_conjunction(setup):
    db, _, _, attrs = setup
    f = [eq("a", 4), eq("b", 4)]
    true = mask_of(f, attrs).mean()
    est = db.optimizer.est.filter_selectivity(f)
    assert est < true / 5  # ~0.01 vs ~0.09: the gap research extension A targets


def test_stats_roundtrip(setup):
    db, *_ = setup
    e = Estimator.from_json(db.optimizer.est.to_json())
    assert e.filter_selectivity([eq("cat", 3)]) == db.optimizer.est.filter_selectivity([eq("cat", 3)])


@pytest.mark.parametrize("f", [[lt("u", 800)], [eq("cat", 7)], [eq("a", 1), eq("b", 1)]])
def test_prefilter_is_exact(setup, f):
    db, base, q, attrs = setup
    gt = ground_truth(base, q, 10, mask=mask_of(f, attrs))
    found = [db.search(x, 10, filter=f, plan="pre")[0] for x in q]
    assert all(np.all(mask_of(f, attrs)[v]) for v in found)
    assert recall_at_k(np.stack([np.pad(v, (0, 10 - len(v)), constant_values=-1) for v in found]), gt, 10) == 1.0


def test_postfilter_returns_only_matches_and_loses_recall_when_selective(setup):
    db, base, q, attrs = setup
    f = [lt("u", 800)]  # ~0.8% -> ~1 match per partition
    m = mask_of(f, attrs)
    found = [db.search(x, 10, nprobe=2, filter=f, plan="post")[0] for x in q]
    assert all(np.all(m[v]) for v in found)
    assert np.mean([len(v) for v in found]) < 10


def test_post_all_partitions_equals_pre(setup):
    db, _, q, _ = setup
    f = [eq("cat", 2)]
    for x in q[:5]:
        assert set(db.search(x, 10, nprobe=60, filter=f, plan="post")[0]) == set(db.search(x, 10, filter=f, plan="pre")[0])


def test_auto_plan_choice(setup):
    db, _, q, _ = setup
    from micronn.index import SearchStats
    s1, s2 = SearchStats(), SearchStats()
    db.search(q[0], 10, nprobe=4, filter=[lt("u", 300)], stats=s1)
    db.search(q[0], 10, nprobe=4, filter=[lt("u", 90_000)], stats=s2)
    assert s1.plan.name == "pre" and s2.plan.name == "post"


def test_query_plans_use_intended_access_paths(setup):
    db, *_ = setup
    c = db.store.conn
    pre = " ".join(r[-1] for r in c.execute(
        "EXPLAIN QUERY PLAN SELECT v.vid FROM attributes a JOIN vid_map m ON m.vid=a.vid "
        "JOIN vectors v ON v.partition_id=m.partition_id AND v.vid=m.vid WHERE a.u < 10"))
    post = [r[-1] for r in c.execute(
        "EXPLAIN QUERY PLAN SELECT v.vid FROM vectors v CROSS JOIN attributes a ON a.vid=v.vid "
        "WHERE v.partition_id=3 AND a.u < 10")]
    assert "idx_attr_u" in pre
    assert post[0].startswith("SEARCH v USING PRIMARY KEY (partition_id=?)")


def test_filter_sees_delta_rows(setup, tmp_path):
    db = MicroNN.create(str(tmp_path / "d.sqlite"), dim=4, attributes={"c": "INTEGER"})
    db.add(np.arange(200), np.random.rand(200, 4), [(i % 2,) for i in range(200)])
    db.build(n_partitions=4)
    db.add([1000], np.full((1, 4), 0.5), [(7,)])
    for plan in ("pre", "post"):
        assert db.search(np.zeros(4), 5, nprobe=1, filter=[eq("c", 7)], plan=plan)[0].tolist() == [1000]
    db.close()
