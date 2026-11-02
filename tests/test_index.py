import numpy as np
import pytest

from bench.datasets import ground_truth, recall_at_k, synthetic
from micronn import BruteForce, InMemoryIVF, MicroNN
from micronn.clustering import minibatch_kmeans


@pytest.fixture(scope="module")
def data():
    base, q, _ = synthetic(5000, dim=16, n_clusters=40, n_query=20, seed=1)
    return base, q, ground_truth(base, q, k=10)


@pytest.fixture
def db(tmp_path, data):
    base, _, _ = data
    d = MicroNN.create(str(tmp_path / "db.sqlite"), dim=16)
    d.add(np.arange(len(base)), base)
    yield d
    d.close()


def test_minibatch_kmeans_finds_separated_clusters():
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [100, 0], [0, 100]], np.float32)
    x = (centers[rng.integers(0, 3, 3000)] + rng.normal(0, 1, (3000, 2))).astype(np.float32)
    c, n = minibatch_kmeans(lambda: iter([(None, x[i:i + 500]) for i in range(0, 3000, 500)]), k=3, seed=0)
    assert n == 3000
    assert np.abs(np.sort(c[:, 0]) - [0, 0, 100]).max() < 2


def test_build_moves_everything_out_of_delta(db, data):
    stats = db.build(n_partitions=50)
    assert db.store.count() == (len(data[0]), 0)
    assert stats.sizes.sum() == len(data[0])
    assert db.n_partitions == 50


def test_all_partitions_probed_equals_exact(db, data):
    base, q, gt = data
    db.build(n_partitions=30)
    found = np.stack([db.search(x, k=10, nprobe=30)[0] for x in q])
    assert recall_at_k(found, gt, 10) == 1.0


def test_recall_increases_with_nprobe(db, data):
    _, q, gt = data
    db.build(n_partitions=50)
    r = [recall_at_k(np.stack([db.search(x, 10, p)[0] for x in q]), gt, 10) for p in (1, 4, 16)]
    assert r[0] <= r[1] <= r[2] and r[2] > 0.9


def test_delta_vectors_are_searchable_before_rebuild(db, data):
    db.build(n_partitions=30)
    new = np.full((1, 16), 1e3, np.float32)
    db.add([999_999], new)
    assert db.search(new[0], k=1, nprobe=1)[0][0] == 999_999


def test_balance_penalty_reduces_size_spread(tmp_path):
    base, _, _ = synthetic(6000, dim=8, n_clusters=5, seed=2)  # few natural clusters -> skewed partitions
    spreads = []
    for lam in (0.0, 2.0):
        d = MicroNN.create(str(tmp_path / f"b{lam}.sqlite"), dim=8)
        d.add(np.arange(len(base)), base)
        spreads.append(d.build(n_partitions=40, balance=lam).sizes.std())
        d.close()
    assert spreads[1] < spreads[0]


def test_in_memory_baseline_matches_micronn(db, data):
    _, q, _ = data
    db.build(n_partitions=30)
    mem = InMemoryIVF(db)
    for x in q[:5]:
        assert set(db.search(x, 10, 4)[0]) == set(mem.search(x, 10, 4)[0])


def test_brute_force(data):
    base, q, gt = data
    bf = BruteForce(np.arange(len(base)), base)
    assert recall_at_k(np.stack([bf.search(x, 10)[0] for x in q]), gt, 10) == 1.0
