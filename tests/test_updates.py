import os
import subprocess
import sys
import threading

import numpy as np
import pytest

from bench.datasets import synthetic
from micronn import MicroNN
from micronn.batch import search_batch
from micronn.filters import eq
from micronn.index import MaintenancePolicy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIM = 16


@pytest.fixture
def db(tmp_path):
    base, _, _ = synthetic(3000, dim=DIM, n_clusters=30, seed=4)
    d = MicroNN.create(str(tmp_path / "db.sqlite"), dim=DIM, attributes={"c": "INTEGER"})
    d.add(np.arange(len(base)), base, [(i % 5,) for i in range(len(base))])
    d.build(n_partitions=30)
    yield d
    d.close()


def all_vids(d):
    return sorted(d.store.conn.execute("SELECT vid FROM vectors UNION ALL SELECT vid FROM delta").fetchall())


def test_batch_matches_single_queries(db):
    _, q, _ = synthetic(10, dim=DIM, n_query=40, seed=9)
    db.add([50_000, 50_001], q[:2] + 0.01)  # include delta rows
    batch = search_batch(db, q, k=20, nprobe=5)
    for x, (bv, bd) in zip(q, batch):
        v, d = db.search(x, 20, nprobe=5)
        assert np.array_equal(v, bv) and np.allclose(d, bd, rtol=1e-4, atol=1e-3)


def test_delete_indexed_and_delta(db):
    db.add([9000], np.zeros((1, DIM)), [(1,)])
    target = db.search(np.zeros(DIM), 1, nprobe=30)[0][0]
    assert target == 9000
    other = 17
    assert db.delete([9000, other, 123456]) == 2
    assert 9000 not in db.search(np.zeros(DIM), 50, nprobe=30)[0]
    assert (other,) not in all_vids(db)
    assert db.store.conn.execute("SELECT COUNT(*) FROM attributes WHERE vid IN (9000, 17)").fetchone()[0] == 0
    assert db.store.conn.execute("SELECT SUM(size) FROM centroids").fetchone()[0] == 2999


def test_merge_delta_preserves_results(db):
    new, _, _ = synthetic(500, dim=DIM, n_clusters=30, seed=5)
    db.add(np.arange(10_000, 10_500), new)
    q = new[:10] + 0.1
    before = [set(db.search(x, 10, nprobe=30)[0]) for x in q]
    epoch = db.store.get_meta("epoch")
    assert db.merge_delta() == 500
    assert db.store.count() == (3500, 0)
    assert db.store.get_meta("epoch") != epoch
    assert db.store.conn.execute("SELECT SUM(size) FROM centroids").fetchone()[0] == 3500
    assert [set(db.search(x, 10, nprobe=30)[0]) for x in q] == before
    # merged rows are found through their partition now
    assert db.search(new[0], 1, nprobe=1)[0][0] == 10_000


def test_maintain_policy(db):
    db.add(np.arange(20_000, 20_050), np.random.rand(50, DIM))
    assert db.maintain(MaintenancePolicy(delta_max=100, rebuild_factor=0)) == "none"
    assert db.maintain(MaintenancePolicy(delta_max=50, rebuild_factor=0)) == "merge"
    # pile many vectors on one spot -> one partition far above 3x the built average
    built_max = int(db.store.get_meta("build_max_size"))
    assert db.maintain(MaintenancePolicy(delta_max=1, rebuild_factor=1.5)) == "none"  # nothing grew
    n_pile = 2 * built_max
    db.add(np.arange(30_000, 30_000 + n_pile), np.full((n_pile, DIM), 5.0) + np.random.rand(n_pile, DIM) * 0.01)
    assert db.maintain(MaintenancePolicy(delta_max=1, rebuild_factor=1.5)) == "rebuild"
    assert db.store.count() == (3050 + n_pile, 0)


def test_reader_snapshot_is_stable_across_concurrent_merge(db):
    db.add([77_777], np.full((1, DIM), 3.0))
    reader = MicroNN.open(db.store.path)
    with reader.snapshot():
        first = reader.search(np.full(DIM, 3.0), 1, nprobe=30)[0].tolist()
        db.merge_delta()  # another connection commits a merge mid-snapshot
        again = reader.search(np.full(DIM, 3.0), 5, nprobe=30)[0].tolist()
        assert reader.store.count() == (3000, 1)  # still the pre-merge snapshot
    assert first == [77_777] and again.count(77_777) == 1
    assert reader.store.count() == (3001, 0)
    reader.close()


def test_reader_reloads_centroids_after_rebuild_elsewhere(db):
    reader = MicroNN.open(db.store.path)
    assert reader.n_partitions == 30
    db.build(n_partitions=12)
    v, _ = reader.search(np.zeros(DIM), 5, nprobe=12)
    assert reader.n_partitions == 12 and len(v) == 5
    reader.close()


def test_concurrent_readers_never_see_duplicates_or_gaps(db):
    path = db.store.path
    errors, stop = [], threading.Event()
    probe_vecs = np.full((1, DIM), 4.0)

    def writer():
        w = MicroNN.open(path)
        try:
            for i in range(300):
                w.add([60_000 + i], probe_vecs + i * 1e-3)
                if i % 2 == 1:
                    w.merge_delta()
        finally:
            w.close()
            stop.set()

    def reader():
        r = MicroNN.open(path)
        try:
            while not stop.is_set():
                with r.snapshot():
                    present = {v for (v,) in r.store.conn.execute(
                        "SELECT vid FROM vid_map WHERE vid >= 60000")}
                    found = r.search(probe_vecs[0], 400, nprobe=30)[0].tolist()
                hits = [v for v in found if v >= 60_000]
                if len(hits) != len(set(hits)) or set(hits) != present:
                    errors.append((sorted(hits), sorted(present)))
        except Exception as e:  # surface thread errors in the test
            errors.append(repr(e))
        finally:
            r.close()

    threads = [threading.Thread(target=reader) for _ in range(2)] + [threading.Thread(target=writer)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors[:3]


def test_crash_during_merge_leaves_db_consistent(db):
    db.add(np.arange(40_000, 40_100), np.random.rand(100, DIM))
    before = all_vids(db)
    script = f"""
import os, sys
sys.path.insert(0, {ROOT!r})
from micronn import MicroNN
d = MicroNN.open({db.store.path!r})
class Conn:
    def __init__(self, c): self.c = c
    def execute(self, sql, *a):
        if sql.startswith("DELETE FROM delta"):
            os._exit(1)  # die mid-transaction: rows already copied into vectors, nothing committed
        return self.c.execute(sql, *a)
    def __getattr__(self, n): return getattr(self.c, n)
d.store.conn = Conn(d.store.conn)
d.merge_delta()
"""
    rc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert rc.returncode == 1, rc.stderr
    fresh = MicroNN.open(db.store.path)
    assert fresh.store.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert all_vids(fresh) == before
    assert fresh.store.count() == (3000, 100)
    fresh.close()
