import numpy as np
import pytest

from micronn.storage import Store, DELTA_PARTITION, from_blob, to_blob


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "db.sqlite"))
    s.create(dim=8, attributes={"city": "TEXT", "year": "INTEGER"})
    yield s
    s.close()


def test_blob_roundtrip():
    v = np.random.rand(8).astype(np.float32)
    assert np.array_equal(from_blob(to_blob(v), 8), v)


def test_wal_and_schema(store):
    assert store.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert store.dim == 8
    assert store.attribute_columns() == ["city", "year"]


def test_partition_write_and_read(store):
    rng = np.random.default_rng(0)
    a, b = rng.random((5, 8), dtype=np.float32), rng.random((3, 8), dtype=np.float32)
    with store.transaction():
        store.write_partition(0, range(5), a)
        store.write_partition(1, range(5, 8), b)
        store.write_centroids(np.stack([a.mean(0), b.mean(0)]), [5, 3])
    vids, vecs = store.read_partitions([1])
    assert sorted(vids.tolist()) == [5, 6, 7]
    assert np.allclose(vecs[np.argsort(vids)], b)
    ids, cents = store.read_centroids()
    assert ids.tolist() == [0, 1] and cents.shape == (2, 8)
    assert store.count() == (8, 0)


def test_delta_and_attributes(store):
    vecs = np.ones((2, 8), dtype=np.float32)
    with store.transaction():
        store.insert_delta([10, 11], vecs, attrs=[("Seattle", 2020), ("Paris", 2021)])
    vids, got = store.read_delta()
    assert sorted(vids.tolist()) == [10, 11]
    assert store.conn.execute("SELECT partition_id FROM vid_map WHERE vid=10").fetchone()[0] == DELTA_PARTITION
    assert store.conn.execute("SELECT vid FROM attributes WHERE city='Seattle'").fetchall() == [(10,)]


def test_rollback_on_error(store):
    with pytest.raises(RuntimeError):
        with store.transaction():
            store.insert_delta([1], np.zeros((1, 8), dtype=np.float32))
            raise RuntimeError("boom")
    assert store.count() == (0, 0)


def test_snapshot_isolation(tmp_path, store):
    """A reader that started before a write keeps seeing the old snapshot (WAL)."""
    reader = Store(store.path)
    reader.conn.execute("BEGIN")
    assert reader.conn.execute("SELECT COUNT(*) FROM delta").fetchone()[0] == 0
    with store.transaction():
        store.insert_delta([1], np.zeros((1, 8), dtype=np.float32))
    assert reader.conn.execute("SELECT COUNT(*) FROM delta").fetchone()[0] == 0
    reader.conn.execute("COMMIT")
    assert reader.conn.execute("SELECT COUNT(*) FROM delta").fetchone()[0] == 1
    reader.close()


def test_iter_all_vectors(store):
    with store.transaction():
        store.write_partition(0, range(5), np.zeros((5, 8), dtype=np.float32))
        store.insert_delta([100, 101], np.ones((2, 8), dtype=np.float32))
    seen = np.concatenate([v for v, _ in store.iter_all_vectors(batch_size=3)])
    assert sorted(seen.tolist()) == [0, 1, 2, 3, 4, 100, 101]
