"""SQLite storage layer for MicroNN.

Tables (see PLAN.md section 1):
  vectors    clustered by (partition_id, vid) so each IVF partition is contiguous on disk
  centroids  one row per partition
  delta      newly inserted vectors not yet merged into a partition
  vid_map    vid -> partition_id, to locate vectors for deletes
  attributes per-vector filterable columns
  meta       key/value config (dim, metric, ...)
"""
import re
import sqlite3
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

DTYPE = np.float32
DELTA_PARTITION = -1
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ATTR_TYPES = {"INTEGER", "REAL", "TEXT"}


def to_blob(vec: np.ndarray) -> bytes:
    return np.ascontiguousarray(vec, dtype=DTYPE).tobytes()


def from_blob(blob: bytes, dim: int) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=DTYPE)
    assert arr.shape[0] == dim, f"blob has {arr.shape[0]} floats, expected {dim}"
    return arr


def blobs_to_matrix(blobs: Sequence[bytes], dim: int) -> np.ndarray:
    """Stack many blobs into an (n, dim) matrix with one copy."""
    if not blobs:
        return np.empty((0, dim), dtype=DTYPE)
    return np.frombuffer(b"".join(blobs), dtype=DTYPE).reshape(len(blobs), dim)


class Store:
    """Thin wrapper over one SQLite database file holding a MicroNN index."""

    def __init__(self, path: str, cache_kib: int = 2048):
        self.path = path
        self.conn = sqlite3.connect(path, isolation_level=None)  # explicit transactions
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        # Negative cache_size is in KiB. Kept small to imitate a memory-constrained device.
        self.conn.execute(f"PRAGMA cache_size=-{int(cache_kib)}")
        self.conn.execute("PRAGMA mmap_size=0")
        self._dim: Optional[int] = None

    # ---------- schema ----------
    def create(self, dim: int, attributes: Optional[Dict[str, str]] = None,
               metric: str = "l2") -> None:
        attributes = attributes or {}
        cols = []
        for name, typ in attributes.items():
            if not _IDENT.match(name) or typ.upper() not in _ATTR_TYPES:
                raise ValueError(f"bad attribute {name!r} {typ!r}")
            cols.append(f"{name} {typ.upper()}")
        attr_cols = "".join(", " + c for c in cols)
        ddl = [
            "CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT) WITHOUT ROWID",
            """CREATE TABLE vectors(
                   partition_id INTEGER NOT NULL,
                   vid INTEGER NOT NULL,
                   vec BLOB NOT NULL,
                   PRIMARY KEY(partition_id, vid)
               ) WITHOUT ROWID""",
            """CREATE TABLE centroids(
                   partition_id INTEGER PRIMARY KEY,
                   vec BLOB NOT NULL,
                   size INTEGER NOT NULL DEFAULT 0
               )""",
            "CREATE TABLE delta(vid INTEGER PRIMARY KEY, vec BLOB NOT NULL)",
            "CREATE TABLE vid_map(vid INTEGER PRIMARY KEY, partition_id INTEGER NOT NULL)",
            f"CREATE TABLE attributes(vid INTEGER PRIMARY KEY{attr_cols})",
        ]
        # executescript() would implicitly COMMIT, so run DDL one statement at a time.
        with self.transaction():
            for stmt in ddl:
                self.conn.execute(stmt)
            for name in attributes:
                self.conn.execute(f"CREATE INDEX idx_attr_{name} ON attributes({name})")
            self.set_meta("dim", dim)
            self.set_meta("metric", metric)
        self._dim = dim

    # ---------- meta ----------
    def set_meta(self, key: str, value) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta VALUES(?, ?)", (key, str(value)))

    def get_meta(self, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.get_meta("dim"))
        return self._dim

    def add_attribute_column(self, name: str, typ: str) -> None:
        """Add a filterable column (with an index) to an existing database."""
        if not _IDENT.match(name) or typ.upper() not in _ATTR_TYPES:
            raise ValueError(f"bad attribute {name!r} {typ!r}")
        self.conn.execute(f"ALTER TABLE attributes ADD COLUMN {name} {typ.upper()}")
        self.conn.execute(f"CREATE INDEX idx_attr_{name} ON attributes({name})")

    def attribute_columns(self) -> List[str]:
        rows = self.conn.execute("PRAGMA table_info(attributes)").fetchall()
        return [r[1] for r in rows if r[1] != "vid"]

    # ---------- transactions ----------
    def transaction(self):
        return _Txn(self.conn)

    # ---------- writes ----------
    def insert_delta(self, vids: Sequence[int], vecs: np.ndarray,
                     attrs: Optional[Sequence[Sequence]] = None) -> None:
        """Insert new vectors into the delta store (caller wraps in a transaction)."""
        vecs = np.asarray(vecs, dtype=DTYPE)
        assert vecs.shape == (len(vids), self.dim)
        self.conn.executemany("INSERT INTO delta VALUES(?, ?)",
                              ((int(v), to_blob(x)) for v, x in zip(vids, vecs)))
        self.conn.executemany("INSERT INTO vid_map VALUES(?, ?)",
                              ((int(v), DELTA_PARTITION) for v in vids))
        if attrs is not None:
            self.insert_attributes(vids, attrs)

    def insert_attributes(self, vids: Sequence[int], attrs: Sequence[Sequence]) -> None:
        cols = self.attribute_columns()
        ph = ", ".join("?" * (len(cols) + 1))
        self.conn.executemany(f"INSERT OR REPLACE INTO attributes(vid, {', '.join(cols)}) VALUES({ph})",
                              ((int(v), *a) for v, a in zip(vids, attrs)))

    def write_partition(self, partition_id: int, vids: Sequence[int], vecs: np.ndarray) -> None:
        self.conn.executemany("INSERT OR REPLACE INTO vectors VALUES(?, ?, ?)",
                              ((partition_id, int(v), to_blob(x)) for v, x in zip(vids, vecs)))
        self.conn.executemany("INSERT OR REPLACE INTO vid_map VALUES(?, ?)",
                              ((int(v), partition_id) for v in vids))

    def write_centroids(self, centroids: np.ndarray, sizes: Sequence[int]) -> None:
        self.conn.execute("DELETE FROM centroids")
        self.conn.executemany("INSERT INTO centroids VALUES(?, ?, ?)",
                              ((i, to_blob(c), int(s)) for i, (c, s) in enumerate(zip(centroids, sizes))))

    # ---------- reads ----------
    def read_centroids(self) -> Tuple[np.ndarray, np.ndarray]:
        rows = self.conn.execute("SELECT partition_id, vec FROM centroids ORDER BY partition_id").fetchall()
        ids = np.array([r[0] for r in rows], dtype=np.int64)
        return ids, blobs_to_matrix([r[1] for r in rows], self.dim)

    def read_partitions(self, partition_ids: Iterable[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Read all vectors in the given partitions (a range scan per partition on the clustered key)."""
        vids, blobs = [], []
        for pid in partition_ids:
            for vid, blob in self.conn.execute(
                    "SELECT vid, vec FROM vectors WHERE partition_id=?", (int(pid),)):
                vids.append(vid)
                blobs.append(blob)
        return np.array(vids, dtype=np.int64), blobs_to_matrix(blobs, self.dim)

    def read_delta(self) -> Tuple[np.ndarray, np.ndarray]:
        rows = self.conn.execute("SELECT vid, vec FROM delta").fetchall()
        return np.array([r[0] for r in rows], dtype=np.int64), blobs_to_matrix([r[1] for r in rows], self.dim)

    def iter_all_vectors(self, batch_size: int = 10_000) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Stream every vector (indexed + delta) in batches, without loading everything at once."""
        cur = self.conn.execute("SELECT vid, vec FROM vectors UNION ALL SELECT vid, vec FROM delta")
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                return
            yield (np.array([r[0] for r in rows], dtype=np.int64),
                   blobs_to_matrix([r[1] for r in rows], self.dim))

    def count(self) -> Tuple[int, int]:
        """(indexed vectors, delta vectors)."""
        n_idx = self.conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
        n_delta = self.conn.execute("SELECT COUNT(*) FROM delta").fetchone()[0]
        return n_idx, n_delta

    def close(self) -> None:
        self.conn.close()


class _Txn:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def __enter__(self):
        self.conn.execute("BEGIN IMMEDIATE")
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        self.conn.execute("ROLLBACK" if exc_type else "COMMIT")
        return False
