# MicroNN Re-implementation Plan

Goal: implement MicroNN (scaled down, Python + SQLite) and **check whether the paper's claims hold** in an independent re-implementation, plus one research extension.

## Claims to validate
Each claim gets its own experiment, a pass/fail criterion fixed in advance, and an honest verdict
(holds / holds partially / does not hold / cannot test). Check the exact numbers and datasets against the paper's PDF first.

| # | Paper claim | How we test it | "Holds" if |
|---|---|---|---|
| C1 | Top-100 at 90% recall in < 7 ms | Sweep `nprobe`, plot latency vs recall@100 on SIFT/GloVe | 90% recall reachable at low-ms latency (allow a constant factor for Python vs native code) |
| C2 | ~10 MB memory, ~100x less than in-memory | Peak RSS / tracemalloc of query process, MicroNN vs in-memory IVF | Memory stays bounded as N grows; ≥ 1 order of magnitude less than in-memory |
| C3 | Disk latency close to in-memory despite on-disk index | Same queries, both systems, cold and warm cache | Gap is small at the same recall |
| C4 | Mini-batch k-means builds with bounded memory; penalty balances clusters | Build memory vs N; partition size histogram with/without penalty | Build memory flat in N; lower size variance with penalty, similar recall |
| C5 | Pre-filter gives 100% recall, wins at low selectivity; post-filter wins at high selectivity | Latency & recall vs selectivity (0.1%–100%) for each plan | Crossover point exists; pre-filter recall = 1.0 |
| C6 | Optimizer picks the right plan | Compare optimizer's choice to the best plan in hindsight | Close to best-plan latency on independent filters |
| C7 | Batching cuts average latency ~33% at batch 1024 | Avg per-query latency vs batch size | Clear reduction; report our % next to theirs |
| C8 | Updates via delta store keep queries fast and correct | Insert stream; latency vs delta size; recall before/after merge; concurrent reads | New vectors visible immediately; latency recovers after merge |
| C9 | ACID / snapshot reads at no extra cost | Concurrent reader during writes/merge; crash mid-transaction | No torn or partial results |

Also look for limitations the paper does not mention (e.g. optimizer failing on correlated filters, slowdown with many centroids). These lead into the research extension.

### Threats to validity (write these up)
- Python vs the paper's native implementation: compare *trends and ratios*, not absolute ms.
- Different hardware (laptop vs Apple devices) and smaller datasets.
- Our attribute data is synthetic unless we use Big-ANN filtered data.
- Fix random seeds, warm up, report median and p95 over ≥ 3 runs.

## 0. Project setup
- Python 3.11+, NumPy, SQLite (stdlib `sqlite3`, WAL mode), matplotlib, pytest, `psutil`/`tracemalloc` for memory.
- Layout:
  ```
  micronn/
    storage.py      # schema, connection, WAL, vector (de)serialization
    clustering.py   # mini-batch k-means with balance penalty
    index.py        # build, search, delta store, merge/rebuild
    filters.py      # attribute predicates -> SQL
    optimizer.py    # selectivity estimation, pre vs post filter choice
    batch.py        # batched query execution
  bench/            # datasets, ground truth, experiment scripts, plots
  tests/
  ```

## 1. Storage layer (SQLite schema)
- `vectors(partition_id INT, vid INT, vec BLOB, PRIMARY KEY(partition_id, vid)) WITHOUT ROWID`
  — clustered on partition so one cluster = contiguous pages.
- `centroids(partition_id INT PRIMARY KEY, vec BLOB, size INT)`
- `attributes(vid INT PRIMARY KEY, <typed columns e.g. city, date, tag>)` + secondary indexes on filter columns.
- `delta(vid INT PRIMARY KEY, vec BLOB)` — partition id = -1 is an alternative.
- `vid_map(vid INT PRIMARY KEY, partition_id INT)` to locate/delete vectors.
- `meta(key, value)` — dim, metric, n_partitions, rebuild thresholds.
- Vectors stored as float32 `bytes` via `np.frombuffer`.
- PRAGMAs: `journal_mode=WAL`, `cache_size` small (to cap memory), `mmap_size=0`.

## 2. Index construction
- Mini-batch k-means (Sculley-style) streaming batches from SQLite; never load the full dataset.
- Balance penalty: add λ·(cluster_size) term to assignment cost so partitions stay similar in size.
- k ≈ √N (tunable); target partition size e.g. ~1000 vectors.
- Final assignment pass writes `vectors` in partition order inside one transaction.

## 3. ANN search (unfiltered)
- Load centroids (small) → compute distances → pick top `nprobe` partitions.
- `SELECT vid, vec FROM vectors WHERE partition_id IN (...)` → NumPy distance → top-k heap.
- Always also scan `delta`.
- Knob: `nprobe` controls recall/latency trade-off.

## 4. Hybrid (filtered) queries
- Predicates over `attributes` (equality, range, IN, AND/OR); optional FTS5 for text.
- **Pre-filter plan**: run filter via attribute index → fetch those vectors → brute-force (exact, 100% recall).
- **Post-filter plan**: IVF search with enlarged k / nprobe → join with attributes → drop non-matches.
- Optimizer: estimate selectivity (per-column histograms/counts, independence assumption for conjunctions as in paper) → cost model (rows scanned) → choose plan by threshold.

## 5. Batch queries
- For a batch of Q queries, map each to its partitions; group queries by partition.
- Scan each needed partition once; compute `Q_p × V_pᵀ` as one matrix multiply; merge per-query top-k.
- Measure average latency vs batch size (1…1024); paper reports ~33% reduction.

## 6. Updates
- Insert → `delta` + `attributes` in a transaction. Delete → remove from `vectors`/`delta`/`attributes`.
- Incremental merge: when delta exceeds threshold, assign delta vectors to nearest centroid, update centroid sizes.
- Full rebuild trigger: a partition grows beyond α × average size (or total drift threshold).
- Concurrency: readers use WAL snapshots while writer inserts/merges; test consistency.

## 7. Baselines
- In-memory IVF (same algorithm, all in NumPy arrays).
- Brute-force exact search (also used to produce ground truth).
- Optional: FAISS IVFFlat / HNSW for reference.

## 8. Datasets
- SIFT1M (128-d) and GloVe-100; start with 100K subsets for fast iteration.
- Synthetic attributes: uniform, skewed (Zipf), and **correlated with vector clusters** (needed for the research part).
- Optional: Big-ANN filtered track (YFCC-10M subset).
- Precompute ground truth top-100 (unfiltered and per filter).

## 9. Experiments (reproduce paper)
1. Latency vs recall@100 sweeping `nprobe` (target: ~90% recall, low ms).
2. Peak memory: MicroNN vs in-memory baseline (cap with small cache_size / `resource.setrlimit`).
3. Hybrid queries: latency & recall vs filter selectivity for pre, post, and optimizer-chosen plans.
4. Batch size vs average latency.
5. Updates: insert throughput, query latency as delta grows, effect of merge/rebuild.
6. Build time & memory of mini-batch k-means vs full k-means; partition size distribution with/without balance penalty.

## 10. Research extension (pick one; recommended first)
- **A (recommended): better selectivity estimation.** Replace independence assumption with
  multi-column histograms / sampling-based estimation / correlation with vector partitions.
  Evaluate on correlated filters: estimation error (q-error), wrong plan choices, end-to-end latency & recall.
- B: Index the centroid table (e.g. small HNSW or hierarchical centroids) for very large k.
- C: Delta merge policies (size threshold vs time vs query-cost-based) under insert-heavy workloads.

## 11. Testing
- Unit tests: serialization, k-means convergence, search == brute force when nprobe = all, filter correctness, delta visibility, transactional rollback.

## 12. Milestones
| Week | Deliverable |
|---|---|
| 1 | Setup, schema, data loading, ground truth |
| 2 | Mini-batch k-means + IVF build + unfiltered search, exp 1–2 |
| 3 | Attributes, pre/post filtering, optimizer, exp 3 |
| 4 | Batch queries + updates/delta/merge, exp 4–5 |
| 5 | Research extension + experiments |
| 6 | Claim-by-claim verdict table, plots, write-up, slides |
