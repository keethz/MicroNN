# Results log

## Milestone 5 — research extension A: correlation-aware selectivity estimation (SIFT1M)

Run: `.venv/bin/python -m bench.exp_research --dataset sift --n 1000000` then
`.venv/bin/python -m bench.analyze_research results/research_sift_1000000.json`
(log `sift_research_run.log`; full tables `research_sift_1000000_report.md`; plots `research_sift_1000000_{qerror,tradeoff}.png`).

**Question.** MicroNN's optimizer assumes (1) filter columns are independent and (2) matches are spread evenly over
partitions. How wrong are the estimates on correlated filters, and does fixing them improve plan choice?

**Setup.** 35 filters in 4 categories (independent; correlated column pairs with 5–100 values, b = a 90% of the
time; anti-correlated pairs; filters correlated with vector location), 30 queries each, nprobe=32, k=100.
Each (filter, query) is executed once with pre and once with post; every estimator is scored on the same
measurements by the plan it picks. The plan rule is the milestone-3 optimizer: cost-based, plus a recall guard
"use pre unless the probed partitions are expected to hold ≥ g·k matches", swept over g = 1…16.

| estimator | build | memory | q-error correlated (median / max) | anti-correlated | vector-correlated |
|---|---|---|---|---|---|
| independence (paper) | 1 s | 27 KiB | 17.9 / 89.9 | 9.8 / 11.7 | 1.0 / 1.0 |
| column-group (joint stats for column pairs) | 59 s | 6.5 MiB | 1.0 / 1.0 | 1.0 / 1.1 | 1.0 / 1.0 |
| uniform sample (32K rows) | 1 s | 3.9 MiB | 1.0 / 1.2 | 1.4 / 7.4 | 1.0 / 1.1 |
| stratified sample (32 rows × 1000 partitions, ours) | 21 s | 4.1 MiB | 1.0 / 1.2 | 1.2 / 7.3 | 1.1 / 1.4 |

**Findings.**
1. *The independence assumption is badly wrong on correlated filters*: a median 18x (max 90x) underestimate for
   correlated pairs and ~10x overestimate for anti-correlated ones. All three alternatives fix this. Column-group stats
   are exact but cost the most to build and only cover column pairs; samples are generic but weaker on very rare
   conjunctions (max 7x).
2. *Better estimates make the optimizer faster at the same recall*, but less than the q-errors suggest. Over all
   filters, to get ≥ 91% of queries to recall ≥ 0.9, independence needs 78 ms per query on average (g=8), while
   any correlation-aware estimator needs 61 ms (g=16): **~22% faster**. At ≥ 94–95%, it is 113 ms vs 80 ms (~30%).
   On the correlated + anti-correlated filters alone, independence needs ~2x the latency for the same quality.
3. *Independence errors partly "help" recall*: underestimating a correlated filter pushes it to pre-filter (exact but
   slow). That is why independence has the best recall at g=1 on correlated filters (0.97) and the worst latency.
4. *Selectivity alone does not decide the plan.* Even with **exact** selectivity (oracle-global), plan accuracy
   against the per-query best plan is at most 83%, and with the exact number of matches in the probed partitions
   (oracle-local) at most 87%. The weak link is the rule "≥ g·k matches ⇒ post-filter recall is fine", not the
   estimate. Post-filter recall depends on whether the *filtered nearest neighbours* sit in the probed partitions,
   which match counts only loosely predict.
5. *Per-partition (stratified) estimation adds intermediate operating points for vector-correlated filters but does
   not dominate.* Global estimators jump from 83% of queries on target at 27 ms straight to 100% at 96 ms
   (all pre-filter, checked offline with g up to 128); stratified reaches 90% at 47 ms, in between. It matches its
   own oracle (oracle-local: 91% at 49 ms) almost exactly, so the estimator is accurate; the decision rule is the limit.

**Conclusion for the report.** The paper's independence assumption is a real weakness: correlated filters break its
estimates by up to two orders of magnitude, and fixing that buys ~20–30% latency at equal recall (up to 2x on
correlated workloads) for ~4–6 MiB of statistics. But a better estimator is not enough: choosing between pre- and
post-filtering needs a model of *post-filter recall*, not just selectivity. That is the most promising follow-up
(e.g. learn a recall predictor from per-partition match distribution relative to the query, or adaptively extend
nprobe until k matches are found).

Caveats: synthetic attributes on real vectors; one dataset; cost model calibrated before the warm-up fix (point/scan
ratio 14 instead of ≈4), which shifts the cost-based part of every estimator equally; stats are built once and not
maintained under inserts.

## Milestone 4 — batch queries, updates, transactions (SIFT1M)

Runs: `bench.exp_batch --dataset sift --n 1000000` (2048 queries, 3 runs; log `sift_batch_run.log`, plot
`batch_sift_1000000_c7.png`) and `bench.exp_updates --dataset sift --n 1000000` (log `sift_updates_run.log`,
plot `updates_sift_1000000_c8.png`).

### C7 — batching
| nprobe | one at a time | batch 16 | batch 64 | batch 256 | batch 1024 | partition scans/query at 1024 |
|---|---|---|---|---|---|---|
| 16 (recall 0.87) | 7.19 ms | 5.91 (−18%) | 3.88 (−46%) | 1.65 (−77%) | 0.58 (−92%) | 1.0 (vs 16) |
| 32 (recall 0.95) | 14.31 ms | 10.04 (−30%) | 5.21 (−64%) | 1.85 (−87%) | 0.73 (−95%) | 1.0 (vs 32) |

Recall is identical to sequential search (tested: batch results == per-query results).
**Verdict: holds, and the effect is much larger than the paper's ~33%.** At batch 1024 with 1000 partitions,
every partition is read about once for the whole batch. In our Python build the per-row fetch cost dominates, so
sharing scans removes most of the work; in a native, compute-bound implementation the shared part is a
smaller share of the total, which plausibly explains the paper's smaller number.

### C8 — updates (seed index 500K, insert 500K in 20K batches, nprobe=16, 30 queries)
Before inserts: 5.9 ms, recall 0.885.

| policy | latency after +500K | recall after +500K | inserts/s incl. maintenance |
|---|---|---|---|
| never merge (delta grows) | 237 ms (grows linearly with delta) | 0.951 | 43K |
| merge at 50K delta | 10–28 ms (saw-tooth with delta size) | 0.905 | 32K (random) / 35K (skewed) |
| merge + rebuild (largest partition > 1.5× built max) | 9–27 ms | 0.875–0.881 | 18K / 13K |

- New vectors are searchable immediately (they sit in the delta, which every query scans exactly).
- Without merging, latency grows linearly: the delta is brute-forced on every query (each 40K delta rows ≈ +20 ms
  here). Merging is required, and the delta threshold directly trades insert throughput for query latency.
- Incremental merge kept recall stable (0.89–0.91) even for the skewed insert order, so we never needed the rebuild
  to protect recall on SIFT. Rebuilds (1 random, 2 skewed) cut throughput roughly in half.
- Caveat: with a rebuild, k grows (707 → 894/959) while nprobe stays 16, so each query scans a smaller fraction of
  the data, hence lower recall. Fair comparison would fix the scanned fraction (nprobe ∝ k). Not done yet.
- **Verdict: holds**, with the condition that the delta store must be merged regularly.

### C9 — transactions and snapshot reads (tests in `tests/test_updates.py`)
- Merge is one transaction. A crash injected mid-merge (after rows were copied into `vectors`, before the delta was
  cleared) leaves the database exactly as before (`PRAGMA integrity_check` = ok, same vectors, no duplicates).
- A reader that holds a snapshot keeps seeing the pre-merge state while another connection commits a merge.
- Stress test: 2 reader threads + 1 writer doing 300 inserts and 150 merges; readers never see a duplicate or
  missing vector.
- **Bug found while testing:** the first `search` ran one SQL statement per partition plus one for the delta, each in
  its own implicit read transaction. A merge committing between them could make a vector appear twice or not at all.
  Fixed by wrapping every search in one read transaction (`MicroNN.snapshot()`), plus an `epoch` counter so a
  reader reloads cached centroids after another connection rebuilds. With the snapshot disabled, the stress test
  fails 3/3 runs; with it, it passes 3/3.
- **Verdict: holds**, but the guarantee covers one search only if the whole search runs in a single transaction.
  "ACID for free" from SQLite still needs this from the library.

## Milestone 3 — hybrid queries on SIFT1M (nprobe=32, 50 queries, k=100)

Run: `.venv/bin/python -m bench.exp_filter --dataset sift --n 1000000 --n-query 50`
(log: `sift_filter_run.log`, plot: `filter_sift_1000000_c5.png`, full table: `filter_sift_1000000_table.md`).
Synthetic attributes on the SIFT vectors: `u` uniform (range filters set selectivity), `cat` Zipf,
`corr` tied to vector location, `a`/`b` correlated pair (b = a 90% of the time), `u2` independent.

Design decisions to state in the report (check against the paper):
- Post-filter evaluates the filter *during* the partition scan (`vectors CROSS JOIN attributes`), not
  top-k' then filter. This is the more favourable post-filter; recall loss would be larger with top-k'.
- Optimizer = per-column stats (exact frequencies ≤1000 distinct values, else 100-bucket equi-depth
  histogram) + independence + calibrated per-row cost + a recall guard (use pre if the probed
  partitions are expected to hold < k matches).

| Claim | SIFT1M result | Verdict |
|---|---|---|
| C5 pre-filter = 100% recall, wins when selective; post wins when broad | Pre recall 1.00 at every selectivity. Latency crossover ≈ 1% selectivity (pre 44.6 ms vs post 40.4 ms). Post recall collapses for selective filters: 0.04 at 0.01%, 0.31 at 0.1%, 0.73 at 1%; reaches 0.90 only at 10% | **Holds** |
| C6 optimizer picks the right plan | Selectivity estimates are near-exact on single columns and independent conjunctions (q-error ≤ 1.2). Auto is always within ~10% of the *fastest* plan. But in the 1%–10% band it picks post, giving recall 0.73–0.90 where pre would give 1.00 at 1.1–14x the latency | **Holds partially**: it optimizes latency, not recall; the recall guard ("≥ k expected matches") is too optimistic |

Failure modes found (inputs for the research extension):
1. **Correlated predicates** — `a = 7 AND b = 7`: true 0.90%, estimated 0.010% (q-error 90). Plan was
   still right here (both sides of the estimate pick pre), but a cost misestimate of 90x is enough to flip
   plans for filters near the crossover.
2. **Filter correlated with vector location** — `corr = v` (1.0%): estimate exact, auto picks post, recall
   0.64. Independence between *filter and query vector* is a second, unmodelled assumption: matches are
   concentrated far from most queries, so probed partitions hold few of them.
3. **Cost calibration ran on a cold cache**: calibrated per-row cost was point 132 µs / scan 9.5 µs
   (ratio 14), but warm queries show ≈4.3 µs / ≈1 µs (ratio ≈4). Warm up before calibrating.

## Milestone 2b — SIFT1M (1,000,000 × 128), 200 queries, 3 runs

Run: `.venv/bin/python -m bench.exp_search --dataset sift --n 1000000 --n-query 200 --scaling 125000,250000,500000,1000000`
(log: `sift_run.log`, plots: `search_sift_1000000_{c1_c3,c2,c4}.png`). k=√N=1000 partitions.
Our exact ground truth matches the official `sift_groundtruth.ivecs` (recall@100 = 1.000 on 100 queries).

| nprobe | recall@100 | MicroNN median / p95 | In-memory median |
|---|---|---|---|
| 8 | 0.775 | 3.5 / 6.0 ms | 0.45 ms |
| 16 | 0.890 | 7.1 / 11.1 ms | 0.94 ms |
| 32 | 0.961 | 14.5 / 20.0 ms | 1.8 ms |
| 64 | 0.991 | 28.2 / 35.7 ms | 3.8 ms |

| Claim | SIFT1M result | Verdict |
|---|---|---|
| C1 <7 ms at 90% recall | 89% recall at 7.1 ms; 96% at 14.5 ms — even with Python overhead | **Holds approximately** (≈7–8 ms at 90%) |
| C2 ~10 MB, ~100x less | N=1M at nprobe=32: MicroNN 6.4 MiB (tracemalloc) / 10.8 MiB RSS; in-memory 551 / 655 MiB → ~86x / ~61x | **Holds** (~10 MB matches the paper almost exactly) |
| C3 disk ≈ in-memory | MicroNN ~7.5x slower at equal recall | **Does not hold in Python**; most of the gap is driver overhead (see milestone 2 attribution) |
| C4 balanced partitions, bounded build | λ=1: size std 491→101, max 4387→1438. Build peak 93.5 MiB, set by init sample (20·k) not N | Balance **holds**; on SIFT the penalty gives *lower* recall per ms at nprobe ≥ 32 and only helps p95 at small nprobe |

Notes:
- MicroNN memory grows slowly with N (2.4→6.4 MiB) because partition size is √N and peak memory ≈ one partition. Expected for IVF with k=√N, not a leak.
- Balanced runs show higher p95 at large nprobe (e.g. 147 ms at nprobe=128); likely noise from thermal throttling/background load — rerun before reporting.

## Milestone 2 — synthetic data, N=100K, dim=128 (preliminary)

Data: Gaussian mixture, 1000 overlapping clusters (spread=0.7). k=√N=316 partitions.
Run: `.venv/bin/python -m bench.exp_search --n 100000 --n-query 200 --scaling 25000,50000,100000,200000,400000`
Plots: `search_synthetic_100000_{c1_c3,c2,c4}.png`

| Claim | Our result | Preliminary verdict |
|---|---|---|
| C1 <7 ms at 90% recall@100 | 90% needs nprobe≈128 → **27 ms** (MicroNN), 3.4 ms (in-memory) | Not met in our Python build; re-test on SIFT |
| C2 ~10 MB, ~100x less than in-memory | MicroNN query peak 1.0→2.6 MiB (tracemalloc), 3.7→5.9 MiB RSS for N=25K→400K; in-memory 35→337 MiB | **Holds** (≈130x at 400K) |
| C3 disk close to in-memory | MicroNN ~8x slower at same recall | Does not hold *here*; see attribution below |
| C4 bounded build memory, balanced partitions | Build peak ~41 MiB (set by init sample of 20·k, not N); penalty λ=1: size std 247→4 | Balance **holds**; penalty costs recall at fixed nprobe (0.955→0.875 at nprobe=128) |

### Latency attribution (nprobe=128, ~40K rows scanned, warm OS cache)
- SQLite B-tree scan alone: 8.9 ms
- Python `sqlite3` row materialization: +14.4 ms
- NumPy distance + top-k: ~3 ms

More than half of the disk-vs-memory gap is the Python driver. A native implementation would be roughly
9–12 ms on this data. Report this as a threat to validity for C1/C3.

### Bugs found during the run (for the write-up)
- Ground-truth cache was keyed by name only, so it served answers for an old dataset. It is now keyed by a hash of the data.
- First version of search held every probed vector in RAM at once (memory grew with N). Fixed by streaming
  one partition at a time into a running top-k — this is what makes C2 hold.

### Open caveats
- Synthetic data; SIFT1M/GloVe not run yet.
- OS page cache is warm (dropping it on macOS needs `sudo purge`); "cold" numbers not measured yet.
