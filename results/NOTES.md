# Results log

## Milestone 7 — GloVe-200 (1,183,514 × 200), paper Table 2 dataset.

Data: ANN-Benchmarks `glove-200-angular.hdf5` (the GloVe row of the paper's Table 2: 200 dims, 1.18M vectors,
10K queries). Vectors are L2-normalised, so L2 ranking = angular ranking. Full base set; 200 queries for search,
50 for filters, 30 for extensions A/B, 512 for batching. k = √N = 1088 partitions, k=100.
Machine: Windows laptop (Intel i7-1355U, 16 GB), Python 3.13. **Latencies are not comparable to the SIFT1M
numbers (different, slower machine); compare trends and ratios, not milliseconds.**

Runs (logs `glove_*_run.log`):
```
python -m bench.exp_search   --dataset glove --n 1183514 --n-query 200 --runs 1
python -m bench.exp_filter   --dataset glove --n 1183514 --n-query 50
python -m bench.exp_adaptive --dataset glove --n 1183514
python -m bench.exp_research --dataset glove --n 1183514
python -m bench.exp_batch    --dataset glove --n 1183514 --n-query 512 --runs 1
python -m bench.exp_updates  --dataset glove --n 1000000      # 1M so 20K checkpoints line up (see below)
python -m bench.plot_updates results/batch_glove_1183514.json results/updates_glove_1000000.json
```
Plots/tables: `search_glove_1183514_{c1_c3,c2,c4}.png`, `filter_glove_1183514_{c5.png,table.md}`,
`research_glove_1183514_{report.md,qerror.png,tradeoff.png}`, `adaptive_glove_1183514_{report.md,tradeoff.png}`, `batch_glove_1183514_c7.png`, `updates_glove_1000000_c8.png`.

### Claim summary (GloVe vs SIFT1M)

| Claim | GloVe result | SIFT1M | Verdict on GloVe |
|---|---|---|---|
| C1 low latency at 90% recall | Recall 0.67 @ nprobe 16, 0.79 @ 64, **0.92 @ 256** (4.1 s median on this laptop) | 0.89 @ 16, 0.99 @ 64 | **Does not hold**: GloVe needs ~24% of partitions for 90% recall |
| C2 ~10 MB, ~100x less | MicroNN 8.7 MiB (tracemalloc) / 17.7 MiB RSS vs in-memory 1,575 / 1,597 MiB → **181x / 90x** | ~86x / ~61x | **Holds** (stronger, since vectors are bigger) |
| C3 disk ≈ in-memory | MicroNN 5–8x slower at equal recall (7.0x at nprobe 32) | ~7.5x | **Does not hold in Python**; same ratio as SIFT → driver overhead, not data |
| C4 balanced partitions | λ=1: size std 905 → 16, max 4,444 → 1,160. Without penalty **262 of 1,088 partitions hold ≤ 10 vectors**. Penalty lowers recall at fixed nprobe (0.725 → 0.679 @ 32) | std 491 → 101 | Balance **holds**; same recall cost as SIFT |
| C5 pre exact / post wins when broad | Pre recall 1.00 everywhere. Latency crossover ≈ 0.5% selectivity. Post recall 0.06 @ 0.01%, 0.36 @ 1%, **max 0.71 at 100%** | crossover ≈ 1%, post 0.90 @ 10% | **Holds**, but post is never good enough on GloVe |
| C6 optimizer picks right plan | Estimates exact on single columns (q-err ≤ 1.2); picks post for every filter ≥ 1% → recall 0.36–0.71 | same pattern, recall 0.73–0.90 | **Holds partially** (latency yes, recall no) — worse than SIFT |
| C7 batching | −92% per query at batch 1024 (nprobe 16 and 32); 1.5–1.6 partition scans/query | −92% / −95% | **Holds**, same size as SIFT |
| C8 updates | Never merge: 22 ms → 1,583 ms over +500K; merge: 53–199 ms, recall 0.65–0.69; rebuild raises recall to 0.70–0.73 | linear growth without merge; rebuild lowered recall | **Holds** with regular merges |

### Findings
1. **GloVe is a much harder dataset for IVF.** Unfiltered recall@100 at nprobe=32 is only 0.73 (SIFT: 0.96).
   Word vectors are not well clustered: without the balance penalty a quarter of partitions are almost empty
   (≤ 10 vectors) and the rest absorb the data. This is a property of the data, not of our build — in-memory IVF
   on the same partitions gets the identical recall.
2. **Memory claim gets stronger.** 200-dim vectors make the in-memory index ~1 GB (957 MB of vectors), while
   MicroNN still streams one partition at a time (8.7 MiB). Same mechanism as SIFT, bigger gap.
3. **Disk/in-memory gap is the same ~7x as on SIFT**, on a different OS and CPU. Supports the milestone-2
   attribution: the gap is the Python sqlite3 driver, not the dataset.
4. **Post-filter recall is capped by unfiltered recall.** On GloVe post-filter can never reach 0.9 at nprobe=32
   (best 0.71 with no filter at all), so pre-filter is the only plan that meets the target for every filter.
   The optimizer still picks post above ~0.5% selectivity because its recall guard only checks match counts.
5. **Extension A (estimators) — mostly the same as SIFT, one negative result.** Independence q-error: median 18x /
   max 91x on correlated pairs, 10x / 15x on anti-correlated (SIFT: 18x / 90x and 10x / 12x). Column-group and
   samples fix this (≈1.0 median), at 223 s / 6.5 MiB build for column-group. **But better estimates did not
   improve plan quality on GloVe**: at g=1, independence gets 60% of queries to recall ≥ 0.9 vs 52% for every
   correction *including the exact oracle*. Independence underestimates correlated filters, which accidentally
   pushes them to pre-filter (the only plan that works here). Confirms SIFT finding 4: the decision rule, not
   the estimate, is the weak link — and on hard data it matters even more.
6. **Extension B (adaptive) — works, smaller win than hoped.**

   | strategy (35 filters × 30 queries, nprobe=32) | mean ms | p95 ms | mean recall | queries ≥ 0.9 |
   |---|---|---|---|---|
   | post | 241 | 364 | 0.336 | 7% |
   | auto independence g=1 | 302 | 1,428 | 0.751 | 60% |
   | auto independence g=16 | 838 | 5,412 | 0.953 | 91% |
   | adaptive e=0.5 | 663 | 2,784 | 0.960 | 89% |
   | **adaptive e=1** | **1,175** | 5,815 | **0.991** | **97%** |
   | pre | 1,464 | 7,977 | 1.000 | 100% |

   - Adaptive e=1 gets 97% of queries on target, **20% faster than pre**; e=0.5 matches tuned auto (g=16)
     quality at ~20% lower latency.
   - Vector-correlated filters (the case auto gets wrong): adaptive recall 0.95–1.00 vs post 0.17–0.57.
   - But on GloVe adaptive **falls back to pre for almost every filter** (100% fallback on 31 of 35 filters),
     because the probed partitions rarely hold enough matches. The partitions scanned before falling back are
     wasted work, so on anti-correlated filters it is ~65% slower than pre (253 vs 153 ms).
   - One interesting case: `cat = common` (28% selectivity) — adaptive never falls back, 625 ms vs pre 8,280 ms
     (13x faster), but recall 0.78, below target. Here the problem is again the low unfiltered recall at nprobe=32.
   - Conclusion: adaptive search removes the need to tune g, but its speed-up depends on how well the data
     clusters. On SIFT-like data it matches tuned auto; on GloVe it is mostly "pre-filter with a small detour".
7. **Batching (C7) is dataset-independent**: −92% at batch 1024 on both datasets, because at that batch size each
   partition is read ~once for the whole batch.

8. **Updates (C8)** — seed 500K, +500K in 20K batches, nprobe=16, 30 queries (`updates_glove_1000000_c8.png`).
   Before inserts: 22 ms, recall 0.626.

   | policy | latency after +500K (random / skewed) | recall after +500K | inserts/s (random / skewed) |
   |---|---|---|---|
   | never merge | 1,583 / 1,396 ms (linear in delta) | 0.839 | 6,131 / 6,254 |
   | merge | 150 / 124 ms (saw-tooth 53–199 ms) | 0.673 / 0.672 | 3,611 / 4,204 |
   | merge + rebuild | 136 / 187 ms | 0.703 / 0.706 | 2,390 / 1,769 |

   - Same pattern as SIFT: without merging, the brute-force delta scan dominates; merging keeps latency bounded.
   - Never-merge recall *rises* (0.63 → 0.84) because a growing share of the data is searched exactly.
   - **Different from SIFT:** rebuilds (1 random, 2 skewed; k 707 → 894 / 825 → 990) *raise* recall (0.70–0.73 vs
     0.66–0.69 merge-only), although nprobe stays 16 while k grows. On SIFT rebuild lowered recall. Likely
     incremental merge degrades GloVe's already weak clustering more and re-clustering repairs it.
   - Throughput is ~6–10x lower than on SIFT (different machine, 200-dim vectors): compare policies, not datasets.
   - Merge-policy first row (42.6 ms vs 22 ms for the same untouched index) is timing noise.

### Issues found (for the write-up)
- **Windows portability (fixed):** `resource` module does not exist on Windows → `peak_rss_bytes()` falls back to
  psutil peak working set. Report files written with the default cp1252 encoding crashed on "≥" → all
  `open(..., "w")` now use `encoding="utf-8"`.
- **exp_updates with N=1,183,514 wrote only the seed line:** checkpoints need `done % 20000 == 0`, but
  N/2 = 591,757 never lines up. Rerun with `--n 1000000` (same as SIFT). A proper fix is to also checkpoint at the end.
- **Timing noise:** balanced MicroNN at nprobe=256 (685 ms) came out faster than nprobe=128 (1,457 ms median,
  4,296 ms mean), and batch b=1 at nprobe=16 was 35% faster than sequential. Single run on a laptop with
  background load/throttling; treat individual points with care, trends are consistent.
- C2 plot has a single N (no `--scaling` run on GloVe to save time), so the C2 figure is one point per system.


## Milestone 6 — research extension B: adaptive filtered search
Run (SIFT-like synthetic, since SIFT1M was not available on this machine):
`python -m bench.exp_adaptive --dataset synthetic-sift --n 100000 --n-query 30 --nprobe 10`, then
`python -m bench.analyze_adaptive results/adaptive_synthetic-sift_100000.json`
(tables `adaptive_synthetic-sift_100000_report.md`, plot `adaptive_synthetic-sift_100000_tradeoff.png`).
`synthetic-sift` = Gaussian mixture with spread 1.2: ~95% unfiltered recall@100 while probing ~3% of
partitions, the same operating point as SIFT1M at nprobe=32. **To do: rerun on SIFT1M (nprobe=32).**

**Question.** Extension A showed the optimizer's weak link is predicting post-filter recall *before* the
query runs (even exact selectivity picks the best plan only ~83% of the time). Can the plan be decided
*during* execution instead?

**Method (`plan="adaptive"` in `micronn/index.py`).** Scan partitions nearest-first with the filter applied.
Once k matches are found, measure the local selectivity s (matches / rows scanned so far) and continue to
`expand * nprobe / s` partitions, because the filtered top-k are roughly the top k/s unfiltered neighbours.
Fall back to the exact pre-filter whenever finishing the scan would cost more than the estimated
pre-filter cost. No recall-guard parameter to tune.

| strategy (all 35 filters × 30 queries) | mean ms | queries with recall ≥ 0.9 |
|---|---|---|
| post | 9.9 | 6% |
| auto independence g=1 (paper-style default) | 11.1 | 80% |
| auto independence g=4 | 20.2 | 97% |
| **adaptive e=1** | **22.4** | **97%** |
| pre / auto independence g=16 | 22.5 | 100% |

**Findings.**
1. Adaptive reaches 97% of queries on target with no tuning; the optimizer needs its recall guard tuned
   (g=1 → 80%, g=4 → 97%) to get there, and the right g is not known in advance.
2. It does **not** beat a well-tuned optimizer on this data: same quality, ~10% slower than g=4, because
   the partitions scanned before falling back to pre are wasted work.
3. Early version bugs found by the experiment (kept for the write-up): stopping as soon as k matches were
   found gave 0.41–0.49 recall on broad filters (k quick matches are not the k nearest matches);
   the fix is the k/s scaling above. A cost budget that ignored the remaining scan also over-scanned.
4. On this synthetic data pre-filter is cheap for almost every filter, so there is little room to win;
   SIFT1M, where post-filter is competitive up to ~1–10% selectivity, is the real test.

### SIFT1M rerun (Keerthana)

Run: `.venv/bin/python -m bench.exp_adaptive --dataset sift --n 1000000` (nprobe=32, 30 queries × 35 filters), twice:
with the cost model stored in the SIFT database (calibrated on a cold cache, point/scan ratio 14, the same one
extension A used) → `adaptive_sift_1000000*`; and after recalibrating on a warm cache (ratio 7.8) →
`adaptive_sift_1000000_warmcost*`. Log: `sift_adaptive_run.log`.

Reproducibility check: rerunning the synthetic-sift experiment on this machine gives identical recall for pre,
post and all six optimizer settings. Adaptive's quality shifts slightly (e=0.5: 95% → 85%, e=1: 97% → 96% of
queries on target), because its fallback budget uses the machine's own cost model. Latencies are about 2× lower
on this M5, and adaptive e=1 is slightly *slower* than always-pre here (12.0 vs 11.4 ms).

| strategy (all filters) | cold cost: mean ms | warm cost: mean ms | queries with recall ≥ 0.9 |
|---|---|---|---|
| auto stratified g=16 | 56.7 | 54.3 | 91% |
| auto independence g=4 | 70.8 | 67.5 | 86% |
| **adaptive e=0.5** | **105.6** | **95.7** | **100%** |
| auto independence g=16 | 110.3 | 104.6 | 95% |
| adaptive e=1 | 119.8 | 108.7 | 100% |
| pre | 174.9 | 168.2 | 100% |

**Findings on SIFT1M.**
1. *Adaptive is the fastest way to guarantee recall.* Every adaptive setting puts 100% of queries at recall ≥ 0.9.
   At e=0.5 and e=1 that is 31–43% faster than always-pre (e=2: only 15–17%). At e=0.5 it dominates the paper's optimizer at g=16 (faster and 100% vs 95%).
   This differs from the synthetic result, where it only tied a tuned optimizer.
2. *It does not help at lower recall targets.* The stratified optimizer (extension A) reaches 91% at about half
   the latency (54–57 ms).
3. *All of the gain comes from broad filters.* Adaptive skips the fallback on 6 of 35 filters (5 with the warm cost
   model), all matching 10–28% of rows, and is 3–10× faster than pre on 4 of them (e.g., `cat=common` 80 vs 779 ms; `p5=0 & q5=0` 196 vs
   706 ms). It falls back to pre on 83–85% of queries, and there it pays for the wasted scan: up to 3× slower
   than pre on selective anti-correlated filters (`p10=0 & q10=1`: 148 vs 51 ms).
4. *The fallback budget is too generous.* `calibrate()` times one SQL statement per point lookup, but pre-filter
   fetches rows in one batched join that measures ≈2.5–5 µs/row against ≈1 µs/row for a scan. The calibrated
   ratio is 7.8 (warm) or 14 (cold), so the budget can exceed the whole index. For `corr10=0`, a 10% filter whose
   matches sit far from most queries, adaptive scanned 709 of 1000 partitions and was slower than pre (316 vs
   248 ms). The warm cost model already cut mean partitions scanned from 68 to 44 and mean latency by 9%.
   Calibrating pre with a real batched pre-filter query should help further.
5. Caveat: within each filter, pre and post run before adaptive, so adaptive sees a warmer cache. Small gaps
   (< 10 ms) between adaptive-with-fallback and pre are within that noise.

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
