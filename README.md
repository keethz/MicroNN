# MicroNN re-implementation

A scaled-down Python + SQLite re-implementation of *MicroNN: An On-device Disk-resident Updatable Vector Database* (Pound et al., SIGMOD-Companion '25), with experiments that test the paper's claims and two research extensions on filtered search: correlation-aware selectivity estimation (A) and adaptive filtered search (B).

- [REPORT.md](REPORT.md) — the write-up for the re-implementation and research extension A (claims, results, verdicts)
- [PLAN.md](PLAN.md) — the original plan and the pass/fail criterion for each claim
- [results/NOTES.md](results/NOTES.md) — running lab notes, including bugs found along the way
- [results/adaptive_synthetic-sift_100000_report.md](results/adaptive_synthetic-sift_100000_report.md) — adaptive filtered search results (research extension B)

## Setup

Requires Python 3.9+ (tested with 3.9.6 on macOS and 3.13 on Linux).

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python -m pytest -q
```

On Windows, use `.venv\Scripts\python` instead of `.venv/bin/python` (and `.venv\Scripts\pip` for pip).

### SIFT1M

Download `sift.tar.gz` from http://corpus-texmex.irisa.fr/ (ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz,
~160 MB) and extract it so the files sit at `data/sift/sift_base.fvecs` etc.

## Using the library

```python
import numpy as np
from micronn import MicroNN
from micronn.filters import eq, lt
from micronn.index import MaintenancePolicy

db = MicroNN.create("photos.sqlite", dim=128, attributes={"city": "TEXT", "year": "INTEGER"})
db.add(vids, vectors, attrs)          # attrs: list of (city, year) tuples; goes to the delta store
db.build()                            # mini-batch k-means, k = sqrt(N) partitions
db.analyze()                          # statistics for the filter optimizer
ids, dists = db.search(q, k=100, nprobe=16)
ids, dists = db.search(q, k=100, nprobe=16, filter=[eq("city", "Seattle"), lt("year", 2020)])
ids, dists = db.search(q, k=100, nprobe=16, filter=[eq("city", "Seattle")], plan="adaptive")  # decides during the scan
db.add(new_vids, new_vectors, new_attrs)   # searchable immediately
db.maintain(MaintenancePolicy(delta_max=50_000))  # merge / rebuild when needed
```

## Reproducing the experiments (SIFT1M)

Each step writes JSON and a log to `results/`. Run them in this order, since later steps reuse databases built
by earlier ones (under `data/db/`, about 4 GB in total).

| Claims | Command |
|---|---|
| C1–C4 | `.venv/bin/python -m bench.exp_search --dataset sift --n 1000000 --n-query 200 --scaling 125000,250000,500000,1000000` |
| C5–C6 | `.venv/bin/python -m bench.exp_filter --dataset sift --n 1000000 --n-query 50` |
| C7 | `.venv/bin/python -m bench.exp_batch --dataset sift --n 1000000` |
| C8 | `.venv/bin/python -m bench.exp_updates --dataset sift --n 1000000` |
| Research | `.venv/bin/python -m bench.exp_research --dataset sift --n 1000000` |
| Research B (adaptive) | `.venv/bin/python -m bench.exp_adaptive --dataset sift --n 1000000` |

Plots and tables:

```bash
.venv/bin/python -m bench.plot_search results/search_sift_1000000.json
```

```bash
.venv/bin/python -m bench.plot_filter results/filter_sift_1000000.json
```

```bash
.venv/bin/python -m bench.plot_updates results/batch_sift_1000000.json results/updates_sift_1000000.json
```

```bash
.venv/bin/python -m bench.analyze_research results/research_sift_1000000.json
```

```bash
.venv/bin/python -m bench.analyze_adaptive results/adaptive_sift_1000000.json
```

Every script also accepts `--dataset synthetic --n 100000` for a quick run without downloading SIFT, or
`--dataset synthetic-sift` for synthetic data at SIFT1M's recall/nprobe operating point (use `--nprobe 10`).

## Layout

```
micronn/   storage, clustering, index, filters, optimizer, batch, baselines, estimators
bench/     datasets + ground truth, experiment scripts, plotting
tests/     51 unit tests (correctness, filters, estimators, updates, concurrency, crash safety)
results/   JSON results, logs, plots, NOTES.md
data/      SIFT1M, cached ground truth, SQLite databases (not needed to read the report)
```
