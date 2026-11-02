# Research extension A — sift N=1,000,000, nprobe=32, 30 queries × 35 filters, sample budget 32,000 rows

## Estimator cost

| estimator | build s | memory KiB |
|---|---|---|
| independence | 1.0 | 27 |
| column-group | 59.0 | 6,663 |
| sample | 1.0 | 4,027 |
| stratified | 20.9 | 4,218 |

## Selectivity q-error (median / max) by filter category

| estimator | independent | correlated | anti-correlated | vector-correlated |
|---|---|---|---|---|
| independence | 1.0 / 1.0 | 17.9 / 89.9 | 9.8 / 11.7 | 1.0 / 1.0 |
| column-group | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.1 | 1.0 / 1.0 |
| sample | 1.0 / 1.1 | 1.0 / 1.2 | 1.4 / 7.4 | 1.0 / 1.1 |
| stratified | 1.0 / 1.4 | 1.0 / 1.2 | 1.2 / 7.3 | 1.1 / 1.4 |

## Plan choice, recall guard = 1·k (target recall@100 ≥ 0.9)

| estimator | category | plan accuracy | queries ≥ target | mean recall | mean ms | regret ms |
|---|---|---|---|---|---|---|
| independence | **all** | 62% | 68% | 0.867 | 47.7 | -45.4 |
| independence | independent | 64% | 64% | 0.903 | 27.0 | -132.8 |
| independence | correlated | 75% | 89% | 0.973 | 84.7 | -26.7 |
| independence | anti-correlated | 56% | 56% | 0.741 | 33.3 | -10.0 |
| independence | vector-correlated | 45% | 45% | 0.802 | 15.4 | -50.8 |
| column-group | **all** | 61% | 61% | 0.881 | 32.8 | -60.3 |
| column-group | independent | 64% | 64% | 0.903 | 27.0 | -132.8 |
| column-group | correlated | 50% | 50% | 0.857 | 41.0 | -70.4 |
| column-group | anti-correlated | 89% | 89% | 0.960 | 38.4 | -5.0 |
| column-group | vector-correlated | 45% | 45% | 0.802 | 15.4 | -50.8 |
| sample | **all** | 61% | 61% | 0.881 | 32.8 | -60.3 |
| sample | independent | 64% | 64% | 0.903 | 27.0 | -132.8 |
| sample | correlated | 50% | 50% | 0.857 | 41.0 | -70.4 |
| sample | anti-correlated | 89% | 89% | 0.960 | 38.4 | -5.0 |
| sample | vector-correlated | 45% | 45% | 0.802 | 15.4 | -50.8 |
| stratified | **all** | 60% | 60% | 0.873 | 33.1 | -60.0 |
| stratified | independent | 52% | 52% | 0.857 | 28.6 | -131.2 |
| stratified | correlated | 50% | 50% | 0.857 | 41.0 | -70.4 |
| stratified | anti-correlated | 89% | 89% | 0.960 | 38.4 | -5.0 |
| stratified | vector-correlated | 46% | 46% | 0.806 | 15.6 | -50.6 |
| oracle-global | **all** | 61% | 61% | 0.881 | 32.8 | -60.3 |
| oracle-global | independent | 64% | 64% | 0.903 | 27.0 | -132.8 |
| oracle-global | correlated | 50% | 50% | 0.857 | 41.0 | -70.4 |
| oracle-global | anti-correlated | 89% | 89% | 0.960 | 38.4 | -5.0 |
| oracle-global | vector-correlated | 45% | 45% | 0.802 | 15.4 | -50.8 |
| oracle-local | **all** | 61% | 61% | 0.885 | 33.7 | -59.4 |
| oracle-local | independent | 53% | 53% | 0.860 | 28.6 | -131.2 |
| oracle-local | correlated | 50% | 50% | 0.857 | 41.0 | -70.4 |
| oracle-local | anti-correlated | 89% | 89% | 0.960 | 38.4 | -5.0 |
| oracle-local | vector-correlated | 54% | 54% | 0.861 | 18.6 | -47.6 |

## Plan choice, recall guard = 16·k (target recall@100 ≥ 0.9)

| estimator | category | plan accuracy | queries ≥ target | mean recall | mean ms | regret ms |
|---|---|---|---|---|---|---|
| independence | **all** | 80% | 95% | 0.986 | 113.3 | +20.1 |
| independence | independent | 83% | 90% | 0.973 | 77.0 | -82.8 |
| independence | correlated | 66% | 100% | 1.000 | 224.8 | +113.4 |
| independence | anti-correlated | 100% | 100% | 1.000 | 43.5 | +0.2 |
| independence | vector-correlated | 79% | 83% | 0.955 | 27.0 | -39.2 |
| column-group | **all** | 83% | 91% | 0.976 | 61.2 | -31.9 |
| column-group | independent | 83% | 90% | 0.973 | 77.0 | -82.8 |
| column-group | correlated | 75% | 89% | 0.973 | 84.7 | -26.7 |
| column-group | anti-correlated | 100% | 100% | 1.000 | 43.5 | +0.2 |
| column-group | vector-correlated | 79% | 83% | 0.955 | 27.0 | -39.2 |
| sample | **all** | 83% | 91% | 0.976 | 61.2 | -31.9 |
| sample | independent | 83% | 90% | 0.973 | 77.0 | -82.8 |
| sample | correlated | 75% | 89% | 0.973 | 84.7 | -26.7 |
| sample | anti-correlated | 100% | 100% | 1.000 | 43.5 | +0.2 |
| sample | vector-correlated | 79% | 83% | 0.955 | 27.0 | -39.2 |
| stratified | **all** | 86% | 91% | 0.976 | 58.5 | -34.6 |
| stratified | independent | 83% | 90% | 0.973 | 76.7 | -83.1 |
| stratified | correlated | 78% | 86% | 0.962 | 66.6 | -44.8 |
| stratified | anti-correlated | 100% | 100% | 1.000 | 43.5 | +0.2 |
| stratified | vector-correlated | 84% | 90% | 0.974 | 47.3 | -18.9 |
| oracle-global | **all** | 83% | 91% | 0.976 | 61.2 | -31.9 |
| oracle-global | independent | 83% | 90% | 0.973 | 77.0 | -82.8 |
| oracle-global | correlated | 75% | 89% | 0.973 | 84.7 | -26.7 |
| oracle-global | anti-correlated | 100% | 100% | 1.000 | 43.5 | +0.2 |
| oracle-global | vector-correlated | 79% | 83% | 0.955 | 27.0 | -39.2 |
| oracle-local | **all** | 87% | 92% | 0.977 | 59.0 | -34.1 |
| oracle-local | independent | 83% | 90% | 0.973 | 77.0 | -82.8 |
| oracle-local | correlated | 81% | 87% | 0.965 | 67.2 | -44.2 |
| oracle-local | anti-correlated | 100% | 100% | 1.000 | 43.5 | +0.2 |
| oracle-local | vector-correlated | 84% | 91% | 0.976 | 48.5 | -17.7 |
