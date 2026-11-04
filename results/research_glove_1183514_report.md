# Research extension A — glove N=1,183,514, nprobe=32, 30 queries × 35 filters, sample budget 32,000 rows

## Estimator cost

| estimator | build s | memory KiB |
|---|---|---|
| independence | 2.6 | 27 |
| column-group | 223.1 | 6,663 |
| sample | 3.1 | 4,027 |
| stratified | 47.8 | 3,296 |

## Selectivity q-error (median / max) by filter category

| estimator | independent | correlated | anti-correlated | vector-correlated |
|---|---|---|---|---|
| independence | 1.0 / 1.0 | 18.2 / 90.7 | 10.4 / 15.0 | 1.0 / 1.0 |
| column-group | 1.0 / 1.0 | 1.0 / 1.1 | 1.0 / 1.1 | 1.0 / 1.0 |
| sample | 1.0 / 1.2 | 1.0 / 1.1 | 1.6 / 13.9 | 1.0 / 1.1 |
| stratified | 1.1 / 1.6 | 1.0 / 1.1 | 1.8 / 15.0 | 1.1 / 1.1 |

## Plan choice, recall guard = 1·k (target recall@100 ≥ 0.9)

| estimator | category | plan accuracy | queries ≥ target | mean recall | mean ms | regret ms |
|---|---|---|---|---|---|---|
| independence | **all** | 58% | 60% | 0.751 | 304.3 | -900.5 |
| independence | independent | 43% | 43% | 0.673 | 196.4 | -2232.1 |
| independence | correlated | 71% | 75% | 0.874 | 561.3 | -986.3 |
| independence | anti-correlated | 67% | 67% | 0.744 | 161.9 | +8.8 |
| independence | vector-correlated | 36% | 36% | 0.600 | 102.3 | -769.0 |
| column-group | **all** | 52% | 52% | 0.719 | 195.5 | -1009.3 |
| column-group | independent | 43% | 43% | 0.673 | 196.4 | -2232.1 |
| column-group | correlated | 33% | 33% | 0.611 | 274.6 | -1273.0 |
| column-group | anti-correlated | 100% | 100% | 1.000 | 153.1 | +0.0 |
| column-group | vector-correlated | 36% | 36% | 0.600 | 102.3 | -769.0 |
| sample | **all** | 52% | 52% | 0.719 | 195.5 | -1009.3 |
| sample | independent | 43% | 43% | 0.673 | 196.4 | -2232.1 |
| sample | correlated | 33% | 33% | 0.611 | 274.6 | -1273.0 |
| sample | anti-correlated | 100% | 100% | 1.000 | 153.1 | +0.0 |
| sample | vector-correlated | 36% | 36% | 0.600 | 102.3 | -769.0 |
| stratified | **all** | 52% | 52% | 0.719 | 195.5 | -1009.3 |
| stratified | independent | 43% | 43% | 0.673 | 196.4 | -2232.1 |
| stratified | correlated | 33% | 33% | 0.611 | 274.6 | -1273.0 |
| stratified | anti-correlated | 100% | 100% | 1.000 | 153.1 | +0.0 |
| stratified | vector-correlated | 36% | 36% | 0.600 | 102.3 | -769.0 |
| oracle-global | **all** | 52% | 52% | 0.719 | 195.5 | -1009.3 |
| oracle-global | independent | 43% | 43% | 0.673 | 196.4 | -2232.1 |
| oracle-global | correlated | 33% | 33% | 0.611 | 274.6 | -1273.0 |
| oracle-global | anti-correlated | 100% | 100% | 1.000 | 153.1 | +0.0 |
| oracle-global | vector-correlated | 36% | 36% | 0.600 | 102.3 | -769.0 |
| oracle-local | **all** | 52% | 52% | 0.719 | 195.5 | -1009.3 |
| oracle-local | independent | 43% | 43% | 0.673 | 196.4 | -2232.1 |
| oracle-local | correlated | 33% | 33% | 0.611 | 274.6 | -1273.0 |
| oracle-local | anti-correlated | 100% | 100% | 1.000 | 153.1 | +0.0 |
| oracle-local | vector-correlated | 36% | 36% | 0.600 | 102.3 | -769.0 |

## Plan choice, recall guard = 16·k (target recall@100 ≥ 0.9)

| estimator | category | plan accuracy | queries ≥ target | mean recall | mean ms | regret ms |
|---|---|---|---|---|---|---|
| independence | **all** | 86% | 91% | 0.953 | 842.3 | -362.5 |
| independence | independent | 71% | 74% | 0.870 | 524.0 | -1904.5 |
| independence | correlated | 91% | 100% | 1.000 | 1812.0 | +264.4 |
| independence | anti-correlated | 100% | 100% | 1.000 | 153.1 | +0.0 |
| independence | vector-correlated | 74% | 76% | 0.874 | 200.2 | -671.1 |
| column-group | **all** | 79% | 81% | 0.906 | 377.7 | -827.0 |
| column-group | independent | 71% | 74% | 0.870 | 524.0 | -1904.5 |
| column-group | correlated | 71% | 75% | 0.874 | 561.3 | -986.3 |
| column-group | anti-correlated | 100% | 100% | 1.000 | 153.1 | +0.0 |
| column-group | vector-correlated | 74% | 76% | 0.874 | 200.2 | -671.1 |
| sample | **all** | 77% | 79% | 0.892 | 340.1 | -864.7 |
| sample | independent | 71% | 74% | 0.870 | 524.0 | -1904.5 |
| sample | correlated | 65% | 68% | 0.836 | 460.0 | -1087.6 |
| sample | anti-correlated | 100% | 100% | 1.000 | 153.1 | +0.0 |
| sample | vector-correlated | 74% | 76% | 0.874 | 200.2 | -671.1 |
| stratified | **all** | 75% | 77% | 0.879 | 308.0 | -896.8 |
| stratified | independent | 72% | 74% | 0.870 | 520.7 | -1907.8 |
| stratified | correlated | 60% | 62% | 0.801 | 375.1 | -1172.6 |
| stratified | anti-correlated | 100% | 100% | 1.000 | 153.1 | +0.0 |
| stratified | vector-correlated | 74% | 76% | 0.874 | 200.2 | -671.1 |
| oracle-global | **all** | 79% | 81% | 0.906 | 377.7 | -827.0 |
| oracle-global | independent | 71% | 74% | 0.870 | 524.0 | -1904.5 |
| oracle-global | correlated | 71% | 75% | 0.874 | 561.3 | -986.3 |
| oracle-global | anti-correlated | 100% | 100% | 1.000 | 153.1 | +0.0 |
| oracle-global | vector-correlated | 74% | 76% | 0.874 | 200.2 | -671.1 |
| oracle-local | **all** | 75% | 76% | 0.878 | 304.5 | -900.2 |
| oracle-local | independent | 71% | 74% | 0.870 | 524.0 | -1904.5 |
| oracle-local | correlated | 59% | 62% | 0.798 | 364.2 | -1183.4 |
| oracle-local | anti-correlated | 100% | 100% | 1.000 | 153.1 | +0.0 |
| oracle-local | vector-correlated | 74% | 76% | 0.874 | 200.2 | -671.1 |
