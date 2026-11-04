# Research extension A — gist N=100,000, nprobe=32, 30 queries × 35 filters, sample budget 32,000 rows

## Estimator cost

| estimator | build s | memory KiB |
|---|---|---|
| independence | 0.1 | 27 |
| column-group | 3.4 | 6,457 |
| sample | 0.1 | 4,027 |
| stratified | 0.6 | 3,171 |

## Selectivity q-error (median / max) by filter category

| estimator | independent | correlated | anti-correlated | vector-correlated |
|---|---|---|---|---|
| independence | 1.0 / 1.1 | 17.6 / 89.3 | 10.1 / 18.9 | 1.0 / 1.0 |
| column-group | 1.0 / 1.1 | 1.0 / 1.1 | 1.0 / 2.5 | 1.0 / 1.0 |
| sample | 1.0 / 1.2 | 1.0 / 1.2 | 1.0 / 9.4 | 1.0 / 1.1 |
| stratified | 1.0 / 1.2 | 1.0 / 1.7 | 1.4 / 12.7 | 1.0 / 1.9 |

## Plan choice, recall guard = 1·k (target recall@100 ≥ 0.9)

| estimator | category | plan accuracy | queries ≥ target | mean recall | mean ms | regret ms |
|---|---|---|---|---|---|---|
| independence | **all** | 79% | 86% | 0.948 | 19.3 | -4.2 |
| independence | independent | 72% | 78% | 0.944 | 18.9 | -19.0 |
| independence | correlated | 79% | 96% | 0.990 | 32.3 | +1.4 |
| independence | anti-correlated | 78% | 78% | 0.878 | 6.6 | +3.2 |
| independence | vector-correlated | 86% | 86% | 0.966 | 11.9 | -11.4 |
| column-group | **all** | 80% | 83% | 0.955 | 13.6 | -9.9 |
| column-group | independent | 72% | 78% | 0.944 | 18.9 | -19.0 |
| column-group | correlated | 67% | 71% | 0.922 | 19.1 | -11.7 |
| column-group | anti-correlated | 100% | 100% | 1.000 | 3.4 | +0.0 |
| column-group | vector-correlated | 86% | 86% | 0.966 | 11.9 | -11.4 |
| sample | **all** | 80% | 83% | 0.955 | 13.6 | -9.9 |
| sample | independent | 72% | 78% | 0.944 | 18.9 | -19.0 |
| sample | correlated | 67% | 71% | 0.922 | 19.1 | -11.7 |
| sample | anti-correlated | 100% | 100% | 1.000 | 3.4 | +0.0 |
| sample | vector-correlated | 86% | 86% | 0.966 | 11.9 | -11.4 |
| stratified | **all** | 80% | 83% | 0.955 | 13.6 | -9.9 |
| stratified | independent | 72% | 78% | 0.944 | 18.9 | -19.0 |
| stratified | correlated | 67% | 71% | 0.922 | 19.1 | -11.7 |
| stratified | anti-correlated | 100% | 100% | 1.000 | 3.4 | +0.0 |
| stratified | vector-correlated | 86% | 86% | 0.966 | 11.9 | -11.4 |
| oracle-global | **all** | 80% | 83% | 0.955 | 13.6 | -9.9 |
| oracle-global | independent | 72% | 78% | 0.944 | 18.9 | -19.0 |
| oracle-global | correlated | 67% | 71% | 0.922 | 19.1 | -11.7 |
| oracle-global | anti-correlated | 100% | 100% | 1.000 | 3.4 | +0.0 |
| oracle-global | vector-correlated | 86% | 86% | 0.966 | 11.9 | -11.4 |
| oracle-local | **all** | 80% | 83% | 0.955 | 13.6 | -9.9 |
| oracle-local | independent | 72% | 78% | 0.944 | 18.9 | -19.0 |
| oracle-local | correlated | 67% | 71% | 0.922 | 19.1 | -11.7 |
| oracle-local | anti-correlated | 100% | 100% | 1.000 | 3.4 | +0.0 |
| oracle-local | vector-correlated | 86% | 86% | 0.966 | 11.9 | -11.4 |

## Plan choice, recall guard = 16·k (target recall@100 ≥ 0.9)

| estimator | category | plan accuracy | queries ≥ target | mean recall | mean ms | regret ms |
|---|---|---|---|---|---|---|
| independence | **all** | 84% | 99% | 0.998 | 36.6 | +13.1 |
| independence | independent | 86% | 96% | 0.991 | 42.3 | +4.4 |
| independence | correlated | 72% | 100% | 1.000 | 58.1 | +27.3 |
| independence | anti-correlated | 100% | 100% | 1.000 | 3.4 | +0.0 |
| independence | vector-correlated | 85% | 100% | 1.000 | 34.5 | +11.2 |
| column-group | **all** | 87% | 98% | 0.995 | 27.0 | +3.5 |
| column-group | independent | 86% | 96% | 0.991 | 42.3 | +4.4 |
| column-group | correlated | 79% | 96% | 0.990 | 32.3 | +1.4 |
| column-group | anti-correlated | 100% | 100% | 1.000 | 3.4 | +0.0 |
| column-group | vector-correlated | 85% | 100% | 1.000 | 34.5 | +11.2 |
| sample | **all** | 87% | 98% | 0.995 | 27.0 | +3.5 |
| sample | independent | 86% | 96% | 0.991 | 42.3 | +4.4 |
| sample | correlated | 79% | 96% | 0.990 | 32.3 | +1.4 |
| sample | anti-correlated | 100% | 100% | 1.000 | 3.4 | +0.0 |
| sample | vector-correlated | 85% | 100% | 1.000 | 34.5 | +11.2 |
| stratified | **all** | 93% | 96% | 0.989 | 18.6 | -4.9 |
| stratified | independent | 92% | 93% | 0.984 | 25.3 | -12.6 |
| stratified | correlated | 86% | 94% | 0.984 | 25.4 | -5.4 |
| stratified | anti-correlated | 100% | 100% | 1.000 | 3.4 | +0.0 |
| stratified | vector-correlated | 96% | 96% | 0.988 | 19.9 | -3.5 |
| oracle-global | **all** | 87% | 98% | 0.995 | 27.0 | +3.5 |
| oracle-global | independent | 86% | 96% | 0.991 | 42.3 | +4.4 |
| oracle-global | correlated | 79% | 96% | 0.990 | 32.3 | +1.4 |
| oracle-global | anti-correlated | 100% | 100% | 1.000 | 3.4 | +0.0 |
| oracle-global | vector-correlated | 85% | 100% | 1.000 | 34.5 | +11.2 |
| oracle-local | **all** | 93% | 96% | 0.989 | 18.7 | -4.8 |
| oracle-local | independent | 92% | 93% | 0.983 | 24.5 | -13.4 |
| oracle-local | correlated | 86% | 94% | 0.984 | 25.6 | -5.2 |
| oracle-local | anti-correlated | 100% | 100% | 1.000 | 3.4 | +0.0 |
| oracle-local | vector-correlated | 97% | 97% | 0.989 | 20.6 | -2.7 |
