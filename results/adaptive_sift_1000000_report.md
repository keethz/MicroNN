# Adaptive filtered search — sift, N=1000000, nprobe=32, 30 queries × 35 filters

### All filters

| strategy | mean latency (ms) | p95 (ms) | mean recall@100 | queries with recall ≥ 0.9 |
|---|---|---|---|---|
| auto stratified g=1 | 31.8 | 48.3 | 0.873 | 60% |
| post | 33.4 | 45.3 | 0.586 | 21% |
| auto stratified g=4 | 42.4 | 99.4 | 0.940 | 77% |
| auto independence g=1 | 44.0 | 175.7 | 0.867 | 68% |
| auto stratified g=16 | 56.7 | 216.4 | 0.976 | 91% |
| auto independence g=4 | 70.8 | 313.2 | 0.953 | 86% |
| adaptive e=0.5 | 105.6 | 316.1 | 0.999 | 100% |
| auto independence g=16 | 110.3 | 582.5 | 0.986 | 95% |
| adaptive e=1 | 119.8 | 317.4 | 1.000 | 100% |
| adaptive e=2 | 145.0 | 449.0 | 1.000 | 100% |
| pre | 174.9 | 759.7 | 1.000 | 100% |

### anti-correlated filters

| strategy | mean latency (ms) | p95 (ms) | mean recall@100 | queries with recall ≥ 0.9 |
|---|---|---|---|---|
| auto independence g=1 | 28.3 | 42.4 | 0.741 | 56% |
| auto independence g=4 | 31.1 | 46.6 | 0.905 | 78% |
| auto stratified g=1 | 35.7 | 73.2 | 0.960 | 89% |
| post | 37.7 | 46.2 | 0.215 | 0% |
| pre | 41.4 | 82.1 | 1.000 | 100% |
| auto independence g=16 | 41.4 | 82.1 | 1.000 | 100% |
| auto stratified g=4 | 41.4 | 82.1 | 1.000 | 100% |
| auto stratified g=16 | 41.4 | 82.1 | 1.000 | 100% |
| adaptive e=2 | 71.9 | 139.3 | 1.000 | 100% |
| adaptive e=1 | 73.2 | 144.0 | 1.000 | 100% |
| adaptive e=0.5 | 73.6 | 139.2 | 1.000 | 100% |

### correlated filters

| strategy | mean latency (ms) | p95 (ms) | mean recall@100 | queries with recall ≥ 0.9 |
|---|---|---|---|---|
| post | 37.0 | 45.1 | 0.768 | 34% |
| auto stratified g=1 | 38.1 | 45.4 | 0.857 | 50% |
| auto stratified g=4 | 43.9 | 99.9 | 0.911 | 66% |
| auto stratified g=16 | 60.9 | 175.3 | 0.962 | 86% |
| auto independence g=1 | 77.0 | 181.1 | 0.973 | 89% |
| auto independence g=4 | 121.6 | 316.9 | 0.989 | 96% |
| adaptive e=0.5 | 130.7 | 317.2 | 0.999 | 100% |
| adaptive e=1 | 143.8 | 316.5 | 1.000 | 100% |
| adaptive e=2 | 172.6 | 392.1 | 1.000 | 100% |
| pre | 216.6 | 620.7 | 1.000 | 100% |
| auto independence g=16 | 216.6 | 620.7 | 1.000 | 100% |

### independent filters

| strategy | mean latency (ms) | p95 (ms) | mean recall@100 | queries with recall ≥ 0.9 |
|---|---|---|---|---|
| auto independence g=1 | 29.7 | 45.8 | 0.903 | 64% |
| auto stratified g=1 | 32.0 | 47.1 | 0.857 | 52% |
| post | 38.0 | 47.1 | 0.698 | 31% |
| auto stratified g=4 | 56.0 | 282.2 | 0.936 | 74% |
| auto independence g=4 | 68.8 | 282.8 | 0.942 | 78% |
| auto stratified g=16 | 77.7 | 282.8 | 0.973 | 90% |
| auto independence g=16 | 77.9 | 282.8 | 0.973 | 90% |
| adaptive e=0.5 | 114.8 | 295.4 | 0.996 | 99% |
| adaptive e=1 | 151.5 | 342.9 | 1.000 | 100% |
| adaptive e=2 | 209.5 | 634.0 | 1.000 | 100% |
| pre | 371.5 | 1028.5 | 1.000 | 100% |

### vector-correlated filters

| strategy | mean latency (ms) | p95 (ms) | mean recall@100 | queries with recall ≥ 0.9 |
|---|---|---|---|---|
| auto independence g=1 | 15.1 | 37.8 | 0.802 | 45% |
| auto stratified g=1 | 15.3 | 38.0 | 0.806 | 46% |
| post | 17.1 | 37.8 | 0.632 | 17% |
| auto stratified g=4 | 29.0 | 76.9 | 0.918 | 70% |
| auto independence g=4 | 29.2 | 76.7 | 0.955 | 83% |
| auto independence g=16 | 29.2 | 76.7 | 0.955 | 83% |
| auto stratified g=16 | 50.3 | 242.0 | 0.974 | 90% |
| adaptive e=0.5 | 92.0 | 445.2 | 0.999 | 100% |
| pre | 100.3 | 285.3 | 1.000 | 100% |
| adaptive e=1 | 108.1 | 445.1 | 1.000 | 100% |
| adaptive e=2 | 132.7 | 453.1 | 1.000 | 100% |
