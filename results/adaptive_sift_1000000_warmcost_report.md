# Adaptive filtered search — sift, N=1000000, nprobe=32, 30 queries × 35 filters

### All filters

| strategy | mean latency (ms) | p95 (ms) | mean recall@100 | queries with recall ≥ 0.9 |
|---|---|---|---|---|
| auto stratified g=1 | 29.8 | 47.0 | 0.881 | 62% |
| post | 32.0 | 44.1 | 0.586 | 21% |
| auto stratified g=4 | 41.0 | 81.6 | 0.940 | 77% |
| auto independence g=1 | 41.6 | 170.3 | 0.867 | 68% |
| auto stratified g=16 | 54.3 | 172.4 | 0.976 | 91% |
| auto independence g=4 | 67.5 | 310.2 | 0.953 | 86% |
| adaptive e=0.5 | 95.7 | 312.2 | 0.999 | 100% |
| auto independence g=16 | 104.6 | 570.5 | 0.986 | 95% |
| adaptive e=1 | 108.7 | 314.3 | 1.000 | 100% |
| adaptive e=2 | 142.6 | 579.7 | 1.000 | 100% |
| pre | 168.2 | 750.4 | 1.000 | 100% |

### anti-correlated filters

| strategy | mean latency (ms) | p95 (ms) | mean recall@100 | queries with recall ≥ 0.9 |
|---|---|---|---|---|
| auto independence g=1 | 27.8 | 42.5 | 0.741 | 56% |
| auto independence g=4 | 29.9 | 47.0 | 0.905 | 78% |
| auto stratified g=1 | 34.2 | 73.2 | 0.960 | 89% |
| post | 35.4 | 43.2 | 0.215 | 0% |
| pre | 39.3 | 81.4 | 1.000 | 100% |
| auto independence g=16 | 39.3 | 81.4 | 1.000 | 100% |
| auto stratified g=4 | 39.3 | 81.4 | 1.000 | 100% |
| auto stratified g=16 | 39.3 | 81.4 | 1.000 | 100% |
| adaptive e=1 | 60.9 | 118.2 | 1.000 | 100% |
| adaptive e=0.5 | 62.1 | 120.1 | 1.000 | 100% |
| adaptive e=2 | 62.3 | 117.9 | 1.000 | 100% |

### correlated filters

| strategy | mean latency (ms) | p95 (ms) | mean recall@100 | queries with recall ≥ 0.9 |
|---|---|---|---|---|
| auto stratified g=1 | 36.2 | 44.9 | 0.857 | 50% |
| post | 36.9 | 45.1 | 0.768 | 34% |
| auto stratified g=4 | 42.3 | 74.5 | 0.911 | 66% |
| auto stratified g=16 | 58.4 | 141.6 | 0.962 | 86% |
| auto independence g=1 | 72.5 | 171.6 | 0.973 | 89% |
| auto independence g=4 | 115.1 | 313.1 | 0.989 | 96% |
| adaptive e=0.5 | 127.2 | 315.7 | 0.999 | 100% |
| adaptive e=1 | 140.5 | 315.6 | 1.000 | 100% |
| adaptive e=2 | 198.2 | 586.8 | 1.000 | 100% |
| pre | 204.8 | 580.8 | 1.000 | 100% |
| auto independence g=16 | 204.8 | 580.8 | 1.000 | 100% |

### independent filters

| strategy | mean latency (ms) | p95 (ms) | mean recall@100 | queries with recall ≥ 0.9 |
|---|---|---|---|---|
| auto independence g=1 | 26.7 | 43.2 | 0.903 | 64% |
| auto stratified g=1 | 26.7 | 43.2 | 0.903 | 64% |
| post | 34.1 | 44.4 | 0.698 | 31% |
| auto stratified g=4 | 54.8 | 282.6 | 0.936 | 74% |
| auto independence g=4 | 68.0 | 284.0 | 0.942 | 78% |
| auto stratified g=16 | 76.4 | 284.0 | 0.973 | 90% |
| auto independence g=16 | 76.7 | 284.0 | 0.973 | 90% |
| adaptive e=0.5 | 107.6 | 292.9 | 0.996 | 99% |
| adaptive e=1 | 138.8 | 330.7 | 1.000 | 100% |
| adaptive e=2 | 194.8 | 607.2 | 1.000 | 100% |
| pre | 367.3 | 1039.3 | 1.000 | 100% |

### vector-correlated filters

| strategy | mean latency (ms) | p95 (ms) | mean recall@100 | queries with recall ≥ 0.9 |
|---|---|---|---|---|
| auto independence g=1 | 14.8 | 36.3 | 0.802 | 45% |
| auto stratified g=1 | 15.0 | 36.6 | 0.806 | 46% |
| post | 16.9 | 36.3 | 0.632 | 17% |
| auto independence g=4 | 26.7 | 74.9 | 0.955 | 83% |
| auto independence g=16 | 26.7 | 74.9 | 0.955 | 83% |
| auto stratified g=4 | 28.8 | 74.9 | 0.918 | 70% |
| auto stratified g=16 | 47.2 | 236.7 | 0.974 | 90% |
| adaptive e=0.5 | 70.4 | 241.3 | 0.999 | 100% |
| adaptive e=1 | 85.3 | 253.3 | 1.000 | 100% |
| pre | 95.2 | 271.3 | 1.000 | 100% |
| adaptive e=2 | 98.0 | 274.5 | 1.000 | 100% |
