# Adaptive filtered search — synthetic-sift, N=100000, nprobe=10, 30 queries × 35 filters

### All filters

| strategy | mean latency (ms) | p95 (ms) | mean recall@100 | queries with recall ≥ 0.9 |
|---|---|---|---|---|
| auto stratified g=1 | 8.4 | 15.9 | 0.699 | 56% |
| auto stratified g=4 | 9.1 | 21.1 | 0.833 | 74% |
| post | 9.9 | 14.1 | 0.258 | 6% |
| auto independence g=1 | 11.1 | 39.5 | 0.871 | 80% |
| auto stratified g=16 | 17.4 | 49.6 | 0.960 | 93% |
| auto independence g=4 | 20.2 | 77.1 | 0.985 | 97% |
| adaptive e=0.5 | 22.0 | 78.3 | 0.980 | 95% |
| adaptive e=1 | 22.4 | 72.7 | 0.992 | 97% |
| pre | 22.5 | 85.4 | 1.000 | 100% |
| auto independence g=16 | 22.5 | 85.4 | 1.000 | 100% |
| adaptive e=2 | 22.9 | 79.5 | 0.997 | 99% |

### anti-correlated filters

| strategy | mean latency (ms) | p95 (ms) | mean recall@100 | queries with recall ≥ 0.9 |
|---|---|---|---|---|
| auto independence g=1 | 6.1 | 11.5 | 0.906 | 89% |
| pre | 6.5 | 15.0 | 1.000 | 100% |
| auto independence g=4 | 6.5 | 15.0 | 1.000 | 100% |
| auto independence g=16 | 6.5 | 15.0 | 1.000 | 100% |
| auto stratified g=1 | 6.5 | 15.0 | 1.000 | 100% |
| auto stratified g=4 | 6.5 | 15.0 | 1.000 | 100% |
| auto stratified g=16 | 6.5 | 15.0 | 1.000 | 100% |
| adaptive e=2 | 9.9 | 30.4 | 1.000 | 100% |
| post | 10.0 | 11.9 | 0.308 | 22% |
| adaptive e=1 | 10.3 | 30.1 | 1.000 | 100% |
| adaptive e=0.5 | 10.4 | 31.0 | 1.000 | 100% |

### correlated filters

| strategy | mean latency (ms) | p95 (ms) | mean recall@100 | queries with recall ≥ 0.9 |
|---|---|---|---|---|
| auto stratified g=1 | 10.6 | 16.6 | 0.539 | 33% |
| post | 11.3 | 15.4 | 0.257 | 0% |
| auto stratified g=4 | 11.4 | 21.2 | 0.743 | 60% |
| auto independence g=1 | 17.0 | 43.8 | 0.910 | 85% |
| auto stratified g=16 | 19.4 | 49.7 | 0.931 | 88% |
| adaptive e=1 | 26.4 | 73.0 | 1.000 | 100% |
| adaptive e=2 | 26.9 | 80.8 | 1.000 | 100% |
| pre | 27.0 | 78.6 | 1.000 | 100% |
| auto independence g=4 | 27.0 | 78.6 | 1.000 | 100% |
| auto independence g=16 | 27.0 | 78.6 | 1.000 | 100% |
| adaptive e=0.5 | 27.5 | 80.0 | 1.000 | 100% |

### independent filters

| strategy | mean latency (ms) | p95 (ms) | mean recall@100 | queries with recall ≥ 0.9 |
|---|---|---|---|---|
| auto stratified g=1 | 10.4 | 38.3 | 0.611 | 42% |
| post | 10.9 | 13.9 | 0.246 | 0% |
| auto independence g=1 | 12.6 | 39.4 | 0.805 | 67% |
| auto stratified g=4 | 12.6 | 39.4 | 0.805 | 67% |
| auto independence g=4 | 35.0 | 149.5 | 0.914 | 83% |
| auto stratified g=16 | 35.0 | 149.5 | 0.914 | 83% |
| adaptive e=0.5 | 36.5 | 143.3 | 0.928 | 82% |
| adaptive e=1 | 40.2 | 144.1 | 0.959 | 84% |
| adaptive e=2 | 43.2 | 142.4 | 0.983 | 93% |
| pre | 48.4 | 149.5 | 1.000 | 100% |
| auto independence g=16 | 48.4 | 149.5 | 1.000 | 100% |

### vector-correlated filters

| strategy | mean latency (ms) | p95 (ms) | mean recall@100 | queries with recall ≥ 0.9 |
|---|---|---|---|---|
| auto independence g=1 | 5.0 | 11.2 | 0.810 | 71% |
| auto stratified g=4 | 5.0 | 11.2 | 0.810 | 71% |
| auto stratified g=1 | 5.2 | 11.3 | 0.682 | 55% |
| post | 6.5 | 12.4 | 0.207 | 0% |
| pre | 12.7 | 35.1 | 1.000 | 100% |
| auto independence g=4 | 12.7 | 35.1 | 1.000 | 100% |
| auto independence g=16 | 12.7 | 35.1 | 1.000 | 100% |
| auto stratified g=16 | 12.7 | 35.1 | 1.000 | 100% |
| adaptive e=0.5 | 14.2 | 39.2 | 0.961 | 90% |
| adaptive e=2 | 14.6 | 37.6 | 1.000 | 100% |
| adaptive e=1 | 15.3 | 42.2 | 0.997 | 99% |
