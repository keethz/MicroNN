# Adaptive filtered search — glove, N=1183514, nprobe=32, 30 queries × 35 filters

### All filters

| strategy | mean latency (ms) | p95 (ms) | mean recall@100 | queries with recall ≥ 0.9 |
|---|---|---|---|---|
| auto stratified g=1 | 195.4 | 353.1 | 0.719 | 52% |
| auto stratified g=4 | 219.8 | 394.4 | 0.757 | 58% |
| post | 241.0 | 364.2 | 0.336 | 7% |
| auto independence g=1 | 302.2 | 1427.5 | 0.751 | 60% |
| auto stratified g=16 | 307.3 | 788.3 | 0.879 | 77% |
| auto independence g=4 | 507.8 | 2703.6 | 0.872 | 78% |
| adaptive e=0.5 | 662.6 | 2783.7 | 0.960 | 89% |
| auto independence g=16 | 837.9 | 5411.7 | 0.953 | 91% |
| adaptive e=1 | 1175.1 | 5815.0 | 0.991 | 97% |
| adaptive e=2 | 1260.5 | 5840.0 | 0.997 | 99% |
| pre | 1463.5 | 7977.1 | 1.000 | 100% |

### anti-correlated filters

| strategy | mean latency (ms) | p95 (ms) | mean recall@100 | queries with recall ≥ 0.9 |
|---|---|---|---|---|
| auto independence g=4 | 149.1 | 307.7 | 0.836 | 78% |
| pre | 153.4 | 332.8 | 1.000 | 100% |
| auto independence g=16 | 153.4 | 332.8 | 1.000 | 100% |
| auto stratified g=1 | 153.4 | 332.8 | 1.000 | 100% |
| auto stratified g=4 | 153.4 | 332.8 | 1.000 | 100% |
| auto stratified g=16 | 153.4 | 332.8 | 1.000 | 100% |
| auto independence g=1 | 159.8 | 323.8 | 0.744 | 67% |
| adaptive e=0.5 | 247.1 | 549.9 | 1.000 | 100% |
| adaptive e=1 | 252.6 | 582.4 | 1.000 | 100% |
| adaptive e=2 | 255.0 | 583.0 | 1.000 | 100% |
| post | 265.1 | 363.7 | 0.125 | 0% |

### correlated filters

| strategy | mean latency (ms) | p95 (ms) | mean recall@100 | queries with recall ≥ 0.9 |
|---|---|---|---|---|
| auto stratified g=1 | 264.7 | 362.2 | 0.611 | 33% |
| auto stratified g=4 | 274.7 | 409.5 | 0.652 | 39% |
| post | 281.1 | 377.6 | 0.435 | 10% |
| auto stratified g=16 | 361.3 | 718.6 | 0.801 | 62% |
| auto independence g=1 | 547.9 | 1616.1 | 0.874 | 75% |
| auto independence g=4 | 943.0 | 2891.1 | 0.941 | 88% |
| adaptive e=0.5 | 1033.3 | 2997.1 | 0.959 | 89% |
| pre | 1788.7 | 5950.9 | 1.000 | 100% |
| auto independence g=16 | 1788.7 | 5950.9 | 1.000 | 100% |
| adaptive e=2 | 1806.0 | 5905.6 | 1.000 | 100% |
| adaptive e=1 | 1806.9 | 5940.6 | 1.000 | 100% |

### independent filters

| strategy | mean latency (ms) | p95 (ms) | mean recall@100 | queries with recall ≥ 0.9 |
|---|---|---|---|---|
| auto independence g=1 | 215.7 | 368.4 | 0.673 | 43% |
| auto stratified g=1 | 215.7 | 368.4 | 0.673 | 43% |
| post | 269.6 | 371.2 | 0.405 | 10% |
| auto stratified g=4 | 303.7 | 1542.9 | 0.713 | 49% |
| auto independence g=4 | 453.5 | 1785.1 | 0.775 | 59% |
| auto stratified g=16 | 538.2 | 1785.1 | 0.870 | 74% |
| auto independence g=16 | 540.5 | 1785.1 | 0.870 | 74% |
| adaptive e=0.5 | 783.2 | 1805.3 | 0.919 | 78% |
| adaptive e=1 | 1581.3 | 7045.1 | 0.959 | 87% |
| adaptive e=2 | 1853.7 | 7045.8 | 0.982 | 94% |
| pre | 3223.5 | 8732.4 | 1.000 | 100% |

### vector-correlated filters

| strategy | mean latency (ms) | p95 (ms) | mean recall@100 | queries with recall ≥ 0.9 |
|---|---|---|---|---|
| auto independence g=1 | 103.2 | 275.1 | 0.600 | 36% |
| auto stratified g=1 | 103.2 | 275.1 | 0.600 | 36% |
| post | 111.3 | 275.1 | 0.364 | 7% |
| auto stratified g=4 | 131.4 | 393.9 | 0.679 | 48% |
| auto independence g=4 | 207.0 | 480.7 | 0.874 | 76% |
| auto independence g=16 | 207.0 | 480.7 | 0.874 | 76% |
| auto stratified g=16 | 207.0 | 480.7 | 0.874 | 76% |
| adaptive e=0.5 | 404.8 | 935.0 | 0.945 | 82% |
| adaptive e=1 | 839.9 | 3043.8 | 0.989 | 96% |
| adaptive e=2 | 1031.8 | 3160.9 | 1.000 | 100% |
| pre | 1035.3 | 3180.7 | 1.000 | 100% |
