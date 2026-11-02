import numpy as np

from bench.datasets import attr_correlated, attr_zipf, ground_truth, recall_at_k, synthetic


def brute(base, queries, k, mask=None):
    d = ((queries[:, None, :] - base[None, :, :]) ** 2).sum(-1)
    if mask is not None:
        d[:, ~mask] = np.inf
    return np.argsort(d, axis=1)[:, :k]


def test_ground_truth_matches_brute_force_across_chunks():
    base, q, _ = synthetic(2000, dim=16, n_query=10)
    gt = ground_truth(base, q, k=10, chunk=300)
    assert recall_at_k(gt, brute(base, q, 10), 10) == 1.0


def test_filtered_ground_truth():
    base, q, _ = synthetic(1000, dim=16, n_query=5)
    mask = np.zeros(len(base), bool)
    mask[::50] = True  # 20 matching rows
    gt = ground_truth(base, q, k=30, mask=mask, chunk=128)
    assert np.all(mask[gt[:, :20]])
    assert np.all(gt[:, 20:] == -1)
    assert recall_at_k(gt[:, :20], brute(base, q, 20, mask), 20) == 1.0


def test_attribute_generators():
    base, _, _ = synthetic(5000, dim=16)
    z = attr_zipf(5000, 10)
    assert np.bincount(z)[0] > np.bincount(z, minlength=10)[9]
    c = attr_correlated(base, 10, noise=0.0)
    assert set(np.unique(c)) <= set(range(10))
