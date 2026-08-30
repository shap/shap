"""Synthetic make_masks inputs for parity capture.

The five production tests that reach make_masks all share one xgboost/adult
scenario and therefore one cluster matrix; this file feeds the capture plugin
clusterings of different sizes and shapes (tiny, balanced, degenerate chain,
real scipy linkage output, and an image-scale tree).
"""

import numpy as np
import pytest
import scipy.cluster.hierarchy
import scipy.spatial.distance

from shap.utils._masked_model import make_masks


def _chain(M):
    """Fully unbalanced clustering: each merge adds one leaf."""
    rows = []
    prev, size = 0, 1
    for k in range(M - 1):
        rows.append([prev, k + 1, float(k + 1), size + 1])
        prev, size = M + k, size + 1
    return np.array(rows, dtype=np.float64)


def _linkage(n, seed):
    rng = np.random.RandomState(seed)
    data = rng.standard_normal((n, 5))
    dist = scipy.spatial.distance.pdist(data)
    return scipy.cluster.hierarchy.complete(dist)


@pytest.mark.parametrize(
    "cluster_matrix",
    [
        np.array([[0.0, 1.0, 1.0, 2.0]]),  # M=2, single merge
        np.array([[0.0, 1.0, 1.0, 2.0], [2.0, 3.0, 1.0, 2.0], [4.0, 5.0, 2.0, 4.0]]),  # M=4 balanced
        _chain(6),  # M=6 degenerate chain
        _linkage(30, seed=0),  # M=30 real linkage
        _linkage(300, seed=1),  # M=300 image-scale-ish
    ],
    ids=["M2", "M4-balanced", "M6-chain", "M30-linkage", "M300-linkage"],
)
def test_make_masks_case(cluster_matrix):
    M = cluster_matrix.shape[0] + 1
    mask = make_masks(cluster_matrix)
    assert mask.shape == (2 * M - 1, M)
    # every leaf row selects exactly itself; the root row selects everything
    assert mask[:M].nnz == M
    assert mask[2 * M - 2].nnz == M
