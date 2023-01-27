import warnings

import numpy as np
import scipy as sp
from numba import jit

from shap.utils._general import safe_isinstance


def partition_tree(X, metric="correlation"):
    X_full_rank = X + np.random.randn(*X.shape) * 1e-8
    D = sp.spatial.distance.pdist(
        X_full_rank.fillna(X_full_rank.mean()).T, metric=metric
    )
    return sp.cluster.hierarchy.complete(D)


def partition_tree_shuffle(indexes, index_mask, partition_tree):
    """Randomly shuffle the indexes in a way that is consistent with the given partition tree.

    Parameters
    ----------
    indexes: np.array
        The output location of the indexes we want shuffled. Note that len(indexes) should equal index_mask.sum().

    index_mask: np.array
        A bool mask of which indexes we want to include in the shuffled list.

    partition_tree: np.array
        The partition tree we should follow.
    """
    M = len(index_mask)
    # switch = np.random.randn(M) < 0
    _pt_shuffle_rec(
        partition_tree.shape[0] - 1, indexes, index_mask, partition_tree, M, 0
    )


@jit
def _pt_shuffle_rec(i, indexes, index_mask, partition_tree, M, pos):
    if i < 0:
        # see if we should include this index in the ordering
        if index_mask[i + M]:
            indexes[pos] = i + M
            return pos + 1
        else:
            return pos
    left = int(partition_tree[i, 0] - M)
    right = int(partition_tree[i, 1] - M)
    if np.random.randn() < 0:
        pos = _pt_shuffle_rec(left, indexes, index_mask, partition_tree, M, pos)
        pos = _pt_shuffle_rec(right, indexes, index_mask, partition_tree, M, pos)
    else:
        pos = _pt_shuffle_rec(right, indexes, index_mask, partition_tree, M, pos)
        pos = _pt_shuffle_rec(left, indexes, index_mask, partition_tree, M, pos)
    return pos


@jit
def delta_minimization_order(all_masks, max_swap_size=100, num_passes=2):
    order = np.arange(len(all_masks))
    for _ in range(num_passes):
        for length in list(range(2, max_swap_size)):
            for i in range(1, len(order) - length):
                if _reverse_window_score_gain(all_masks, order, i, length) > 0:
                    _reverse_window(order, i, length)
    return order


@jit
def _reverse_window(order, start, length):
    for i in range(length // 2):
        tmp = order[start + i]
        order[start + i] = order[start + length - i - 1]
        order[start + length - i - 1] = tmp


@jit
def _reverse_window_score_gain(masks, order, start, length):
    forward_score = _mask_delta_score(
        masks[order[start - 1]], masks[order[start]]
    ) + _mask_delta_score(
        masks[order[start + length - 1]], masks[order[start + length]]
    )
    reverse_score = _mask_delta_score(
        masks[order[start - 1]], masks[order[start + length - 1]]
    ) + _mask_delta_score(masks[order[start]], masks[order[start + length]])

    return forward_score - reverse_score


@jit
def _mask_delta_score(m1, m2):
    return (m1 ^ m2).sum()


def hclust_ordering(X, metric="sqeuclidean", anchor_first=False):
    """A leaf ordering is under-defined, this picks the ordering that keeps nearby samples similar."""

    # compute a hierarchical clustering and return the optimal leaf ordering
    D = sp.spatial.distance.pdist(X, metric)
    cluster_matrix = sp.cluster.hierarchy.complete(D)
    return sp.cluster.hierarchy.leaves_list(
        sp.cluster.hierarchy.optimal_leaf_ordering(cluster_matrix, D)
    )


def hclust(X, y=None, linkage="single", metric="auto", random_state=0):
    if safe_isinstance(X, "pandas.core.frame.DataFrame"):
        X = X.values

    if y is not None:
        warnings.warn(
            "Ignoring the y argument passed to shap.utils.hclust since the given clustering metric is not based on label fitting!"
        )
    if safe_isinstance(X, "pandas.core.frame.DataFrame"):
        bg_no_nan = X.values.copy()
    else:
        bg_no_nan = X.copy()
    for i in range(bg_no_nan.shape[1]):
        np.nan_to_num(bg_no_nan[:, i], nan=np.nanmean(bg_no_nan[:, i]), copy=False)
    dist = sp.spatial.distance.pdist(
        bg_no_nan.T + np.random.randn(*bg_no_nan.T.shape) * 1e-8, metric=metric
    )

    # build linkage
    if linkage == "single":
        return sp.cluster.hierarchy.single(dist)
    elif linkage == "complete":
        return sp.cluster.hierarchy.complete(dist)
    elif linkage == "average":
        return sp.cluster.hierarchy.average(dist)
    else:
        raise Exception("Unknown linkage: " + str(linkage))
