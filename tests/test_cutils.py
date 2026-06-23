"""Tests for the C++ utility functions in shap._cutils."""

import numpy as np
import scipy.cluster.hierarchy
from shap._cutils import (
    delta_minimization_order,
    mask_delta_score,
    pt_shuffle_rec,
    reverse_window,
    reverse_window_score_gain,
)

# Tests for reverse_window


# reverses a window in the middle of the array, leaving surrounding elements untouched
def test_reverse_window_basic():
    order = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    reverse_window(order, 1, 3)
    np.testing.assert_array_equal(order, [0, 3, 2, 1, 4])


# reverses the entire array
def test_reverse_window_full_array():
    order = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    reverse_window(order, 0, 5)
    np.testing.assert_array_equal(order, [4, 3, 2, 1, 0])


# length=1 is a no-op, array must be unchanged
def test_reverse_window_single_element():
    order = np.array([0, 1, 2, 3], dtype=np.int64)
    reverse_window(order, 2, 1)
    np.testing.assert_array_equal(order, [0, 1, 2, 3])


# length=0 is a no-op, array must be unchanged
def test_reverse_window_empty():
    order = np.array([0, 1, 2, 3], dtype=np.int64)
    reverse_window(order, 1, 0)
    np.testing.assert_array_equal(order, [0, 1, 2, 3])


# window positioned at the tail of the array
def test_reverse_window_at_end():
    order = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    reverse_window(order, 2, 3)
    np.testing.assert_array_equal(order, [0, 1, 4, 3, 2])


# Tests for mask_delta_score


# identical masks have zero Hamming distance
def test_mask_delta_score_identical():
    m = np.array([True, False, True, True], dtype=bool)
    assert mask_delta_score(m, m) == 0


# completely opposite masks have distance equal to array length
def test_mask_delta_score_all_different():
    m1 = np.array([True, True, False, False], dtype=bool)
    m2 = np.array([False, False, True, True], dtype=bool)
    assert mask_delta_score(m1, m2) == 4


# partially overlapping masks return the count of differing positions
def test_mask_delta_score_partial():
    m1 = np.array([True, False, True, False], dtype=bool)
    m2 = np.array([True, True, False, False], dtype=bool)
    assert mask_delta_score(m1, m2) == 2


# single-element arrays — one matching, one different
def test_mask_delta_score_single_element():
    assert mask_delta_score(np.array([True], dtype=bool), np.array([True], dtype=bool)) == 0
    assert mask_delta_score(np.array([True], dtype=bool), np.array([False], dtype=bool)) == 1


# Tests for reverse_window_score_gain


# reversing reduces total delta — gain is positive
def test_reverse_window_score_gain_positive():
    masks = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=bool)
    order = np.array([0, 1, 2, 3], dtype=np.int64)
    # forward: delta(row0,row1) + delta(row2,row3) = 2 + 2 = 4
    # reverse: delta(row0,row2) + delta(row1,row3) = 0 + 0 = 0
    assert reverse_window_score_gain(masks, order, 1, 2) == 4


# reversing increases total delta — gain is negative
def test_reverse_window_score_gain_negative():
    masks = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=bool)
    order = np.array([0, 1, 2, 3], dtype=np.int64)
    # forward: delta(row0,row1) + delta(row2,row3) = 0 + 0 = 0
    # reverse: delta(row0,row2) + delta(row1,row3) = 2 + 2 = 4
    assert reverse_window_score_gain(masks, order, 1, 2) == -4


# all identical masks — all deltas are zero, gain is always zero
def test_reverse_window_score_gain_zero():
    masks = np.array([[1, 0], [1, 0], [1, 0], [1, 0]], dtype=bool)
    order = np.array([0, 1, 2, 3], dtype=np.int64)
    assert reverse_window_score_gain(masks, order, 1, 2) == 0


# length=1 window — forward and reverse expressions are identical, gain is always zero
def test_reverse_window_score_gain_length_one():
    masks = np.array([[1, 0], [0, 1], [1, 1]], dtype=bool)
    order = np.array([0, 1, 2], dtype=np.int64)
    assert reverse_window_score_gain(masks, order, 1, 1) == 0


# Tests for delta_minimization_order


def _total_delta(masks, order):
    """Sum of adjacent Hamming distances for a given ordering."""
    return sum(int((masks[order[i]] ^ masks[order[i + 1]]).sum()) for i in range(len(order) - 1))


# single row — output must be [0]
def test_delta_minimization_order_single_row():
    masks = np.array([[1, 0]], dtype=bool)
    order = delta_minimization_order(masks)
    np.testing.assert_array_equal(order, [0])


# all identical masks — any ordering is equally good, output must be a valid permutation
def test_delta_minimization_order_identical_masks():
    masks = np.array([[1, 0], [1, 0], [1, 0]], dtype=bool)
    order = delta_minimization_order(masks)
    np.testing.assert_array_equal(sorted(order), [0, 1, 2])


# alternating pattern — identity ordering has total delta 6, optimal is 2
# rows [1,0],[0,1],[1,0],[0,1]: grouping similar rows cuts crossings from 3 to 1
def test_delta_minimization_order_reorders():
    masks = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=bool)
    order = delta_minimization_order(masks)
    np.testing.assert_array_equal(sorted(order), [0, 1, 2, 3])
    assert _total_delta(masks, order) < _total_delta(masks, np.array([0, 1, 2, 3]))


# Tests for pt_shuffle_rec


def _subtree_leaves(node, pt, M):
    """Recursively collect all leaf feature indices under a given internal node."""
    if node < 0:
        return {node + M}
    left = int(pt[node, 0]) - M
    right = int(pt[node, 1]) - M
    return _subtree_leaves(left, pt, M) | _subtree_leaves(right, pt, M)


def _build_tree(M):
    """Build a small scipy complete-linkage tree for M features."""
    np.random.seed(0)
    X = np.random.randn(M, 3)
    return scipy.cluster.hierarchy.complete(X).astype(np.float64)


# full mask — pos equals M and all feature indices 0..M-1 are written exactly once
def test_pt_shuffle_rec_full_mask():
    M = 8
    pt = _build_tree(M)
    index_mask = np.ones(M, dtype=bool)
    indexes = np.zeros(M, dtype=np.int64)
    pos = pt_shuffle_rec(int(pt.shape[0] - 1), indexes, index_mask, pt, M, 0)
    assert pos == M
    np.testing.assert_array_equal(sorted(indexes[:pos]), list(range(M)))


# partial mask — only masked features appear, count matches number of True entries
def test_pt_shuffle_rec_partial_mask():
    M = 8
    pt = _build_tree(M)
    index_mask = np.array([i % 2 == 0 for i in range(M)], dtype=bool)
    indexes = np.zeros(M, dtype=np.int64)
    pos = pt_shuffle_rec(int(pt.shape[0] - 1), indexes, index_mask, pt, M, 0)
    assert pos == int(index_mask.sum())
    np.testing.assert_array_equal(sorted(indexes[:pos]), sorted(np.where(index_mask)[0]))


# contiguity — every internal node's leaves appear as a contiguous block in the output
def test_pt_shuffle_rec_contiguity():
    M = 8
    pt = _build_tree(M)
    index_mask = np.ones(M, dtype=bool)
    indexes = np.zeros(M, dtype=np.int64)
    pos = pt_shuffle_rec(int(pt.shape[0] - 1), indexes, index_mask, pt, M, 0)
    result = indexes[:pos].tolist()
    for node in range(pt.shape[0]):
        leaves = _subtree_leaves(node, pt, M)
        positions = [result.index(leaf) for leaf in leaves]
        assert max(positions) - min(positions) + 1 == len(positions)


# stability — contiguity holds across 20 random runs, ruling out flaky RNG behaviour
def test_pt_shuffle_rec_contiguity_stable():
    M = 8
    pt = _build_tree(M)
    index_mask = np.ones(M, dtype=bool)
    for _ in range(20):
        indexes = np.zeros(M, dtype=np.int64)
        pos = pt_shuffle_rec(int(pt.shape[0] - 1), indexes, index_mask, pt, M, 0)
        result = indexes[:pos].tolist()
        for node in range(pt.shape[0]):
            leaves = _subtree_leaves(node, pt, M)
            positions = [result.index(leaf) for leaf in leaves]
            assert max(positions) - min(positions) + 1 == len(positions)
