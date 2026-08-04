import numpy as np
import pandas as pd
import pytest

from shap.utils import hclust
from shap.utils._clustering import (
    _mask_delta_score,
    _reverse_window,
    _reverse_window_score_gain,
    delta_minimization_order,
    hclust_ordering,
    partition_tree,
    partition_tree_shuffle,
)
from shap.utils._exceptions import DimensionError

# ============================================================
# Tests for hclust (existing + new)
# ============================================================


@pytest.mark.parametrize("linkage", ["single", "complete", "average"])
def test_hclust_runs(linkage):
    # GH #3290
    pytest.importorskip("xgboost")
    X = np.column_stack((np.arange(1, 10), np.arange(100, 1000, step=100)))
    y = np.where(X[:, 0] > 5, 1, 0)

    # just check if clustered ran successfully (using xgboost_distances_r2)
    clustered = hclust(X, y, linkage=linkage, random_state=0)
    assert isinstance(clustered, np.ndarray)
    assert clustered.shape == (1, 4)

    # Check clustering runs if y=None (using scipy metrics)
    clustered = hclust(X, linkage=linkage, random_state=0)
    assert isinstance(clustered, np.ndarray)
    assert clustered.shape == (1, 4)


@pytest.mark.parametrize(
    "X",
    [
        np.arange(1, 10),
        list(range(1, 10)),
    ],
)
def test_hclust_errors_on_input_shapes(X):
    # hclust only accepts 2-d arrays for X
    with pytest.raises(DimensionError):
        hclust(X, random_state=0)


def test_hclust_errors_on_unknown_linkages():
    X = np.column_stack((np.arange(1, 10), np.arange(100, 1000, step=100)))
    with pytest.raises(ValueError, match=r"Unknown linkage type:"):
        hclust(X, linkage="random-string", random_state=0)  # type: ignore


def test_hclust_accepts_dataframe():
    """hclust should accept a pandas DataFrame as input."""
    X = pd.DataFrame(np.column_stack((np.arange(1, 10), np.arange(100, 1000, step=100))))
    clustered = hclust(X, random_state=0)
    assert isinstance(clustered, np.ndarray)
    assert clustered.shape == (1, 4)


def test_hclust_with_explicit_metric():
    """hclust should accept an explicit scipy distance metric."""
    X = np.column_stack((np.arange(1, 10), np.arange(100, 1000, step=100)))
    clustered = hclust(X, metric="euclidean", random_state=0)
    assert isinstance(clustered, np.ndarray)
    assert clustered.shape == (1, 4)


def test_hclust_warns_when_y_ignored():
    """hclust should warn when y is provided but metric is not label-based."""
    X = np.column_stack((np.arange(1, 10), np.arange(100, 1000, step=100)))
    y = np.where(X[:, 0] > 5, 1, 0)
    with pytest.warns(UserWarning, match="Ignoring the y argument"):
        hclust(X, y=y, metric="cosine", random_state=0)


# ============================================================
# Tests for _mask_delta_score
# ============================================================


def test_mask_delta_score_identical():
    """Identical masks should have a delta score of 0."""
    m = np.array([1, 0, 1, 0], dtype=np.int64)
    assert _mask_delta_score(m, m) == 0


def test_mask_delta_score_completely_different():
    """Completely opposite masks should have delta score equal to length."""
    m1 = np.array([1, 1, 1, 1], dtype=np.int64)
    m2 = np.array([0, 0, 0, 0], dtype=np.int64)
    assert _mask_delta_score(m1, m2) == 4


def test_mask_delta_score_partial():
    """Partially different masks should return the count of differing bits."""
    m1 = np.array([1, 0, 1, 0], dtype=np.int64)
    m2 = np.array([1, 1, 0, 0], dtype=np.int64)
    # XOR: [0, 1, 1, 0] -> sum = 2
    assert _mask_delta_score(m1, m2) == 2


def test_mask_delta_score_empty():
    """Empty masks should have a delta score of 0."""
    m1 = np.array([], dtype=np.int64)
    m2 = np.array([], dtype=np.int64)
    assert _mask_delta_score(m1, m2) == 0


# ============================================================
# Tests for _reverse_window
# ============================================================


def test_reverse_window_basic():
    """Reversing a window in the middle of an array."""
    order = np.array([0, 1, 2, 3, 4, 5])
    _reverse_window(order, start=1, length=3)
    # Reverses indices 1..3: [1, 2, 3] -> [3, 2, 1]
    np.testing.assert_array_equal(order, [0, 3, 2, 1, 4, 5])


def test_reverse_window_length_two():
    """Reversing a window of length 2 should swap two elements."""
    order = np.array([10, 20, 30, 40])
    _reverse_window(order, start=1, length=2)
    np.testing.assert_array_equal(order, [10, 30, 20, 40])


def test_reverse_window_length_one():
    """Reversing a window of length 1 should be a no-op."""
    order = np.array([10, 20, 30, 40])
    _reverse_window(order, start=2, length=1)
    np.testing.assert_array_equal(order, [10, 20, 30, 40])


def test_reverse_window_entire_array():
    """Reversing the entire array."""
    order = np.array([1, 2, 3, 4, 5])
    _reverse_window(order, start=0, length=5)
    np.testing.assert_array_equal(order, [5, 4, 3, 2, 1])


# ============================================================
# Tests for _reverse_window_score_gain
# ============================================================


def test_reverse_window_score_gain_positive():
    """Score gain should be positive when reversal reduces delta."""
    # Alternating pattern: reversing adjacent similar rows should help
    masks = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
        ],
        dtype=np.int64,
    )
    order = np.arange(5)
    gain = _reverse_window_score_gain(masks, order, start=1, length=2)
    # forward: score(masks[0], masks[1]) + score(masks[2], masks[3]) = 3 + 3 = 6
    # reverse: score(masks[0], masks[2]) + score(masks[1], masks[3]) = 0 + 0 = 0
    # gain = 6
    assert gain == 6


def test_reverse_window_score_gain_zero():
    """Score gain should be zero when reversal doesn't change anything."""
    masks = np.array(
        [
            [0, 0],
            [1, 1],
            [1, 1],
            [0, 0],
        ],
        dtype=np.int64,
    )
    order = np.arange(4)
    gain = _reverse_window_score_gain(masks, order, start=1, length=2)
    # forward: score(masks[0], masks[1]) + score(masks[2], masks[3]) = 2 + 2 = 4
    # reverse: score(masks[0], masks[2]) + score(masks[1], masks[3]) = 2 + 2 = 4
    # gain = 0
    assert gain == 0


# ============================================================
# Tests for delta_minimization_order
# ============================================================


def test_delta_minimization_order_returns_permutation():
    """The result should be a permutation of the input indices."""
    masks = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 0],
        ],
        dtype=np.int64,
    )
    order = delta_minimization_order(masks, max_swap_size=3, num_passes=1)
    assert set(order) == {0, 1, 2, 3}
    assert len(order) == 4


def test_delta_minimization_order_reduces_total_delta():
    """The optimized order should have total delta <= the original order."""
    masks = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
        ],
        dtype=np.int64,
    )

    def total_delta(masks, order):
        return sum(_mask_delta_score(masks[order[i]], masks[order[i + 1]]) for i in range(len(order) - 1))

    original_order = np.arange(len(masks))
    original_delta = total_delta(masks, original_order)

    optimized_order = delta_minimization_order(masks, max_swap_size=4, num_passes=2)
    optimized_delta = total_delta(masks, optimized_order)

    assert optimized_delta <= original_delta


# ============================================================
# Tests for partition_tree
# ============================================================


def test_partition_tree_output_shape():
    """partition_tree should return an (n_features-1, 4) linkage matrix."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(20, 5))
    tree = partition_tree(X)
    assert isinstance(tree, np.ndarray)
    # For 5 features, the linkage matrix has (5-1) = 4 rows
    assert tree.shape == (4, 4)


def test_partition_tree_with_nan():
    """partition_tree should handle DataFrames containing NaN values."""
    rng = np.random.RandomState(42)
    data = rng.randn(20, 4)
    data[0, 0] = np.nan
    data[5, 2] = np.nan
    X = pd.DataFrame(data)
    tree = partition_tree(X)
    assert isinstance(tree, np.ndarray)
    assert tree.shape == (3, 4)
    assert not np.any(np.isnan(tree))


# ============================================================
# Tests for partition_tree_shuffle
# ============================================================


def test_partition_tree_shuffle_produces_permutation():
    """partition_tree_shuffle should produce a valid permutation of masked indices."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(20, 4))
    tree = partition_tree(X)

    M = 4  # number of leaf nodes (features)
    index_mask = np.array([True, True, True, True])
    indexes = np.zeros(M, dtype=np.intp)

    partition_tree_shuffle(indexes, index_mask, tree)

    # All indices from 0..M-1 should appear exactly once
    assert set(indexes) == {0, 1, 2, 3}


def test_partition_tree_shuffle_respects_mask():
    """Only indices where index_mask is True should appear in the output."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(20, 4))
    tree = partition_tree(X)

    index_mask = np.array([True, False, True, False])
    num_selected = index_mask.sum()
    indexes = np.zeros(num_selected, dtype=np.intp)

    partition_tree_shuffle(indexes, index_mask, tree)

    # Only indices 0 and 2 should appear
    assert set(indexes) == {0, 2}


# ============================================================
# Tests for hclust_ordering
# ============================================================


def test_hclust_ordering_returns_permutation():
    """hclust_ordering should return a valid permutation of sample indices."""
    rng = np.random.RandomState(42)
    X = rng.randn(10, 3)
    ordering = hclust_ordering(X)
    assert isinstance(ordering, np.ndarray)
    assert set(ordering) == set(range(10))
    assert len(ordering) == 10
