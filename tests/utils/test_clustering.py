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


def test_partition_tree_returns_ndarray():
    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    assert isinstance(partition_tree(X), np.ndarray)


def test_partition_tree_shape():
    X = pd.DataFrame(np.random.randn(10, 4))
    assert partition_tree(X).shape == (3, 4)


def test_partition_tree_with_nan():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan]})
    assert partition_tree(X).shape == (1, 4)


def test_partition_tree_euclidean_metric():
    X = pd.DataFrame(np.random.randn(10, 3))
    assert partition_tree(X, metric="euclidean").shape == (2, 4)


def test_partition_tree_shuffle_contains_all_masked_indexes():
    X = pd.DataFrame(np.random.randn(10, 4))
    ptree = partition_tree(X)
    index_mask = np.array([True, True, True, True])
    indexes = np.zeros(4, dtype=np.intp)
    partition_tree_shuffle(indexes, index_mask, ptree)
    assert sorted(indexes) == [0, 1, 2, 3]


def test_partition_tree_shuffle_respects_mask():
    X = pd.DataFrame(np.random.randn(10, 4))
    ptree = partition_tree(X)
    index_mask = np.array([True, False, True, False])
    indexes = np.zeros(2, dtype=np.intp)
    partition_tree_shuffle(indexes, index_mask, ptree)
    assert set(indexes).issubset({0, 2})


def test_partition_tree_shuffle_output_length():
    X = pd.DataFrame(np.random.randn(10, 5))
    ptree = partition_tree(X)
    index_mask = np.array([True, True, False, True, False])
    indexes = np.zeros(3, dtype=np.intp)
    partition_tree_shuffle(indexes, index_mask, ptree)
    assert len(indexes) == 3


def test_mask_delta_score_identical_masks():
    m = np.array([True, False, True, True])
    assert _mask_delta_score(m, m) == 0


def test_mask_delta_score_fully_different():
    m1 = np.array([True, True, False, False])
    m2 = np.array([False, False, True, True])
    assert _mask_delta_score(m1, m2) == 4


def test_mask_delta_score_partial_overlap():
    m1 = np.array([True, False, True, False])
    m2 = np.array([True, True, False, False])
    assert _mask_delta_score(m1, m2) == 2


def test_reverse_window_reverses_slice():
    order = np.array([0, 1, 2, 3, 4])
    _reverse_window(order, 1, 3)
    np.testing.assert_array_equal(order, [0, 3, 2, 1, 4])


def test_reverse_window_length_two():
    order = np.array([0, 1, 2, 3])
    _reverse_window(order, 0, 2)
    np.testing.assert_array_equal(order, [1, 0, 2, 3])


def test_reverse_window_doesnt_touch_outside_range():
    order = np.array([0, 1, 2, 3, 4])
    _reverse_window(order, 2, 2)
    assert order[0] == 0 and order[1] == 1 and order[4] == 4


def test_reverse_window_score_gain_is_integer():
    masks = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.bool_)
    result = _reverse_window_score_gain(masks, np.arange(4), 1, 2)
    assert isinstance(result, (int, np.integer))


def test_reverse_window_score_gain_uniform_masks():
    masks = np.ones((5, 3), dtype=np.bool_)
    assert _reverse_window_score_gain(masks, np.arange(5), 1, 3) == 0


def test_delta_minimization_order_is_permutation():
    masks = np.random.randint(0, 2, size=(8, 4)).astype(np.bool_)
    order = delta_minimization_order(masks)
    assert sorted(order) == list(range(8))


def test_delta_minimization_order_shape():
    masks = np.random.randint(0, 2, size=(6, 3)).astype(np.bool_)
    assert delta_minimization_order(masks).shape == (6,)


def test_delta_minimization_order_single_pass():
    masks = np.random.randint(0, 2, size=(5, 3)).astype(np.bool_)
    order = delta_minimization_order(masks, num_passes=1)
    assert sorted(order) == list(range(5))


def test_hclust_ordering_is_valid_permutation():
    X = np.random.randn(10, 4)
    assert sorted(hclust_ordering(X)) == list(range(10))


def test_hclust_ordering_shape():
    X = np.random.randn(8, 3)
    assert hclust_ordering(X).shape == (8,)


def test_hclust_ordering_sqeuclidean():
    X = np.random.randn(6, 2)
    assert len(hclust_ordering(X, metric="sqeuclidean")) == 6
