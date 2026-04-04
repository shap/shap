import warnings

import numpy as np
import pandas as pd
import pytest

from shap.utils import hclust
from shap.utils._clustering import (
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


def test_hclust_with_dataframe_input():
    """hclust should accept a pandas DataFrame as input."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": [7.0, 8.0, 9.0]})
    result = hclust(df, random_state=0)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 4)


def test_hclust_with_nan_values():
    """hclust should handle NaN values in input by replacing them with column means."""
    X = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
    result = hclust(X, random_state=0)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 4)


def test_hclust_warns_when_y_passed_with_scipy_metric():
    """hclust should warn when y is provided but metric is not xgboost-based."""
    X = np.column_stack((np.arange(1, 10), np.arange(100, 1000, step=100)))
    y = np.ones(9)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        hclust(X, y=y, metric="cosine", random_state=0)
        assert any("Ignoring the y argument" in str(warning.message) for warning in w)


@pytest.mark.parametrize("metric", ["euclidean", "cosine", "sqeuclidean"])
def test_hclust_with_various_scipy_metrics(metric):
    """hclust should work with different scipy distance metrics."""
    X = np.random.RandomState(0).randn(10, 3)
    result = hclust(X, metric=metric, random_state=0)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 4)


def test_partition_tree():
    """partition_tree should return a valid linkage matrix."""
    df = pd.DataFrame(np.random.RandomState(0).randn(20, 4), columns=["a", "b", "c", "d"])
    result = partition_tree(df)
    assert isinstance(result, np.ndarray)
    # linkage matrix has shape (n_features - 1, 4)
    assert result.shape == (3, 4)


def test_partition_tree_shuffle():
    """partition_tree_shuffle should fill indexes consistent with the partition tree."""
    df = pd.DataFrame(np.random.RandomState(0).randn(20, 4), columns=["a", "b", "c", "d"])
    pt = partition_tree(df)
    M = 4
    index_mask = np.ones(M, dtype=np.bool_)
    indexes = np.zeros(M, dtype=np.intp)
    partition_tree_shuffle(indexes, index_mask, pt)
    # all original indices should appear in the output
    assert set(indexes) == set(range(M))


def test_partition_tree_shuffle_partial_mask():
    """partition_tree_shuffle should only include masked indices."""
    df = pd.DataFrame(np.random.RandomState(0).randn(20, 4), columns=["a", "b", "c", "d"])
    pt = partition_tree(df)
    M = 4
    index_mask = np.array([True, False, True, False], dtype=np.bool_)
    num_selected = index_mask.sum()
    indexes = np.zeros(num_selected, dtype=np.intp)
    partition_tree_shuffle(indexes, index_mask, pt)
    # only indices where mask is True should appear
    assert set(indexes) == {0, 2}


def test_delta_minimization_order():
    """delta_minimization_order should return a valid permutation of row indices."""
    np.random.seed(0)
    masks = np.random.randint(0, 2, size=(10, 5)).astype(np.bool_)
    order = delta_minimization_order(masks, max_swap_size=5, num_passes=1)
    assert isinstance(order, np.ndarray)
    assert len(order) == 10
    assert set(order) == set(range(10))


def test_hclust_ordering():
    """hclust_ordering should return a valid ordering of samples."""
    X = np.random.RandomState(0).randn(10, 3)
    order = hclust_ordering(X)
    assert isinstance(order, np.ndarray)
    assert len(order) == 10
    assert set(order) == set(range(10))
