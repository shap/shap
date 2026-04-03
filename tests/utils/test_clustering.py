import numpy as np
import pandas as pd
import pytest

from shap.utils import hclust, partition_tree
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


def test_hclust_with_nan_values():
    # NaN values in X should be imputed (filled with column mean) and not crash
    X = np.array(
        [
            [1.0, 100.0],
            [np.nan, 200.0],
            [3.0, 300.0],
            [4.0, np.nan],
            [5.0, 500.0],
            [6.0, 600.0],
            [7.0, 700.0],
            [8.0, 800.0],
            [9.0, 900.0],
        ]
    )
    result = hclust(X, random_state=0)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 4)


def test_partition_tree_with_nan():
    # partition_tree should handle NaN via fillna(mean)
    X = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, np.nan, 40.0, 50.0],
            "c": [100.0, 200.0, 300.0, 400.0, 500.0],
        }
    )
    result = partition_tree(X)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 4)
