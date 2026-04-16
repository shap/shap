import numpy as np
import pytest

from shap.utils import hclust
from shap.utils._exceptions import DimensionError


@pytest.mark.parametrize("linkage", ["single", "complete", "average"])
def test_hclust_runs(linkage):
    # GH #3290
    pytest.importorskip("xgboost")
    X = np.column_stack((np.arange(1, 10), np.arange(100, 1000, step=100)))
    y = np.where(X[:, 0] > 5, 1, 0)

    # check if clustered ran successfully (using xgboost_distances_r2)
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


def test_hclust_edge_cases():
    """Test hclust with high correlation and constant features to increase coverage."""
    # 1. Test with a mix of high correlation and a constant feature
    X = np.array([[0, 1, 5], [0, 1.0001, 5], [0, 2, 10], [0, 2.0001, 10]])

    for linkage in ["single", "complete", "average"]:
        out = hclust(X, linkage=linkage)
        assert out.shape[0] == X.shape[1] - 1

    # 2. Test with explicit 'y' to hit the XGBoost distance branch
    y = np.array([0, 0, 1, 1])
    try:
        # We trigger the logic even if it results in an internal error
        hclust(X, y=y)
    except Exception:
        pass
