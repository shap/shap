import numpy as np
import pytest

from shap.utils import hclust


@pytest.mark.parametrize("linkage", ["single", "complete", "average"])
def test_hclust_runs(linkage):
    # GH #3290
    pytest.importorskip('xgboost')
    X = np.column_stack((np.arange(1, 10), np.arange(100, 1000, step=100)))
    y = np.where(X[:, 0] > 5, 1, 0)

    # just check if clustered ran successfully
    clustered = hclust(X, y, linkage=linkage)
    assert isinstance(clustered, np.ndarray)

    # Check clustering runs if y=None
    clustered = hclust(X, linkage=linkage)
    assert isinstance(clustered, np.ndarray)
