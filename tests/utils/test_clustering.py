import numpy as np
import pytest

from shap.utils import hclust


def test_hclust_runs():
    # GH 3290
    pytest.importorskip('xgboost')
    X = np.column_stack((np.arange(1, 10), np.arange(100, 1000, step=100)))
    y = np.where(X[:, 0] > 5, 1, 0)

    clustered = hclust(X, y)
    # just check if clustered ran successfully
    assert isinstance(clustered, np.ndarray)
