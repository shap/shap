import numpy as np
from shap.utils import rank_interactions


def test_rank_interactions_basic():
    np.random.seed(0)

    X = np.random.randn(100, 3)

    # create synthetic interaction between feature 0 and 1
    shap_values = np.zeros_like(X)
    shap_values[:, 0] = X[:, 0] * X[:, 1]
    shap_values[:, 1] = X[:, 0] * X[:, 1]
    shap_values[:, 2] = np.random.randn(100)

    result = rank_interactions(shap_values, X)

    top_pair = result[0][:2]

    assert set(top_pair) == {"f0", "f1"}


def test_rank_interactions_max_pairs():
    X = np.random.randn(50, 4)
    shap_values = np.random.randn(50, 4)

    result = rank_interactions(shap_values, X, max_pairs=2)

    assert len(result) == 2