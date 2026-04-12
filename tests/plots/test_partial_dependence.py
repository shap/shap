"""Tests for the partial_dependence plot."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor

import shap


@pytest.fixture()
def pd_model_and_data():
    """Train a simple model and return (model.predict, X) for partial dependence tests."""
    rs = np.random.RandomState(42)
    X = rs.randn(50, 4)
    y = X[:, 0] + 2 * X[:, 1] + rs.randn(50) * 0.1
    model = GradientBoostingRegressor(n_estimators=10, max_depth=2, random_state=42)
    model.fit(X, y)
    return model.predict, X


@pytest.mark.mpl_image_compare
def test_partial_dependence_1d(pd_model_and_data):
    """Check that the 1D partial dependence plot works."""
    predict, X = pd_model_and_data
    fig = plt.figure()
    shap.plots.partial_dependence(0, predict, X, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_partial_dependence_no_ice(pd_model_and_data):
    """Check that partial dependence plot works without ICE lines."""
    predict, X = pd_model_and_data
    fig = plt.figure()
    shap.plots.partial_dependence(0, predict, X, ice=False, show=False)
    plt.tight_layout()
    return fig


def test_partial_dependence_returns_ax(pd_model_and_data):
    """Check that the function returns an Axes object when show=False."""
    predict, X = pd_model_and_data
    result = shap.plots.partial_dependence(0, predict, X, show=False)
    assert isinstance(result, plt.Axes)
    plt.close()


def test_partial_dependence_accepts_ax(pd_model_and_data):
    """Check that a user-provided ax is used and returned."""
    predict, X = pd_model_and_data
    fig, ax = plt.subplots()
    result = shap.plots.partial_dependence(0, predict, X, ax=ax, show=False)
    assert result is ax
    plt.close(fig)


def test_partial_dependence_2d_returns_ax(pd_model_and_data):
    """Check that 2D partial dependence returns an Axes when show=False."""
    predict, X = pd_model_and_data
    result = shap.plots.partial_dependence((0, 1), predict, X, npoints=5, show=False)
    assert result is not None
    plt.close()


def test_partial_dependence_with_feature_names(pd_model_and_data):
    """Check that feature_names parameter works."""
    predict, X = pd_model_and_data
    names = ["a", "b", "c", "d"]
    result = shap.plots.partial_dependence("a", predict, X, feature_names=names, show=False)
    assert isinstance(result, plt.Axes)
    plt.close()
