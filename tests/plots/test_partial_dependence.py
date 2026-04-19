import matplotlib

matplotlib.use("Agg")  # non-interactive backend, important for testing

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

import shap
from shap.plots._partial_dependence import compute_bounds, partial_dependence

# Helpers


@pytest.fixture
def regression_data():
    """Simple dataset and trained model for reuse across tests."""
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = X[:, 0] + X[:, 1] ** 2
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    return X, y, model


# Bounds


def test_compute_bounds_with_none():
    xv = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    xmin, xmax = compute_bounds(None, 6.0, xv)
    assert xmin is not None
    assert xmax == 6.0


def test_compute_bounds_with_percentile():
    xv = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    xmin, xmax = compute_bounds("percentile(10)", "percentile(90)", xv)
    assert xmin is not None
    assert xmax is not None
    assert xmin < xmax


def test_compute_bounds_both_none():
    xv = np.array([1.0, 2.0, 3.0])
    xmin, xmax = compute_bounds(None, None, xv)
    assert xmin is None
    assert xmax is None


# 1d tests


def test_basic_1d(regression_data):
    X, y, model = regression_data
    fig, ax = partial_dependence(0, model.predict, X, show=False)
    assert fig is not None
    assert ax is not None


def test_1d_no_ice(regression_data):
    X, y, model = regression_data
    fig, ax = partial_dependence(0, model.predict, X, ice=False, show=False)
    assert fig is not None


def test_1d_no_hist(regression_data):
    X, y, model = regression_data
    fig, ax = partial_dependence(0, model.predict, X, hist=False, show=False)
    assert fig is not None


def test_1d_with_dataframe(regression_data):
    X, y, model = regression_data
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    fig, ax = partial_dependence("a", model.predict, df, show=False)
    assert fig is not None


def test_1d_feature_expected_value(regression_data):
    X, y, model = regression_data
    fig, ax = partial_dependence(0, model.predict, X, feature_expected_value=True, show=False)
    assert fig is not None


def test_1d_model_expected_value(regression_data):
    X, y, model = regression_data
    fig, ax = partial_dependence(0, model.predict, X, model_expected_value=True, show=False)
    assert fig is not None


def test_1d_xmin_xmax_as_numbers(regression_data):
    X, y, model = regression_data
    fig, ax = partial_dependence(0, model.predict, X, xmin=-2.0, xmax=2.0, show=False)
    assert fig is not None


def test_1d_custom_npoints(regression_data):
    X, y, model = regression_data
    fig, ax = partial_dependence(0, model.predict, X, npoints=20, show=False)
    assert fig is not None


def test_1d_custom_ylabel(regression_data):
    X, y, model = regression_data
    fig, ax = partial_dependence(0, model.predict, X, ylabel="my label", show=False)
    assert ax.get_ylabel() == "my label"


@pytest.mark.xfail(reason="known bug: base_values is 2D array, causes matplotlib set_yticks to fail")
def test_1d_with_shap_values(regression_data):
    X, y, model = regression_data
    explainer = shap.Explainer(model.predict, X)
    shap_values = explainer(X)
    fig, ax = partial_dependence(0, model.predict, shap_values, show=False)
    assert fig is not None


# ── partial_dependence 2D tests ───────────────────────────────────────────────


def test_2d_basic(regression_data):
    X, y, model = regression_data
    fig, ax = partial_dependence((0, 1), model.predict, X, npoints=5, show=False)
    assert fig is not None
