# tests/benchmark/test_methods.py

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

import shap.benchmark.methods as methods

# =========================
# Fixtures
# =========================


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=25, n_features=5, random_state=0)
    return X, y


@pytest.fixture
def linear_model(regression_data):
    X, y = regression_data
    return LinearRegression().fit(X, y)


@pytest.fixture
def tree_model(regression_data):
    X, y = regression_data
    return RandomForestRegressor(n_estimators=5, random_state=0).fit(X, y)


# =========================
# REAL TESTS (CORE EXPLAINERS)
# =========================


def test_linear_shap_corr_real(linear_model, regression_data):
    X, _ = regression_data
    f = methods.linear_shap_corr(linear_model, X)
    out = f(X)

    assert isinstance(out, np.ndarray)
    assert out.shape == X.shape


def test_linear_shap_ind_real(linear_model, regression_data):
    X, _ = regression_data
    f = methods.linear_shap_ind(linear_model, X)
    out = f(X)

    assert isinstance(out, np.ndarray)
    assert out.shape == X.shape


def test_tree_shap_tree_path_dependent_real(tree_model, regression_data):
    X, _ = regression_data
    f = methods.tree_shap_tree_path_dependent(tree_model, X)
    out = f(X)

    assert isinstance(out, np.ndarray)
    assert out.shape == X.shape


def test_tree_shap_independent_real(tree_model, regression_data):
    X, _ = regression_data
    f = methods.tree_shap_independent_200(tree_model, X)
    out = f(X)

    assert isinstance(out, np.ndarray)
    assert out.shape == X.shape


def test_mean_abs_tree_shap_real(tree_model, regression_data):
    X, _ = regression_data
    f = methods.mean_abs_tree_shap(tree_model, X)
    out = f(X)

    assert isinstance(out, np.ndarray)
    assert out.shape == X.shape
    assert np.all(out >= 0)


def test_kernel_shap_real(linear_model, regression_data):
    X, _ = regression_data

    background = X[:10]
    test_data = X[:5]

    f = methods.kernel_shap_1000_meanref(linear_model, background)
    out = f(test_data)

    assert isinstance(out, np.ndarray)
    assert out.shape == test_data.shape


def test_sampling_shap_real(linear_model, regression_data):
    X, _ = regression_data

    f = methods.sampling_shap_1000(linear_model, X)
    out = f(X[:5])

    assert isinstance(out, np.ndarray)
    assert out.shape == (5, X.shape[1])


# =========================
# CONDITIONAL TESTS (NO MOCKS)
# =========================


def test_deep_shap_conditional():
    pytest.importorskip("tensorflow")

    X = np.random.randn(10, 5)

    # Dummy model (only structure, not deep model)
    model = lambda x: x

    f = methods.deep_shap(model, X)
    out = f(X)

    assert out is not None


def test_expected_gradients_conditional():
    pytest.importorskip("tensorflow")

    X = np.random.randn(10, 5)
    model = lambda x: x

    f = methods.expected_gradients(model, X)
    out = f(X)

    assert out is not None


# =========================
# MOCKED TESTS (NON-CORE)
# =========================


@patch.object(methods.other, "CoefficentExplainer", create=True)
def test_coef_mock(mock_explainer):
    mock_instance = MagicMock()
    mock_instance.attributions = np.ones((5, 3))
    mock_explainer.return_value = mock_instance

    result = methods.coef("model", None)
    assert result.shape == (5, 3)


@patch.object(methods.other, "RandomExplainer", create=True)
def test_random_mock(mock_explainer):
    mock_instance = MagicMock()
    mock_instance.attributions = np.ones((5, 3))
    mock_explainer.return_value = mock_instance

    result = methods.random("model", None)
    assert result.shape == (5, 3)


@patch.object(methods.other, "TreeGainExplainer", create=True)
def test_tree_gain_mock(mock_explainer):
    mock_instance = MagicMock()
    mock_instance.attributions = np.ones((5, 3))
    mock_explainer.return_value = mock_instance

    result = methods.tree_gain("model", None)
    assert result.shape == (5, 3)


@patch.object(methods.other, "LimeTabularExplainer", create=True)
def test_lime_regression_mock(mock_explainer):
    mock_instance = MagicMock()
    mock_instance.attributions.return_value = np.ones((5, 3))
    mock_explainer.return_value = mock_instance

    model = MagicMock()
    model.predict = MagicMock()

    f = methods.lime_tabular_regression_1000(model, np.random.randn(10, 3))
    out = f(np.random.randn(5, 3))

    assert out.shape == (5, 3)


@patch.object(methods.other, "LimeTabularExplainer", create=True)
def test_lime_classification_mock(mock_explainer):
    mock_instance = MagicMock()
    mock_instance.attributions.return_value = [None, np.ones((5, 3))]
    mock_explainer.return_value = mock_instance

    model = MagicMock()
    model.predict_proba = MagicMock()

    f = methods.lime_tabular_classification_1000(model, np.random.randn(10, 3))
    out = f(np.random.randn(5, 3))

    assert out.shape == (5, 3)


@patch.object(methods.other, "MapleExplainer", create=True)
def test_maple_mock(mock_explainer):
    mock_instance = MagicMock()
    mock_instance.attributions.return_value = np.ones((5, 3))
    mock_explainer.return_value = mock_instance

    model = MagicMock()
    model.predict = MagicMock()

    f = methods.maple(model, np.random.randn(10, 3))
    out = f(np.random.randn(5, 3))

    assert out.shape == (5, 3)


@patch.object(methods.other, "TreeMapleExplainer", create=True)
def test_tree_maple_mock(mock_explainer):
    mock_instance = MagicMock()
    mock_instance.attributions.return_value = np.ones((5, 3))
    mock_explainer.return_value = mock_instance

    model = MagicMock()

    f = methods.tree_maple(model, np.random.randn(10, 3))
    out = f(np.random.randn(5, 3))

    assert out.shape == (5, 3)
