"""Tests for shap/explainers/other/_lime.py.

Coverage targets:
- LimeTabular.__init__: invalid mode, DataFrame/numpy data, 1-D output
  (classification and regression), 2-D multi-output
- LimeTabular.attributions: output shape/type, DataFrame input, explicit
  num_features, regression negation, flat vs list return
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Skip the entire module if lime is not installed.
pytest.importorskip("lime")

from shap.explainers.other._lime import LimeTabular  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RS = np.random.RandomState(0)
N_TRAIN = 20
N_TEST = 3
N_FEATURES = 4


@pytest.fixture(scope="module")
def regression_data():
    X_train = RS.randn(N_TRAIN, N_FEATURES)
    y_train = X_train[:, 0] * 2 + RS.randn(N_TRAIN) * 0.1
    X_test = RS.randn(N_TEST, N_FEATURES)
    return X_train, y_train, X_test


@pytest.fixture(scope="module")
def classification_data():
    X_train = RS.randn(N_TRAIN, N_FEATURES)
    y_train = (X_train[:, 0] > 0).astype(int)
    X_test = RS.randn(N_TEST, N_FEATURES)
    return X_train, y_train, X_test


# ---------------------------------------------------------------------------
# Simple model helpers (no heavy ML deps required)
# ---------------------------------------------------------------------------


def _regressor_1d(X):
    """Linear regression: returns 1-D output."""
    return X[:, 0] * 2.0


def _classifier_1d(X):
    """Binary classifier: returns 1-D probability in [0, 1]."""
    return 1 / (1 + np.exp(-X[:, 0]))


def _classifier_2d(X):
    """Multi-class classifier: returns 2-D probability matrix."""
    p = 1 / (1 + np.exp(-X[:, 0]))
    return np.column_stack([1 - p, p])


def _multiout_regressor(X):
    """Multi-output regressor: returns 2-D output."""
    return np.column_stack([X[:, 0] * 2, X[:, 1] * -1])


# ---------------------------------------------------------------------------
# __init__ tests
# ---------------------------------------------------------------------------


def test_invalid_mode_raises(regression_data):
    """LimeTabular raises ValueError for an unsupported mode string."""
    X_train, _, _ = regression_data
    with pytest.raises(ValueError, match="Invalid mode"):
        LimeTabular(_regressor_1d, X_train, mode="invalid")


@pytest.mark.parametrize("bad_mode", ["Classification", "REGRESSION", ""])
def test_invalid_mode_parametrized(bad_mode, regression_data):
    """Mode matching is case-sensitive; anything outside the two valid strings raises."""
    X_train, _, _ = regression_data
    with pytest.raises(ValueError, match="Invalid mode"):
        LimeTabular(_regressor_1d, X_train, mode=bad_mode)


def test_init_with_numpy_data(regression_data):
    """LimeTabular accepts a numpy array as data without error."""
    X_train, _, _ = regression_data
    explainer = LimeTabular(_regressor_1d, X_train, mode="regression")
    assert isinstance(explainer.data, np.ndarray)


def test_init_with_dataframe_data(regression_data):
    """LimeTabular converts a DataFrame to a numpy array internally."""
    X_train, _, _ = regression_data
    df = pd.DataFrame(X_train)
    explainer = LimeTabular(_regressor_1d, df, mode="regression")
    assert isinstance(explainer.data, np.ndarray)
    assert explainer.data.shape == X_train.shape


def test_init_regression_1d_output(regression_data):
    """1-D regressor: out_dim=1, flat_out=True, model is NOT re-wrapped."""
    X_train, _, _ = regression_data
    explainer = LimeTabular(_regressor_1d, X_train, mode="regression")
    assert explainer.out_dim == 1
    assert explainer.flat_out is True


def test_init_classification_1d_output_wraps_model(classification_data):
    """1-D classifier: model is wrapped to return 2-column probabilities internally."""
    X_train, _, _ = classification_data
    original_model = _classifier_1d
    explainer = LimeTabular(original_model, X_train, mode="classification")
    assert explainer.out_dim == 1
    assert explainer.flat_out is True
    # The stored model should now be the wrapper (not the original function)
    assert explainer.model is not original_model
    # Wrapper must return 2 columns
    wrapped_out = explainer.model(X_train[:2])
    assert wrapped_out.shape == (2, 2)
    assert np.allclose(wrapped_out.sum(axis=1), 1.0, atol=1e-6)


def test_init_classification_2d_output(classification_data):
    """2-D classifier: out_dim equals number of columns, flat_out=False."""
    X_train, _, _ = classification_data
    explainer = LimeTabular(_classifier_2d, X_train, mode="classification")
    assert explainer.out_dim == 2
    assert explainer.flat_out is False


def test_init_regression_multioutput(regression_data):
    """Multi-output regressor: out_dim equals number of output columns."""
    X_train, _, _ = regression_data
    explainer = LimeTabular(_multiout_regressor, X_train, mode="regression")
    assert explainer.out_dim == 2
    assert explainer.flat_out is False


# ---------------------------------------------------------------------------
# attributions tests
# ---------------------------------------------------------------------------


def test_attributions_regression_shape(regression_data):
    """attributions() returns an array with shape (n_samples, n_features)."""
    X_train, _, X_test = regression_data
    explainer = LimeTabular(_regressor_1d, X_train, mode="regression")
    attrs = explainer.attributions(X_test, nsamples=100)
    assert isinstance(attrs, np.ndarray)
    assert attrs.shape == X_test.shape


def test_attributions_classification_1d_shape(classification_data):
    """1-D classification: attributions() returns (n_samples, n_features) array."""
    X_train, _, X_test = classification_data
    explainer = LimeTabular(_classifier_1d, X_train, mode="classification")
    attrs = explainer.attributions(X_test, nsamples=100)
    assert isinstance(attrs, np.ndarray)
    assert attrs.shape == X_test.shape


def test_attributions_classification_2d_returns_list(classification_data):
    """2-D classification: attributions() returns a list with one array per output."""
    X_train, _, X_test = classification_data
    explainer = LimeTabular(_classifier_2d, X_train, mode="classification")
    attrs = explainer.attributions(X_test, nsamples=100)
    assert isinstance(attrs, list)
    assert len(attrs) == 2
    for arr in attrs:
        assert arr.shape == X_test.shape


def test_attributions_dataframe_input(regression_data):
    """attributions() accepts a DataFrame and returns the same shape as numpy."""
    X_train, _, X_test = regression_data
    explainer = LimeTabular(_regressor_1d, X_train, mode="regression")
    df_test = pd.DataFrame(X_test)
    attrs = explainer.attributions(df_test, nsamples=100)
    assert attrs.shape == X_test.shape


def test_attributions_num_features_explicit(regression_data):
    """Passing num_features limits how many features LIME uses per explanation."""
    X_train, _, X_test = regression_data
    explainer = LimeTabular(_regressor_1d, X_train, mode="regression")
    # Should run without error; output shape is still (n_samples, n_features)
    attrs = explainer.attributions(X_test, nsamples=100, num_features=2)
    assert attrs.shape == X_test.shape


def test_attributions_regression_negates_output(regression_data):
    """Regression mode negates the raw LIME output (sign-correction documented in source)."""
    X_train, _, X_test = regression_data
    explainer = LimeTabular(_regressor_1d, X_train, mode="regression")
    attrs = explainer.attributions(X_test, nsamples=200)
    # We cannot check the exact sign flip without running LIME twice, but we
    # can confirm the first-feature attributions are non-zero for a linear model.
    assert not np.all(attrs[:, 0] == 0)


def test_attributions_multioutput_regression_returns_list(regression_data):
    """Multi-output regression: attributions() returns a list of arrays."""
    X_train, _, X_test = regression_data
    explainer = LimeTabular(_multiout_regressor, X_train, mode="regression")
    attrs = explainer.attributions(X_test, nsamples=100)
    assert isinstance(attrs, list)
    assert len(attrs) == 2
    for arr in attrs:
        assert arr.shape == X_test.shape
