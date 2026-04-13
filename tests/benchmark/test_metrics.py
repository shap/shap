import numpy as np
import pytest

from sklearn.linear_model import LinearRegression

import shap.benchmark.metrics as metrics


@pytest.fixture
def data():
    X = np.random.randn(200, 4)  # IMPORTANT FIX
    y = X @ np.array([1.0, -2.0, 0.5, 1.5]) + 0.1 * np.random.randn(200)
    return X, y


@pytest.fixture
def model_gen():
    def f():
        return LinearRegression()
    return f


# =========================
# Core metrics
# =========================

def test_consistency_guarantees(data, model_gen):
    X, y = data

    _, val = metrics.consistency_guarantees(
        X, y, model_gen, "linear_shap_corr"
    )

    assert val == 1.0


def test_runtime(data, model_gen):
    X, y = data

    _, val = metrics.runtime(
        X, y, model_gen, "linear_shap_ind"
    )

    assert isinstance(val, float)
    assert val >= 0


def test_local_accuracy(data, model_gen):
    X, y = data

    _, val = metrics.local_accuracy(
        X, y, model_gen, "linear_shap_ind"
    )

    assert isinstance(val, float)


# =========================
# Wrapper tests
# =========================

def test_keep_positive_mask(data, model_gen):
    X, y = data

    fcounts, scores = metrics.keep_positive_mask(
        X, y, model_gen, "linear_shap_ind"
    )

    assert len(fcounts) > 0
    assert scores is not None


def test_remove_negative_mask(data, model_gen):
    X, y = data

    fcounts, scores = metrics.remove_negative_mask(
        X, y, model_gen, "linear_shap_ind"
    )

    assert len(fcounts) > 0
    assert scores is not None