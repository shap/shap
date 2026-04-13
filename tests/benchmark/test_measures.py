import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import shap.benchmark.measures as measures

# =========================
# Fixtures
# =========================


@pytest.fixture
def data():
    X_train = np.random.randn(100, 5)
    X_test = np.random.randn(20, 5)
    y_train = np.random.randn(100)
    y_test = np.random.randn(20)

    # fake attributions (no need for SHAP here)
    attr_test = np.random.randn(20, 5)

    return X_train, X_test, y_train, y_test, attr_test


@pytest.fixture
def model(data):
    X_train, _, y_train, _, _ = data
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


@pytest.fixture
def metric():
    return lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2)


# =========================
# Helper functions
# =========================


def test_to_array():
    df = pd.DataFrame([[1, 2], [3, 4]])
    out = measures.to_array(df)[0]
    assert isinstance(out, np.ndarray)


def test_const_rand():
    a = measures.const_rand(5)
    b = measures.const_rand(5)
    assert np.allclose(a, b)  # deterministic


def test_strip_list():
    arr = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    out = measures.strip_list(arr)
    assert isinstance(out, np.ndarray)


# =========================
# Core masking logic
# =========================


def test_remove_mask(data, model, metric):
    X_train, X_test, y_train, y_test, attr_test = data

    nmask = np.random.randint(0, 5, size=len(y_test))

    val = measures.remove_mask(
        nmask,
        X_train,
        y_train,
        X_test,
        y_test,
        attr_test,
        model_generator=lambda: LinearRegression(),
        metric=metric,
        trained_model=model,
        random_state=0,
    )

    assert isinstance(val, float)


def test_keep_mask(data, model, metric):
    X_train, X_test, y_train, y_test, attr_test = data

    nkeep = np.random.randint(0, 5, size=len(y_test))

    val = measures.keep_mask(
        nkeep,
        X_train,
        y_train,
        X_test,
        y_test,
        attr_test,
        model_generator=lambda: LinearRegression(),
        metric=metric,
        trained_model=model,
        random_state=0,
    )

    assert isinstance(val, float)


# =========================
# Advanced (imputation)
# =========================


def test_remove_impute(data, model, metric):
    X_train, X_test, y_train, y_test, attr_test = data

    nmask = np.random.randint(0, 5, size=len(y_test))

    val = measures.remove_impute(
        nmask,
        X_train,
        y_train,
        X_test,
        y_test,
        attr_test,
        model_generator=lambda: LinearRegression(),
        metric=metric,
        trained_model=model,
        random_state=0,
    )

    assert isinstance(val, float)


def test_keep_impute(data, model, metric):
    X_train, X_test, y_train, y_test, attr_test = data

    nkeep = np.random.randint(0, 5, size=len(y_test))

    val = measures.keep_impute(
        nkeep,
        X_train,
        y_train,
        X_test,
        y_test,
        attr_test,
        model_generator=lambda: LinearRegression(),
        metric=metric,
        trained_model=model,
        random_state=0,
    )

    assert isinstance(val, float)


# =========================
# Optional: resampling
# =========================


def test_remove_resample(data, model, metric):
    X_train, X_test, y_train, y_test, attr_test = data

    nmask = np.random.randint(0, 5, size=len(y_test))

    val = measures.remove_resample(
        nmask,
        X_train,
        y_train,
        X_test,
        y_test,
        attr_test,
        model_generator=lambda: LinearRegression(),
        metric=metric,
        trained_model=model,
        random_state=0,
    )

    assert isinstance(val, float)
