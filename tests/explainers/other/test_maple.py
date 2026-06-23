import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

import shap.explainers.other._maple as maple_mod
from shap.explainers.other._maple import MAPLE


def test_maple_small():
    """Test a small MAPLE model."""
    rs = np.random.RandomState(0)
    X_train = rs.randn(10, 4)
    y_train = X_train[:, 0] * 2 + rs.randn(10) * 0.1
    X_val = rs.randn(5, 4)
    y_val = X_val[:, 0] * 2 + rs.randn(5) * 0.1

    maple = MAPLE(X_train, y_train, X_val, y_val)

    preds = maple.predict(X_val)
    assert preds.shape == (5,)


def test_maple_wrapper_with_dataframe_and_multiply(monkeypatch):
    backend_calls = []

    class DummyBackend:
        def __init__(self, *args, **kwargs):
            backend_calls.append((args, kwargs))

        def explain(self, x):
            return {"coefs": np.array([0.0, 1.0, -2.0])}

    monkeypatch.setattr(maple_mod, "MAPLE", DummyBackend)

    data_df = pd.DataFrame(np.array([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0], [4.0, 6.0], [5.0, 7.0]]), columns=["a", "b"])
    explainer = maple_mod.Maple(lambda X: X[:, 0] + X[:, 1], data_df)

    attrs = explainer.attributions(data_df.iloc[:2], multiply_by_input=True)
    expected = np.vstack(
        [
            np.array([1.0, -2.0]) * (data_df.values[0] - explainer.data_mean),
            np.array([1.0, -2.0]) * (data_df.values[1] - explainer.data_mean),
        ]
    )

    assert backend_calls
    assert attrs.shape == (2, 2)
    np.testing.assert_allclose(attrs, expected)


def test_maple_wrapper_multioutput_returns_list(monkeypatch):
    class DummyBackend:
        def __init__(self, *args, **kwargs):
            pass

        def explain(self, x):
            return {"coefs": np.array([0.0, 5.0, 6.0])}

    monkeypatch.setattr(maple_mod, "MAPLE", DummyBackend)

    data = np.array([[1.0, 2.0], [3.0, 4.0], [0.5, 0.25], [1.5, 3.5], [2.5, 1.0]])
    explainer = maple_mod.Maple(lambda X: np.column_stack([X[:, 0], X[:, 1]]), data)

    attrs = explainer.attributions(data[:1], multiply_by_input=False)

    assert isinstance(attrs, list)
    assert len(attrs) == 2
    np.testing.assert_allclose(attrs[0], np.array([[5.0, 6.0]]))
    np.testing.assert_allclose(attrs[1], np.zeros((1, 2)))


def test_treemaple_rf_wrapper_and_attributions(monkeypatch):
    backend_calls = []

    class DummyBackend:
        def __init__(self, *args, **kwargs):
            backend_calls.append((args, kwargs))

        def explain(self, x):
            return {"coefs": np.array([0.0, 1.0, 3.0])}

    class FakeRFModel:
        def predict(self, X):
            return X[:, 0] * 2.0

    FakeRFModel.__module__ = "sklearn.ensemble.forest"
    FakeRFModel.__qualname__ = "RandomForestRegressor"

    monkeypatch.setattr(maple_mod, "MAPLE", DummyBackend)

    model = FakeRFModel()
    data_df = pd.DataFrame(np.array([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]]), columns=["a", "b"])
    explainer = maple_mod.TreeMaple(model, data_df)

    attrs = explainer.attributions(data_df.iloc[:2], multiply_by_input=True)
    expected = np.vstack(
        [
            np.array([1.0, 3.0]) * (data_df.values[0] - explainer.data_mean),
            np.array([1.0, 3.0]) * (data_df.values[1] - explainer.data_mean),
        ]
    )

    assert backend_calls
    assert backend_calls[0][1]["fe"] is model
    assert backend_calls[0][1]["fe_type"] == "rf"
    np.testing.assert_allclose(attrs, expected)


def test_treemaple_gbdt_multioutput_and_unsupported_model(monkeypatch):
    backend_calls = []

    class DummyBackend:
        def __init__(self, *args, **kwargs):
            backend_calls.append((args, kwargs))

        def explain(self, x):
            return {"coefs": np.array([0.0, 2.0, -1.0])}

    class FakeGBDTModel:
        def predict(self, X):
            return np.column_stack([X[:, 0], X[:, 1]])

    FakeGBDTModel.__module__ = "sklearn.ensemble.gradient_boosting"
    FakeGBDTModel.__qualname__ = "GradientBoostingRegressor"

    monkeypatch.setattr(maple_mod, "MAPLE", DummyBackend)

    data = np.array([[1.0, 2.0], [0.5, 0.25], [3.0, 4.0]])
    explainer = maple_mod.TreeMaple(FakeGBDTModel(), data)
    attrs = explainer.attributions(data[:1], multiply_by_input=False)

    assert backend_calls
    assert backend_calls[0][1]["fe_type"] == "gbdt"
    assert isinstance(attrs, list)
    assert len(attrs) == 2
    np.testing.assert_allclose(attrs[0], np.array([[2.0, -1.0]]))
    np.testing.assert_allclose(attrs[1], np.zeros((1, 2)))

    with pytest.raises(NotImplementedError, match="not yet supported"):
        maple_mod.TreeMaple(object(), data)


def test_maple_gbrt_predict_fe_and_predict_silo():
    rs = np.random.RandomState(1)
    X_train = rs.randn(12, 3)
    y_train = X_train[:, 0] * 1.5 - X_train[:, 1] * 0.7 + rs.randn(12) * 0.01
    X_val = rs.randn(5, 3)
    y_val = X_val[:, 0] * 1.5 - X_val[:, 1] * 0.7 + rs.randn(5) * 0.01

    maple = MAPLE(
        X_train,
        y_train,
        X_val,
        y_val,
        fe_type="gbrt",
        n_estimators=3,
        max_features=1.0,
        min_samples_leaf=1,
    )

    pred_fe = maple.predict_fe(X_val[:3])
    pred_silo = maple.predict_silo(X_val[:3])

    assert pred_fe.shape == (3,)
    assert pred_silo.shape == (3,)


def test_maple_with_provided_fe_and_unknown_type_exit():
    rs = np.random.RandomState(2)
    X_train = rs.randn(10, 3)
    y_train = X_train[:, 0] * 2 + rs.randn(10) * 0.05
    X_val = rs.randn(4, 3)
    y_val = X_val[:, 0] * 2 + rs.randn(4) * 0.05

    fe = RandomForestRegressor(n_estimators=4, random_state=0, min_samples_leaf=1, max_features=1.0)
    fe.fit(X_train, y_train)

    maple = MAPLE(X_train, y_train, X_val, y_val, fe_type="rf", fe=fe, n_estimators=1)
    assert maple.n_estimators == len(fe.estimators_)

    with pytest.raises(SystemExit):
        MAPLE(X_train, y_train, X_val, y_val, fe_type="unknown", n_estimators=1)
