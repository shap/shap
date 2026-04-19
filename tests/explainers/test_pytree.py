import sys
import types

import numpy as np
import sklearn.ensemble

from shap.explainers.pytree import TreeExplainer as PyTreeExplainer


def test_pytree_supports_modern_random_forest():
    X = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
        ]
    )
    y = np.array([0.5, 1.5, 2.5, 3.5])

    model = sklearn.ensemble.RandomForestRegressor(n_estimators=4, random_state=0)
    model.fit(X, y)

    explainer = PyTreeExplainer(model)
    shap_values = explainer.shap_values(X[0])

    assert shap_values.shape == (X.shape[1] + 1,)


def test_pytree_uses_iteration_range_for_xgboost(monkeypatch):
    xgboost = types.ModuleType("xgboost")
    xgboost_core = types.ModuleType("xgboost.core")

    class DMatrix:
        def __init__(self, data):
            self.data = np.asarray(data)

    class Booster:
        def __init__(self):
            self.calls = []

        def predict(self, data, **kwargs):
            self.calls.append((data, kwargs))
            assert "ntree_limit" not in kwargs
            assert "iteration_range" in kwargs
            if kwargs.get("pred_contribs"):
                return np.ones((data.data.shape[0], data.data.shape[1] + 1))
            if kwargs.get("pred_interactions"):
                n_features = data.data.shape[1] + 1
                return np.ones((data.data.shape[0], n_features, n_features))
            raise AssertionError("Unexpected predict call")

    xgboost.DMatrix = DMatrix
    xgboost_core.Booster = Booster
    xgboost.core = xgboost_core

    monkeypatch.setitem(sys.modules, "xgboost", xgboost)
    monkeypatch.setitem(sys.modules, "xgboost.core", xgboost_core)

    booster = Booster()
    explainer = PyTreeExplainer(booster)

    contribs = explainer.shap_values([[1.0, 2.0, 3.0]], tree_limit=5)
    interactions = explainer.shap_interaction_values([[1.0, 2.0, 3.0]], tree_limit=6)

    assert contribs.shape == (1, 4)
    assert interactions.shape == (1, 4, 4)
    assert len(booster.calls) == 2
    assert booster.calls[0][1]["iteration_range"] == (0, 5)
    assert booster.calls[1][1]["iteration_range"] == (0, 6)
    assert isinstance(booster.calls[0][0], DMatrix)
    assert isinstance(booster.calls[1][0], DMatrix)
