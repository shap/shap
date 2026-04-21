"""Tests for the TreeGain explainer."""

from __future__ import annotations

import numpy as np
import pytest

import shap.explainers.other._treegain as treegain_module
from shap.explainers.other._treegain import TreeGain


class _ModelWithImportances:
    def __init__(self):
        self.feature_importances_ = np.array([0.2, 0.8])


class _ModelWithoutImportances:
    pass


@pytest.mark.parametrize(
    "supported_type_suffix",
    [
        "sklearn.tree.tree.DecisionTreeRegressor'>",
        "sklearn.tree.tree.DecisionTreeClassifier'>",
        "sklearn.ensemble.forest.RandomForestRegressor'>",
        "sklearn.ensemble.forest.RandomForestClassifier'>",
        "xgboost.sklearn.XGBRegressor'>",
        "xgboost.sklearn.XGBClassifier'>",
    ],
)
def test_treegain_accepts_supported_model_type_strings(monkeypatch, supported_type_suffix):
    model = _ModelWithImportances()

    monkeypatch.setattr(treegain_module, "str", lambda _x: f"<class '{supported_type_suffix}", raising=False)

    explainer = TreeGain(model)
    assert explainer.model is model


def test_treegain_raises_not_implemented_for_unsupported_model():
    with pytest.raises(NotImplementedError, match="not yet supported by TreeGainExplainer"):
        TreeGain(object())


def test_treegain_requires_feature_importances(monkeypatch):
    monkeypatch.setattr(
        treegain_module,
        "str",
        lambda _x: "<class 'sklearn.tree.tree.DecisionTreeRegressor'>",
        raising=False,
    )

    with pytest.raises(AssertionError, match="does not have a feature_importances_ attribute"):
        TreeGain(_ModelWithoutImportances())


def test_treegain_attributions_tile_global_importances(monkeypatch):
    monkeypatch.setattr(
        treegain_module,
        "str",
        lambda _x: "<class 'sklearn.tree.tree.DecisionTreeRegressor'>",
        raising=False,
    )
    model = _ModelWithImportances()
    explainer = TreeGain(model)

    X = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    attributions = explainer.attributions(X)

    expected = np.tile(model.feature_importances_, (X.shape[0], 1))
    np.testing.assert_allclose(attributions, expected)
