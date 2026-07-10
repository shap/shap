import numpy as np
import pytest

import shap


@pytest.fixture
def ebm_classifier_no_interaction():
    interpret = pytest.importorskip("interpret")
    from interpret.glassbox import ExplainableBoostingClassifier

    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    model = ExplainableBoostingClassifier(interactions=0)
    model.fit(X, y)
    return model


@pytest.fixture
def ebm_classifier_with_interaction():
    interpret = pytest.importorskip("interpret")
    from interpret.glassbox import ExplainableBoostingClassifier

    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    model = ExplainableBoostingClassifier(interactions=1)
    model.fit(X, y)
    return model


@pytest.fixture
def ebm_regressor_no_interaction():
    interpret = pytest.importorskip("interpret")
    from interpret.glassbox import ExplainableBoostingRegressor

    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    model = ExplainableBoostingRegressor(interactions=0)
    model.fit(X, y)
    return model


def test_supports_classifier_no_interaction(ebm_classifier_no_interaction):
    assert shap.explainers.AdditiveExplainer.supports_model_with_masker(
        ebm_classifier_no_interaction, None
    )


def test_supports_classifier_with_interaction(ebm_classifier_with_interaction):
    assert not shap.explainers.AdditiveExplainer.supports_model_with_masker(
        ebm_classifier_with_interaction, None
    )


def test_supports_regressor_no_interaction(ebm_regressor_no_interaction):
    assert shap.explainers.AdditiveExplainer.supports_model_with_masker(
        ebm_regressor_no_interaction, None
    )


def test_additive_explainer_with_ebm_regressor(ebm_regressor_no_interaction):
    X = np.array([[1, 2], [3, 4], [5, 6]])
    explainer = shap.AdditiveExplainer(ebm_regressor_no_interaction, shap.maskers.Independent(X))
    shap_values = explainer.shap_values(X)
    assert shap_values.shape == X.shape
