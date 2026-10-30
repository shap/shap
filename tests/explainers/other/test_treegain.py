import numpy as np
import pytest

from shap.explainers.other import TreeGain


def test_treegain_xgbregressor():
    pytest.importorskip("xgboost")
    import xgboost

    # Train a simple model
    X = np.random.randn(10, 3)
    y = np.random.randn(10)
    model = xgboost.XGBRegressor(n_estimators=10)
    model.fit(X, y)

    # Check that TreeGain can explain it
    explainer = TreeGain(model)
    attributions = explainer.attributions(X)

    assert isinstance(attributions, np.ndarray)
    assert attributions.shape == (10, 3)
    # attributions should be tiled feature_importances_
    np.testing.assert_allclose(attributions[0], model.feature_importances_)
    np.testing.assert_allclose(attributions[-1], model.feature_importances_)


def test_treegain_unsupported_model():
    class UnsupportedModel:
        pass

    model = UnsupportedModel()
    with pytest.raises(NotImplementedError, match="The passed model is not yet supported by TreeGainExplainer"):
        TreeGain(model)


def test_treegain_missing_feature_importances():
    pytest.importorskip("xgboost")
    import xgboost

    # Unfitted model lacks feature_importances_
    model = xgboost.XGBRegressor()

    with pytest.raises(AssertionError, match="The passed model does not have a feature_importances_ attribute"):
        TreeGain(model)
