import numpy as np
import pytest
import sklearn

import shap


# Regression Test
def test_pytree_regression():
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)
    model = sklearn.ensemble.RandomForestRegressor()
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    assert shap_values.shape == X.shape
    assert isinstance(shap_values, np.ndarray)
    assert np.allclose(shap_values.sum(1) + explainer.expected_value, model.predict(X), atol=1e-4)


# Classification Test
def test_pytree_classification():
    iris = sklearn.datasets.load_iris()
    X, y = iris.data, iris.target
    model = sklearn.ensemble.RandomForestClassifier()
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        assert (
            len(shap_values) == model.n_classes_
        ), f"Expected {model.n_classes_} SHAP value sets, got {len(shap_values)}"
    else:
        assert shap_values.shape[0] == X.shape[0], f"Expected {X.shape[0]} SHAP value sets, got {shap_values.shape[0]}"


# Multi-class Classification Test
def test_pytree_multiclass_classification():
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=100, n_features=20, n_classes=3, n_informative=6, random_state=42)
    model = sklearn.ensemble.RandomForestClassifier()
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        assert len(shap_values) == model.n_classes_
    else:
        assert shap_values.shape[0] == X.shape[0]
        assert shap_values.shape[2] == model.n_classes_


# XGBoost Test
def test_pytree_xgboost():
    xgboost = pytest.importorskip("xgboost")
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)
    model = xgboost.XGBRegressor()
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    assert shap_values.shape == X.shape
    assert isinstance(shap_values, np.ndarray)
    assert np.allclose(shap_values.sum(1) + explainer.expected_value, model.predict(X), atol=1e-4)


# LightGBM Test
def test_pytree_lightgbm():
    lightgbm = pytest.importorskip("lightgbm")
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)
    dataset = lightgbm.Dataset(data=X, label=y)
    model = lightgbm.train({"objective": "regression"}, dataset)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    assert shap_values.shape == X.shape
    assert isinstance(shap_values, np.ndarray)
    assert np.allclose(shap_values.sum(1) + explainer.expected_value, model.predict(X, raw_score=True), atol=1e-4)


# Edge Case: Single Instance
def test_pytree_single_instance():
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)
    model = sklearn.ensemble.RandomForestRegressor()
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[0])

    assert shap_values.shape == (X.shape[1],)
    assert isinstance(shap_values, np.ndarray)


# Edge Case: Single Feature
def test_pytree_single_feature():
    X = np.random.randn(500, 1)
    y = np.random.randn(500)
    model = sklearn.ensemble.RandomForestRegressor()
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    assert shap_values.shape == X.shape
    assert isinstance(shap_values, np.ndarray)
    assert np.allclose(shap_values.sum(1) + explainer.expected_value, model.predict(X), atol=1e-4)


# Edge Case: Interaction Values
def test_pytree_interaction_values():
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)
    model = sklearn.ensemble.RandomForestRegressor()
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    interaction_values = explainer.shap_interaction_values(X)

    assert interaction_values.shape == (X.shape[0], X.shape[1], X.shape[1])
    assert isinstance(interaction_values, np.ndarray)
    assert np.allclose(interaction_values.sum(2).sum(1) + explainer.expected_value, model.predict(X), atol=1e-4)


# Test with Missing Values
def test_pytree_with_missing_values():
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)
    X[0, 0] = np.nan  # introduce missing value
    model = sklearn.ensemble.RandomForestRegressor()
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    assert shap_values.shape == X.shape
    assert isinstance(shap_values, np.ndarray)


# Invalid Model Test
def test_pytree_invalid_model():
    class InvalidModel:
        pass

    with pytest.raises(Exception, match="Model type not yet supported by TreeExplainer"):
        _ = shap.TreeExplainer(InvalidModel())


if __name__ == "__main__":
    pytest.main([__file__])
