import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from shap.explainers.other._random import Random
from shap.maskers import Independent


@pytest.fixture
def model_and_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 4))
    y = X[:, 0] + X[:, 1]
    model = LinearRegression().fit(X, y)
    return model, X


def test_random_explainer_init(model_and_data):
    model, X = model_and_data
    masker = Independent(X, max_samples=50)
    explainer = Random(model.predict, masker)
    assert explainer is not None


def test_random_explainer_output_shape(model_and_data):
    model, X = model_and_data
    masker = Independent(X, max_samples=50)
    explainer = Random(model.predict, masker)
    shap_values = explainer(X[:5])
    assert shap_values.values.shape == (5, 4)


def test_random_explainer_constant_false(model_and_data):
    model, X = model_and_data
    masker = Independent(X, max_samples=50)
    explainer = Random(model.predict, masker, constant=False)
    shap_values1 = explainer(X[:3])
    shap_values2 = explainer(X[:3])
    assert not np.allclose(shap_values1.values, shap_values2.values)


def test_random_explainer_values_small(model_and_data):
    model, X = model_and_data
    masker = Independent(X, max_samples=50)
    explainer = Random(model.predict, masker)
    shap_values = explainer(X[:5])
    assert np.all(np.abs(shap_values.values) < 1)


def test_random_explainer_with_call_args(model_and_data):
    model, X = model_and_data
    masker = Independent(X, max_samples=50)
    explainer = Random(model.predict, masker, max_evals=100)
    assert explainer is not None


def test_random_explainer_with_clustering(model_and_data):
    model, X = model_and_data
    masker = Independent(X, max_samples=50)
    clustering = np.array([[0, 1, 0.5, 2],
                           [2, 3, 0.8, 2],
                           [4, 5, 1.2, 2]])
    masker.clustering = clustering
    explainer = Random(model.predict, masker)
    shap_values = explainer(X[:3])
    assert shap_values.values.shape == (3, 4)


def test_random_explainer_clustering_callable(model_and_data):
    model, X = model_and_data
    masker = Independent(X, max_samples=50)
    masker.clustering = lambda *args: np.array([[0, 1, 0.5, 2],
                                                [2, 3, 0.8, 2],
                                                [4, 5, 1.2, 2]])
    explainer = Random(model.predict, masker)
    shap_values = explainer(X[:3])
    assert shap_values.values.shape == (3, 4)


def test_random_explainer_clustering_not_implemented(model_and_data):
    model, X = model_and_data
    masker = Independent(X, max_samples=50)
    masker.clustering = "invalid"
    explainer = Random(model.predict, masker)
    with pytest.raises(NotImplementedError):
        explainer(X[:3])