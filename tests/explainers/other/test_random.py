import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

import shap
from shap.explainers.other import Random


@pytest.fixture
def basic_model_data():
    """Fixture to provide standard model and data for tests."""
    np.random.seed(42)
    X = np.random.randn(20, 3)
    y = X[:, 0] * 2 + X[:, 1] + np.random.randn(20) * 0.1
    model = LinearRegression().fit(X, y)
    masker = shap.maskers.Independent(X)
    return model, masker, X


def test_random_explainer_basic_api(basic_model_data):
    """Test that the explainer works through the standard public API."""
    model, masker, X = basic_model_data

    # Initialize and explain
    explainer = Random(model.predict, masker)
    shap_values = explainer(X[:2])

    # Assert shapes match the input (2 samples, 3 features)
    assert shap_values.values.shape == (2, 3)
    assert shap_values.base_values.shape == (2,)

    # Assert values are the randomly generated small numbers (line 48)
    assert np.all(np.abs(shap_values.values) < 0.1)


def test_call_args_storage(basic_model_data):
    """Test that extra kwargs are correctly stored in __call__.__kwdefaults__"""
    model, masker, X = basic_model_data

    # Pass a random argument like max_evals=500
    explainer = Random(model.predict, masker, max_evals=500)

    # Verify lines 24-25 executed correctly
    assert explainer.__call__.__kwdefaults__["max_evals"] == 500


def test_clustering_ndarray(basic_model_data):
    """Test the path where masker.clustering is a numpy array."""
    model, masker, X = basic_model_data

    # Force clustering to be a numpy array
    fake_cluster = np.array([[0, 1, 0.5, 2]])
    masker.clustering = fake_cluster

    explainer = Random(model.predict, masker)
    shap_values = explainer(X[:1])

    # Verify lines 35-36 executed correctly
    np.testing.assert_array_equal(shap_values.clustering[0], fake_cluster)


def test_clustering_callable(basic_model_data):
    """Test the path where masker.clustering is a callable function."""
    model, masker, X = basic_model_data

    # Force clustering to be a function
    fake_cluster = np.array([[0, 1, 0.5, 2]])
    masker.clustering = lambda *args: fake_cluster

    explainer = Random(model.predict, masker)
    shap_values = explainer(X[:1])

    # Verify lines 37-38 executed correctly
    np.testing.assert_array_equal(shap_values.clustering[0], fake_cluster)


def test_clustering_unsupported_error(basic_model_data):
    """Test that an invalid clustering type raises the correct error."""
    model, masker, X = basic_model_data

    # Force clustering to be an unsupported type (a string)
    masker.clustering = "invalid_clustering_format"

    explainer = Random(model.predict, masker)

    # Verify lines 39-41 raise the expected NotImplementedError
    with pytest.raises(NotImplementedError, match="not yet supported"):
        explainer(X[:1])
