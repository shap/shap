import numpy as np
import pytest

import shap


@pytest.fixture
def shap_values():
    """Create a simple SHAP values matrix for testing."""
    return np.random.randn(50, 5)


@pytest.fixture
def custom_embedding():
    """Create a custom 2D embedding array for testing."""
    return np.random.rand(50, 2)


def test_embedding_integer_index(shap_values):
    """Test embedding with integer index colors by that feature."""
    shap.plots.embedding(ind=0, shap_values=shap_values, show=False)


def test_embedding_sum_index(shap_values):
    """Test embedding with sum() colors by sum of all SHAP values."""
    shap.plots.embedding(ind="sum()", shap_values=shap_values, show=False)


def test_embedding_pca_method(shap_values):
    """Test embedding uses PCA to compute 2D positions."""
    shap.plots.embedding(ind=0, shap_values=shap_values, method="pca", show=False)


def test_embedding_custom_method(shap_values, custom_embedding):
    """Test embedding accepts a custom 2D numpy array for positioning."""
    shap.plots.embedding(ind=0, shap_values=shap_values, method=custom_embedding, show=False)


def test_embedding_feature_names_none(shap_values):
    """Test embedding auto-generates feature names when None."""
    shap.plots.embedding(ind=0, shap_values=shap_values, feature_names=None, show=False)


def test_embedding_feature_names(shap_values):
    """Test embedding works with custom feature names."""
    feature_names = ["feature_a", "feature_b", "feature_c", "feature_d", "feature_e"]
    shap.plots.embedding(ind=0, shap_values=shap_values, feature_names=feature_names, show=False)


def test_embedding_alpha(shap_values):
    """Test embedding works with custom alpha value."""
    shap.plots.embedding(ind=0, shap_values=shap_values, alpha=0.5, show=False)


def test_embedding_show(shap_values):
    """Test embedding with show=True executes without error."""
    shap.plots.embedding(ind=0, shap_values=shap_values, show=True)


def test_embedding_unsupported_method(shap_values):
    """Test embedding with unsupported method crashes with UnboundLocalError.

    This documents the current buggy behaviour. See issue #4394 for the fix.
    """
    with pytest.raises(UnboundLocalError):
        shap.plots.embedding(ind=0, shap_values=shap_values, method="unsupported", show=False)
