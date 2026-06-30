"""Tests for shap.plots.embedding."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap


@pytest.fixture()
def shap_values_2d():
    """Simple 2D SHAP values array for embedding tests."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((50, 5))


def test_embedding_pca(shap_values_2d):
    """Embedding plot with method='pca' should not raise."""
    shap.plots.embedding(0, shap_values_2d, show=False)
    plt.close("all")


def test_embedding_numpy_array(shap_values_2d):
    """Embedding plot with a custom numpy embedding array should not raise."""
    rng = np.random.default_rng(1)
    embedding = rng.standard_normal((shap_values_2d.shape[0], 2))
    shap.plots.embedding(0, shap_values_2d, method=embedding, show=False)
    plt.close("all")


def test_embedding_unsupported_method_raises(shap_values_2d):
    """Unsupported method argument should raise ValueError instead of UnboundLocalError (closes #4394)."""
    with pytest.raises(ValueError, match="Unsupported embedding method"):
        shap.plots.embedding(0, shap_values_2d, method="tsne", show=False)
    plt.close("all")
