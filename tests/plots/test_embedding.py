import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap


@pytest.mark.mpl_image_compare
def test_embedding_pca(explainer):
    """Test PCA embedding"""
    fig = plt.figure()
    shap_values = explainer(explainer.data).values
    shap.plots.embedding(0, shap_values, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_embedding_sum(explainer):
    """Test the sum() index"""
    fig = plt.figure()
    shap_values = explainer(explainer.data).values
    shap.plots.embedding("sum()", shap_values, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_embedding_custom_method():
    """Test for computing the embedding"""
    fig = plt.figure()
    shap_values = np.random.randn(100, 5)
    custom_emb = np.random.randn(100, 2)

    shap.plots.embedding(0, shap_values, method=custom_emb, show=False)
    plt.tight_layout()
    return fig


def test_embedding_invalid_method():
    """Test for invalid input method"""
    shap_values = np.random.randn(10, 5)

    with pytest.raises(ValueError, match="Unsupported embedding method"):
        shap.plots.embedding(0, shap_values, method="invalid_method", show=False)
