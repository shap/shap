import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap


@pytest.mark.mpl_image_compare(tolerance=3)
def test_embedding_pca(explainer):
    """Test embedding plot with PCA method."""
    shap_values = explainer.shap_values(explainer.data)
    fig = plt.figure()
    shap.plots.embedding("Age", shap_values, feature_names=explainer.feature_names, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare(tolerance=3)
def test_embedding_rank(explainer):
    """Test embedding plot with rank selection."""
    shap_values = explainer.shap_values(explainer.data)
    fig = plt.figure()
    shap.plots.embedding("rank(0)", shap_values, feature_names=explainer.feature_names, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare(tolerance=3)
def test_embedding_sum(explainer):
    """Test embedding plot with sum of SHAP values."""
    shap_values = explainer.shap_values(explainer.data)
    fig = plt.figure()
    shap.plots.embedding("sum()", shap_values, feature_names=explainer.feature_names, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare(tolerance=3)
def test_embedding_custom_alpha(explainer):
    """Test embedding plot with custom alpha transparency."""
    shap_values = explainer.shap_values(explainer.data)
    fig = plt.figure()
    shap.plots.embedding("Age", shap_values, feature_names=explainer.feature_names, alpha=0.3, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare(tolerance=3)
def test_embedding_custom_method(explainer):
    """Test embedding plot with custom 2D embedding."""
    shap_values = explainer.shap_values(explainer.data)
    # Create a simple custom 2D embedding
    custom_embedding = np.random.RandomState(42).randn(shap_values.shape[0], 2)
    fig = plt.figure()
    shap.plots.embedding("Age", shap_values, feature_names=explainer.feature_names, method=custom_embedding, show=False)
    plt.tight_layout()
    return fig


def test_embedding_no_feature_names(explainer):
    """Test embedding plot without feature names."""
    shap_values = explainer.shap_values(explainer.data)
    shap.plots.embedding(0, shap_values, show=False)
    plt.close()


def test_embedding_show_true(explainer, monkeypatch):
    """Test embedding plot with show=True."""
    shap_values = explainer.shap_values(explainer.data)
    # Mock plt.show() to avoid actually displaying
    show_called = []
    monkeypatch.setattr(plt, 'show', lambda: show_called.append(True))
    shap.plots.embedding("Age", shap_values, feature_names=explainer.feature_names, show=True)
    assert len(show_called) == 1
    plt.close()
