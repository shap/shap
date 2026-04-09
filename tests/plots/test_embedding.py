import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap


@pytest.fixture()
def embedding_data(explainer):
    """Return shap values array and feature names for embedding tests."""
    explanation = explainer(explainer.data)
    shap_values = explanation.values
    feature_names = explainer.data_feature_names
    return shap_values, feature_names


@pytest.mark.mpl_image_compare
def test_embedding(embedding_data):
    """Check that the embedding plot is unchanged."""
    np.random.seed(0)
    shap_values, feature_names = embedding_data
    fig, ax = plt.subplots()
    shap.plots.embedding("rank(0)", shap_values, feature_names=feature_names, show=False, ax=ax)
    return fig


def test_embedding_returns_ax(embedding_data):
    """embedding() should always return an Axes object."""
    np.random.seed(0)
    shap_values, feature_names = embedding_data
    result = shap.plots.embedding("rank(0)", shap_values, feature_names=feature_names, show=False)
    assert isinstance(result, plt.Axes)
    plt.close("all")


def test_embedding_accepts_ax(embedding_data):
    """embedding() should draw into a provided Axes and return it."""
    np.random.seed(0)
    shap_values, feature_names = embedding_data
    fig, ax = plt.subplots()
    result = shap.plots.embedding("rank(0)", shap_values, feature_names=feature_names, show=False, ax=ax)
    assert result is ax
    plt.close(fig)