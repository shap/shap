"""This file contains tests for the embedding plot."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap


@pytest.mark.mpl_image_compare
def test_embedding(explainer):
    """Check that the embedding plot is unchanged."""
    shap_values = explainer(explainer.data)

    # Avoid PCA in mpl image tests for stability across sklearn versions.
    rs = np.random.RandomState(0)
    embedding_values = rs.normal(size=(shap_values.shape[0], 2))

    fig = plt.figure()
    shap.plots.embedding(0, shap_values, method=embedding_values, show=False)
    plt.tight_layout()
    return fig


def test_embedding_returns_ax(explainer):
    """Check that the embedding plot returns the provided axis when show=False."""
    shap_values = explainer(explainer.data)
    rs = np.random.RandomState(0)
    embedding_values = rs.normal(size=(shap_values.shape[0], 2))

    _, ax = plt.subplots()
    ax_out = shap.plots.embedding(0, shap_values, method=embedding_values, show=False, ax=ax)
    assert ax_out is ax


def test_embedding_raises_on_invalid_input_type():
    """Check that a TypeError is raised when shap_values is not an Explanation."""
    with pytest.raises(TypeError, match="shap_values parameter must be a shap\\.Explanation"):
        shap.plots.embedding(0, np.zeros((5, 3)))


def test_embedding_raises_on_invalid_method_shape(explainer):
    """Check that invalid method arrays raise a ValueError."""
    shap_values = explainer(explainer.data)
    with pytest.raises(ValueError, match="Unsupported embedding method"):
        shap.plots.embedding(0, shap_values, method=np.zeros((shap_values.shape[0], 3)), show=False)
