import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap


@pytest.mark.mpl_image_compare
def test_embedding_integer_index(explainer):
    """Check the embedding plot is correct when ind is an integer."""
    fig = plt.figure()
    shap_values = explainer(explainer.data)
    shap.plots.embedding(ind=0, shap_values=shap_values.values, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_embedding_sum_index(explainer):
    """Check the embedding plot is correct when ind is 'sum()'."""
    fig = plt.figure()
    shap_values = explainer(explainer.data)
    shap.plots.embedding(ind="sum()", shap_values=shap_values.values, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_embedding_custom_method(explainer):
    """Check the embedding plot is correct when a pre-computed (n_samples, 2) array is used."""
    fig = plt.figure()
    shap_values = explainer(explainer.data)
    rng = np.random.RandomState(0)
    custom_embedding = rng.rand(shap_values.values.shape[0], 2)
    shap.plots.embedding(ind=0, shap_values=shap_values.values, method=custom_embedding, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_embedding_feature_names_provided(explainer):
    """Check the embedding plot is correct when feature names are provided."""
    fig = plt.figure()
    shap_values = explainer(explainer.data)
    n_features = shap_values.values.shape[1]
    feature_names = [f"feat_{i}" for i in range(n_features)]
    shap.plots.embedding(ind=0, shap_values=shap_values.values, feature_names=feature_names, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_embedding_alpha(explainer):
    """Check the embedding plot is correct when alpha is set to a custom value."""
    fig = plt.figure()
    shap_values = explainer(explainer.data)
    shap.plots.embedding(ind=0, shap_values=shap_values.values, alpha=0.5, show=False)
    plt.tight_layout()
    return fig


def test_embedding_unsupported_method(explainer):
    """Check that an UnboundLocalError is raised when an unsupported method is passed."""
    # GH 4394 - will be updated to ValueError once fix in PR #4395 is merged
    shap_values = explainer(explainer.data)
    with pytest.raises(UnboundLocalError):
        shap.plots.embedding(
            ind=0,
            shap_values=shap_values.values,
            method="unsupported_method",
            show=False,
        )
