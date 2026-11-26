"""Tests for shap.plots.embedding"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap


@pytest.fixture()
def shap_values_2d():
    """Simple synthetic SHAP values: 30 samples, 5 features."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((30, 5))


def test_embedding_pca(shap_values_2d):
    """embedding() with method='pca' (default) should complete without error."""
    shap.plots.embedding(0, shap_values_2d, show=False)
    plt.close("all")


def test_embedding_precomputed_array(shap_values_2d):
    """embedding() accepts a pre-computed (n_samples, 2) array as method."""
    rng = np.random.default_rng(1)
    precomputed = rng.standard_normal((shap_values_2d.shape[0], 2))
    shap.plots.embedding(0, shap_values_2d, method=precomputed, show=False)
    plt.close("all")


def test_embedding_unsupported_string_raises(shap_values_2d):
    """Passing an unrecognised string for method must raise ValueError, not NameError."""
    with pytest.raises(ValueError, match="Unsupported embedding method"):
        shap.plots.embedding(0, shap_values_2d, method="tsne", show=False)
    plt.close("all")


def test_embedding_unsupported_object_raises(shap_values_2d):
    """Passing an arbitrary non-array object for method must raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported embedding method"):
        shap.plots.embedding(0, shap_values_2d, method=42, show=False)
    plt.close("all")


def test_embedding_wrong_shape_array_raises(shap_values_2d):
    """A numpy array that is not (n_samples, 2) must raise ValueError."""
    bad_array = np.zeros((shap_values_2d.shape[0], 3))  # 3 columns, not 2
    with pytest.raises(ValueError, match="Unsupported embedding method"):
        shap.plots.embedding(0, shap_values_2d, method=bad_array, show=False)
    plt.close("all")


def test_embedding_sum_ind(shap_values_2d):
    """ind='sum()' should colour by sum of all SHAP values without error."""
    shap.plots.embedding("sum()", shap_values_2d, show=False)
    plt.close("all")


def test_embedding_feature_names(shap_values_2d):
    """Custom feature_names are accepted and do not cause an error."""
    names = [f"feat_{i}" for i in range(shap_values_2d.shape[1])]
    shap.plots.embedding("feat_0", shap_values_2d, feature_names=names, show=False)
    plt.close("all")
