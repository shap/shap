import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap


@pytest.fixture()
def violin_data(explainer):
    """Return shap values, features, and feature names for violin tests."""
    X, _ = shap.datasets.adult()
    X = X.iloc[:100]
    explanation = explainer(X)
    shap_values = explanation.values
    features = X.values
    feature_names = list(X.columns)
    return shap_values, features, feature_names


@pytest.mark.mpl_image_compare
def test_violin(violin_data):
    """Check that the violin plot is unchanged (no-features variant)."""
    np.random.seed(0)
    shap_values, _, feature_names = violin_data
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.violin(shap_values, feature_names=feature_names, show=False, ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_violin_with_features(violin_data):
    """Check that the violin plot with feature values is unchanged."""
    np.random.seed(0)
    shap_values, features, feature_names = violin_data
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.violin(shap_values, features=features, feature_names=feature_names, show=False, ax=ax)
    return fig


@pytest.mark.mpl_image_compare
def test_violin_layered(violin_data):
    """Check that the layered violin plot is unchanged."""
    np.random.seed(0)
    shap_values, features, feature_names = violin_data
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.violin(
        shap_values,
        features=features,
        feature_names=feature_names,
        plot_type="layered_violin",
        show=False,
        ax=ax,
    )
    return fig


def test_violin_returns_ax(violin_data):
    """violin() should always return an Axes object."""
    np.random.seed(0)
    shap_values, features, feature_names = violin_data
    result = shap.plots.violin(shap_values, features=features, feature_names=feature_names, show=False)
    assert isinstance(result, plt.Axes)
    plt.close("all")


def test_violin_accepts_ax(violin_data):
    """violin() should draw into a provided Axes and return it."""
    np.random.seed(0)
    shap_values, features, feature_names = violin_data
    fig, ax = plt.subplots()
    result = shap.plots.violin(shap_values, features=features, feature_names=feature_names, show=False, ax=ax)
    assert result is ax
    plt.close(fig)
