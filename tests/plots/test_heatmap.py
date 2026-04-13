import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap
from shap import Explanation


@pytest.mark.mpl_image_compare
def test_heatmap(explainer):
    """Make sure the heatmap plot is unchanged."""
    fig = plt.figure()
    shap_values = explainer(explainer.data)
    shap.plots.heatmap(shap_values, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_heatmap_feature_order(explainer):
    """Make sure the heatmap plot is unchanged when we apply a feature ordering."""
    fig = plt.figure()
    shap_values = explainer(explainer.data)
    shap.plots.heatmap(
        shap_values, max_display=5, feature_order=np.array(range(shap_values.shape[1]))[::-1], show=False
    )
    plt.tight_layout()
    return fig


def test_heatmap_opchain_feature_order(explainer):
    """Heatmap should not raise IndexError when feature_order is an OpChain (closes #4460)."""
    shap_values = explainer(explainer.data)
    ax = shap.plots.heatmap(shap_values, feature_order=Explanation.abs.mean(0).argsort, show=False)
    plt.close("all")
    assert ax is not None


def test_heatmap_all_zero_shap_values():
    """Heatmap should render without error when all SHAP values are zero (closes #4448)."""
    n_samples, n_features = 20, 5
    values = np.zeros((n_samples, n_features))
    data = np.random.default_rng(0).random((n_samples, n_features))
    feature_names = [f"f{i}" for i in range(n_features)]
    shap_values = Explanation(
        values=values,
        base_values=np.zeros(n_samples),
        data=data,
        feature_names=feature_names,
    )
    ax = shap.plots.heatmap(shap_values, show=False)
    plt.close("all")
    assert ax is not None
