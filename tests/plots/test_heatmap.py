import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap


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


@pytest.mark.mpl_image_compare
def test_heatmap_feature_order_opchain(explainer):
    """Make sure the heatmap plot is unchanged when we apply an opchain feature ordering."""
    fig = plt.figure()
    shap_values = explainer(explainer.data)
    shap.plots.heatmap(shap_values, max_display=5, feature_order=shap.Explanation.abs.mean(0).argsort, show=False)
    plt.tight_layout()
    return fig


def test_heatmap_feature_order_invalid(explainer):
    """Handle unsupported feature order condition"""
    shap_values = explainer(explainer.data)

    with pytest.raises(Exception, match="Unsupported feature_order"):
        shap.plots.heatmap(shap_values, feature_order=123, show=False)


def test_heatmap_show(explainer, monkeypatch):
    """Triggers plt.show() in the heatmap"""
    monkeypatch.setattr(plt, "show", lambda: None)

    shap_values = explainer(explainer.data)
    shap.plots.heatmap(shap_values, show=True)
