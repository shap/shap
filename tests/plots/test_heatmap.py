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


def test_heatmap_show_true(explainer, monkeypatch):
    """Test heatmap plot with show=True."""
    shap_values = explainer(explainer.data)
    show_called = []
    monkeypatch.setattr(plt, 'show', lambda: show_called.append(True))
    shap.plots.heatmap(shap_values, show=True)
    assert len(show_called) == 1
    plt.close()


def test_heatmap_invalid_feature_order(explainer):
    """Test that invalid feature_order raises exception."""
    shap_values = explainer(explainer.data)
    with pytest.raises(Exception, match="Unsupported feature_order"):
        shap.plots.heatmap(shap_values, feature_order=42, show=False)
