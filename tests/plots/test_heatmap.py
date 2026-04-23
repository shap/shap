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


def test_heatmap_single_row_explanation():
    """Heatmap should handle a single-row Explanation without clustering failures."""
    ex = shap.Explanation(
        values=np.array([[1.0, -2.0, 3.0]]),
        base_values=np.array([0.0]),
        data=np.array([[10.0, 20.0, 30.0]]),
        feature_names=["a", "b", "c"],
    )

    ax = shap.plots.heatmap(ex, show=False)
    plt.tight_layout()

    assert ax is not None
