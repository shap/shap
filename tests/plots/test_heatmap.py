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


def test_heatmap_all_zero_shap_values():
    """Regression test: heatmap should not divide by zero when all SHAP values are zero."""
    import warnings

    from shap import Explanation

    n_samples, n_features = 20, 5
    exp = Explanation(
        values=np.zeros((n_samples, n_features)),
        base_values=np.zeros(n_samples),
        data=np.zeros((n_samples, n_features)),
        feature_names=[f"feature_{i}" for i in range(n_features)],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # treat any RuntimeWarning as a test failure
        ax = shap.plots.heatmap(exp, show=False)

    assert ax is not None
    plt.close()
