import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap


@pytest.mark.mpl_image_compare
def test_group_difference(explainer):
    """Check that the group_difference plot is unchanged."""
    np.random.seed(0)

    shap_values = explainer(explainer.data).values
    group_mask = np.random.randint(2, size=shap_values.shape[0])
    feature_names = explainer.data_feature_names
    fig, ax = plt.subplots()
    shap.plots.group_difference(shap_values, group_mask, feature_names, show=False, ax=ax)
    plt.tight_layout()
    return fig

def test_group_difference_returns_ax(explainer):
    """group_difference() should always return an Axes object."""
    np.random.seed(0)
    shap_values = explainer(explainer.data).values
    group_mask = np.random.randint(2, size=shap_values.shape[0]).astype(bool)
    result = shap.plots.group_difference(shap_values, group_mask, show=False)
    assert isinstance(result, plt.Axes)
    plt.close("all")


def test_group_difference_accepts_ax(explainer):
    """group_difference() should draw into a provided Axes and return it."""
    np.random.seed(0)
    shap_values = explainer(explainer.data).values
    group_mask = np.random.randint(2, size=shap_values.shape[0]).astype(bool)
    fig, ax = plt.subplots()
    result = shap.plots.group_difference(shap_values, group_mask, show=False, ax=ax)
    assert result is ax
    plt.close(fig)
