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


@pytest.mark.mpl_image_compare
def test_group_difference_no_ax(explainer):
    """Test group_difference without providing ax (creates own figure)."""
    np.random.seed(0)
    shap_values = explainer(explainer.data).values
    group_mask = np.random.randint(2, size=shap_values.shape[0])
    feature_names = explainer.data_feature_names
    shap.plots.group_difference(shap_values, group_mask, feature_names, show=False)
    plt.tight_layout()
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_group_difference_1d_values(explainer):
    """Test group_difference with 1D model output vector."""
    np.random.seed(0)
    # Use just model outputs (1D) instead of SHAP values matrix
    model_outputs = np.random.randn(100)
    group_mask = np.random.randint(2, size=100).astype(bool)
    shap.plots.group_difference(model_outputs, group_mask, show=False)
    plt.tight_layout()
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_group_difference_no_feature_names(explainer):
    """Test group_difference without feature names."""
    np.random.seed(0)
    shap_values = explainer(explainer.data).values
    group_mask = np.random.randint(2, size=shap_values.shape[0])
    shap.plots.group_difference(shap_values, group_mask, show=False)
    plt.tight_layout()
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_group_difference_no_sort(explainer):
    """Test group_difference with sort=False."""
    np.random.seed(0)
    shap_values = explainer(explainer.data).values
    group_mask = np.random.randint(2, size=shap_values.shape[0])
    feature_names = explainer.data_feature_names
    shap.plots.group_difference(shap_values, group_mask, feature_names, sort=False, show=False)
    plt.tight_layout()
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_group_difference_max_display(explainer):
    """Test group_difference with max_display parameter."""
    np.random.seed(0)
    shap_values = explainer(explainer.data).values
    group_mask = np.random.randint(2, size=shap_values.shape[0])
    feature_names = explainer.data_feature_names
    shap.plots.group_difference(shap_values, group_mask, feature_names, max_display=5, show=False)
    plt.tight_layout()
    return plt.gcf()


def test_group_difference_show_true(explainer, monkeypatch):
    """Test group_difference with show=True."""
    np.random.seed(0)
    shap_values = explainer(explainer.data).values
    group_mask = np.random.randint(2, size=shap_values.shape[0])
    feature_names = explainer.data_feature_names
    show_called = []
    monkeypatch.setattr(plt, "show", lambda: show_called.append(True))
    shap.plots.group_difference(shap_values, group_mask, feature_names, show=True)
    assert len(show_called) == 1
    plt.close()
