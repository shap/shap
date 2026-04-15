import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap
from shap.plots import group_difference


@pytest.fixture()
def basic_data():
    rs = np.random.RandomState(42)
    shap_values = rs.randn(20, 4)
    group_mask = np.array([True] * 10 + [False] * 10)
    feature_names = ["feat_0", "feat_1", "feat_2", "feat_3"]
    return shap_values, group_mask, feature_names


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
def test_group_difference_1d(basic_data):
    """Check that a 1-D shap_values array (single model output) renders correctly."""
    shap_values, group_mask, _ = basic_data
    shap_values_1d = shap_values[:, 0]  # reduce to 1-D
    fig, ax = plt.subplots()
    group_difference(shap_values_1d, group_mask, show=False, ax=ax)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_group_difference_max_display(basic_data):
    """Check that max_display limits the number of features shown."""
    shap_values, group_mask, feature_names = basic_data
    fig, ax = plt.subplots()
    group_difference(shap_values, group_mask, feature_names=feature_names, max_display=2, show=False, ax=ax)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_group_difference_custom_xlabel(basic_data):
    """Check that a custom xlabel is rendered on the plot."""
    shap_values, group_mask, feature_names = basic_data
    fig, ax = plt.subplots()
    group_difference(
        shap_values,
        group_mask,
        feature_names=feature_names,
        xlabel="Fairness gap",
        show=False,
        ax=ax,
    )
    plt.tight_layout()
    return fig
