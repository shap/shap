import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap


@pytest.fixture()
def monitoring_data(explainer):
    """Return shap values array, features, and feature names for monitoring tests."""
    explanation = explainer(explainer.data)
    shap_values = explanation.values
    features = explainer.data
    feature_names = explainer.data_feature_names
    return shap_values, features, feature_names


@pytest.mark.mpl_image_compare
def test_monitoring(monitoring_data):
    """Check that the monitoring plot is unchanged."""
    np.random.seed(0)
    shap_values, features, feature_names = monitoring_data
    fig, ax = plt.subplots(figsize=(10, 3))
    shap.plots.monitoring(0, shap_values, features, feature_names=feature_names, show=False, ax=ax)
    return fig


def test_monitoring_returns_ax(monitoring_data):
    """monitoring() should always return an Axes object."""
    np.random.seed(0)
    shap_values, features, feature_names = monitoring_data
    result = shap.plots.monitoring(0, shap_values, features, feature_names=feature_names, show=False)
    assert isinstance(result, plt.Axes)
    plt.close("all")


def test_monitoring_accepts_ax(monitoring_data):
    """monitoring() should draw into a provided Axes and return it."""
    np.random.seed(0)
    shap_values, features, feature_names = monitoring_data
    fig, ax = plt.subplots()
    result = shap.plots.monitoring(0, shap_values, features, feature_names=feature_names, show=False, ax=ax)
    assert result is ax
    plt.close(fig)
