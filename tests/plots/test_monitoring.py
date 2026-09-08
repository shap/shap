import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import shap


@pytest.fixture
def basic_data():
    np.random.seed(42)
    shap_values = np.random.randn(200, 3)
    features = np.random.randn(200, 3)
    return shap_values, features


def test_monitoring_basic_numpy(basic_data):
    shap_values, features = basic_data
    shap.plots.monitoring(ind=0, shap_values=shap_values, features=features, show=False)
    plt.close()


def test_monitoring_dataframe_input(basic_data):
    shap_values, features = basic_data
    df = pd.DataFrame(features, columns=["feat_a", "feat_b", "feat_c"])
    shap.plots.monitoring(ind=0, shap_values=shap_values, features=df, show=False)
    plt.close()


def test_monitoring_custom_feature_names(basic_data):
    shap_values, features = basic_data
    names = ["Temperature", "Humidity", "Pressure"]
    shap.plots.monitoring(ind=1, shap_values=shap_values, features=features, feature_names=names, show=False)
    plt.close()


def test_monitoring_auto_feature_names(basic_data):
    shap_values, features = basic_data
    shap.plots.monitoring(ind=0, shap_values=shap_values, features=features, feature_names=None, show=False)
    plt.close()


def test_monitoring_different_feature_index(basic_data):
    shap_values, features = basic_data
    shap.plots.monitoring(ind=2, shap_values=shap_values, features=features, show=False)
    plt.close()


def test_monitoring_long_feature_name(basic_data):
    shap_values, features = basic_data
    names = ["A" * 50, "feat_b", "feat_c"]
    shap.plots.monitoring(ind=0, shap_values=shap_values, features=features, feature_names=names, show=False)
    plt.close()
