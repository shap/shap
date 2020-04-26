import matplotlib
import numpy as np
matplotlib.use('Agg')
import shap


def test_random_summary_display_feature_by_index():
    shap.summary_plot(
        shap_values=np.random.randn(20, 5),
        display_features=[0,1],
        show=False
    )


def test_random_summary_with_data_display_feature_by_index():
    shap.summary_plot(
        shap_values=np.random.randn(20, 5),
        features=np.random.randn(20, 5),
        display_features=[0,1],
        show=False
    )


def test_random_summary_with_data_display_feature_by_name():
    shap.summary_plot(
        shap_values=np.random.randn(20, 5),
        features=np.random.randn(20, 5),
        feature_names=["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"],
        display_features=["feature_1", "feature_2"],
        show=False
    )

def test_random_summary_with_data_display_feature_by_name_and_index():
    shap.summary_plot(
        shap_values=np.random.randn(20, 5),
        features=np.random.randn(20, 5),
        feature_names=["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"],
        display_features=[0, "feature_2"],
        show=False
    )

def test_random_multi_class_summary_display_feature_by_index():
    shap.summary_plot(
        shap_values=[np.random.randn(20, 5) for i in range(3)],
        features=np.random.randn(20, 5),
        feature_names=["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"],
        display_features=[0, 1],
        show=False
    )
