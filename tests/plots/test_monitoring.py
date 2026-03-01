import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap


@pytest.fixture()
def monitoring_data():
    """Create data for monitoring plots - needs more than 100 samples."""
    xgboost = pytest.importorskip("xgboost")
    X, y = shap.datasets.adult()
    X_train = X.iloc[:100]
    y_train = y[:100]
    X_test = X.iloc[100:300]  # Need >100 samples for monitoring plots

    model = xgboost.XGBClassifier(random_state=0, tree_method="exact", base_score=0.5).fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    return shap_values, X_test.values, X_test.columns.tolist()


@pytest.mark.mpl_image_compare(tolerance=3)
def test_monitoring_basic(monitoring_data):
    """Test basic monitoring plot."""
    shap_values, features, feature_names = monitoring_data
    shap.plots.monitoring(0, shap_values, features, feature_names=feature_names, show=False)
    plt.tight_layout()
    return plt.gcf()


@pytest.mark.mpl_image_compare(tolerance=3)
def test_monitoring_dataframe():
    """Test monitoring plot with DataFrame input."""
    pd = pytest.importorskip("pandas")
    np.random.seed(42)
    shap_values = np.random.randn(200, 3)
    features_df = pd.DataFrame(np.random.randn(200, 3), columns=["Feature 1", "Feature 2", "Feature 3"])
    shap.plots.monitoring(1, shap_values, features_df, show=False)
    plt.tight_layout()
    return plt.gcf()


@pytest.mark.mpl_image_compare(tolerance=3)
def test_monitoring_age_feature(monitoring_data):
    """Test monitoring plot for Age feature."""
    shap_values, features, feature_names = monitoring_data
    age_idx = feature_names.index("Age")
    shap.plots.monitoring(age_idx, shap_values, features, feature_names=feature_names, show=False)
    plt.tight_layout()
    return plt.gcf()


def test_monitoring_no_feature_names(monitoring_data):
    """Test monitoring plot without feature names."""
    shap_values, features, _ = monitoring_data
    shap.plots.monitoring(0, shap_values, features, show=False)
    plt.close()


@pytest.mark.mpl_image_compare(tolerance=3)
def test_monitoring_long_feature_name():
    """Test monitoring plot with long feature name that gets truncated."""
    np.random.seed(42)
    shap_values = np.random.randn(200, 3)
    features = np.random.randn(200, 3)
    feature_names = ["Short", "This is a very long feature name that should be truncated in the plot", "Medium"]
    shap.plots.monitoring(1, shap_values, features, feature_names=feature_names, show=False)
    plt.tight_layout()
    return plt.gcf()


def test_monitoring_show_true(monitoring_data, monkeypatch):
    """Test monitoring plot with show=True."""
    shap_values, features, feature_names = monitoring_data
    # Mock plt.show() to avoid actually displaying
    show_called = []
    monkeypatch.setattr(plt, "show", lambda: show_called.append(True))
    shap.plots.monitoring(0, shap_values, features, feature_names=feature_names, show=True)
    assert len(show_called) == 1
    plt.close()
