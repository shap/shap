import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import shap


def _make_data(n=200, p=5, seed=0):
    rng = np.random.default_rng(seed)
    shap_values = rng.standard_normal((n, p))
    features = rng.standard_normal((n, p))
    return shap_values, features


def test_monitoring_show_false_returns_ax():
    """show=False with no explicit ax returns an Axes object."""
    shap_values, features = _make_data()
    result = shap.plots.monitoring(0, shap_values, features, show=False)
    assert result is not None
    assert hasattr(result, "scatter")
    plt.close("all")


def test_monitoring_ax_parameter():
    """Passing ax= returns that exact axes and leaves sibling axes untouched."""
    shap_values, features = _make_data()
    fig, axes = plt.subplots(1, 2)
    result = shap.plots.monitoring(0, shap_values, features, show=False, ax=axes[0])
    assert result is axes[0]
    assert len(axes[1].collections) == 0
    plt.close("all")


def test_monitoring_dataframe_features():
    """DataFrame features are handled correctly (column names used as feature_names)."""
    shap_values, features_arr = _make_data()
    features_df = pd.DataFrame(features_arr, columns=[f"f{i}" for i in range(features_arr.shape[1])])
    result = shap.plots.monitoring(0, shap_values, features_df, show=False)
    assert result is not None
    plt.close("all")
