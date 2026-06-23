import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor

import shap
from shap.plots import _style


def test_waterfall_input_is_explanation():
    """Checks an error is raised if a non-Explanation object is passed as input."""
    with pytest.raises(
        TypeError,
        match="waterfall plot requires an `Explanation` object",
    ):
        _ = shap.plots.waterfall(np.random.randn(20, 5), show=False)


def test_waterfall_wrong_explanation_shape(explainer):
    explanation = explainer(explainer.data)

    emsg = "waterfall plot can currently only plot a single explanation"
    with pytest.raises(ValueError, match=emsg):
        shap.plots.waterfall(explanation, show=False)


@pytest.mark.mpl_image_compare(tolerance=3)
def test_waterfall(explainer):
    """Test the new waterfall plot."""
    fig = plt.figure()
    explanation = explainer(explainer.data)
    shap.plots.waterfall(explanation[0], show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare(tolerance=3)
def test_waterfall_legacy(explainer):
    """Test the old waterfall plot."""
    shap_values = explainer.shap_values(explainer.data)
    fig = plt.figure()
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare(tolerance=3)
def test_waterfall_bounds(explainer):
    """Test the waterfall plot with upper and lower error bounds plotted."""
    fig = plt.figure()
    explanation = explainer(explainer.data)
    explanation._s.lower_bounds = explanation.values - 0.1
    explanation._s.upper_bounds = explanation.values + 0.1
    shap.plots.waterfall(explanation[0])
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare(tolerance=5)
def test_waterfall_custom_style(explainer):
    """Test the waterfall plot in the context of custom styles"""

    # Note: the tolerance is set to 5 because matplotlib 3.10 changed the way negative values are displayed
    # There is now an increased space before the negative sign, which leads to a RMS diff of ~4.4
    # See: GH #3946

    # TODO: reset tolerance to 3 when python 3.9 is dropped, and all tests use matplotlib 3.10+
    with _style.style_context(
        primary_color_positive="#9ACD32",
        primary_color_negative="#FFA500",
        text_color="black",
        hlines_color="red",
        vlines_color="red",
        tick_labels_color="red",
    ):
        fig = plt.figure()
        explanation = explainer(explainer.data)
        shap.plots.waterfall(explanation[0])
        plt.tight_layout()
    return fig


def test_waterfall_plot_for_decision_tree_explanation():
    # Regression tests for GH #3129
    X = pd.DataFrame({"A": [1, 2, 3], "B": [2, 1, 3]})
    y = pd.Series([1, 2, 3])
    model = DecisionTreeRegressor()
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)
    shap.plots.waterfall(explanation[0], show=False)


def test_waterfall_plot_for_data_with_number_columns():
    # GH 4150
    model = KNeighborsClassifier()

    def f(x):
        return model.predict_proba(x)[:, 1]

    X = pd.DataFrame(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
        ]
    )
    y = pd.Series([0, 1, 2, 3, 4, 3, 2, 1])
    model.fit(X, y)
    med = X.median().values.reshape((1, X.shape[1]))
    explainer = shap.Explainer(f, med)
    shap_values = explainer(X)
    shap.plots.waterfall(shap_values[0], show=False)


def _make_explanation(n_features=5, seed=0):
    """Helper to build a simple Explanation object for waterfall tests."""
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(n_features)
    data = rng.standard_normal(n_features)
    feature_names = [f"feature_{i}" for i in range(n_features)]
    return shap.Explanation(
        values=values,
        base_values=0.5,
        data=data,
        feature_names=feature_names,
    )


def test_waterfall_max_display_triggers_grouping():
    """When max_display < n_features, remaining features are grouped into one bar."""
    exp = _make_explanation(n_features=10)
    shap.plots.waterfall(exp, max_display=4, show=False)
    plt.close()


def test_waterfall_no_feature_names():
    """Fallback feature names (Feature 0, Feature 1, ...) are used when feature_names is None."""
    exp = shap.Explanation(
        values=np.array([0.3, -0.2, 0.1]),
        base_values=0.0,
        data=np.array([1.0, 2.0, 3.0]),
        feature_names=None,
    )
    shap.plots.waterfall(exp, show=False)
    plt.close()


def test_waterfall_features_none():
    """waterfall works when data is None (only feature names are shown)."""
    exp = shap.Explanation(
        values=np.array([0.4, -0.1, 0.2]),
        base_values=0.0,
        data=None,
        feature_names=["a", "b", "c"],
    )
    shap.plots.waterfall(exp, show=False)
    plt.close()


def test_waterfall_pandas_series_features():
    """Features passed as a pandas Series should be unwrapped correctly."""
    features = pd.Series([1.5, 2.5, 3.5], index=["x", "y", "z"])
    exp = shap.Explanation(
        values=np.array([0.3, -0.2, 0.1]),
        base_values=0.0,
        data=features,
    )
    shap.plots.waterfall(exp, show=False)
    plt.close()


def test_waterfall_legacy_multi_output_expected_value_raises():
    """waterfall_legacy raises when expected_value is an array (multi-output)."""
    shap_values = np.array([0.1, -0.2, 0.3])
    with pytest.raises(Exception, match="scalar expected_value"):
        shap.plots._waterfall.waterfall_legacy(np.array([0.5, 0.6]), shap_values, show=False)


def test_waterfall_legacy_2d_shap_values_raises():
    """waterfall_legacy raises when shap_values is 2D."""
    shap_values = np.array([[0.1, -0.2, 0.3], [0.4, 0.5, -0.1]])
    with pytest.raises(Exception, match="single explanation"):
        shap.plots._waterfall.waterfall_legacy(0.5, shap_values, show=False)


def test_waterfall_legacy_no_feature_names():
    """waterfall_legacy uses default feature names when feature_names is None."""
    shap_values = np.array([0.3, -0.1, 0.2])
    shap.plots._waterfall.waterfall_legacy(
        0.5, shap_values, features=np.array([1.0, 2.0, 3.0]), feature_names=None, show=False
    )
    plt.close()


def test_waterfall_legacy_max_display_triggers_grouping():
    """waterfall_legacy groups remaining features when max_display < n_features."""
    shap_values = np.random.default_rng(0).standard_normal(10)
    features = np.random.default_rng(1).standard_normal(10)
    feature_names = [f"f{i}" for i in range(10)]
    shap.plots._waterfall.waterfall_legacy(
        0.5, shap_values, features=features, feature_names=feature_names, max_display=4, show=False
    )
    plt.close()


def test_waterfall_legacy_pandas_series_features():
    """waterfall_legacy unwraps pandas Series features correctly."""
    shap_values = np.array([0.3, -0.2, 0.1])
    features = pd.Series([1.5, 2.5, 3.5], index=["x", "y", "z"])
    shap.plots._waterfall.waterfall_legacy(0.5, shap_values, features=features, show=False)
    plt.close()
