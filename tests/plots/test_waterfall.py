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


@pytest.mark.mpl_image_compare(tolerance=3)
def test_waterfall_max_display():
    """Test waterfall plot with max_display to trigger feature grouping."""
    # Create a model with many features
    X = pd.DataFrame(np.random.randn(50, 15), columns=[f"Feature {i}" for i in range(15)])
    y = pd.Series(np.random.randn(50))
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)

    fig = plt.figure()
    # Use max_display=10 to trigger the "other features" grouping
    shap.plots.waterfall(explanation[0], max_display=10, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare(tolerance=3)
def test_waterfall_no_features():
    """Test waterfall plot when features=None (only feature names shown)."""
    X = pd.DataFrame({"A": [1, 2, 3], "B": [2, 1, 3], "C": [3, 3, 1]})
    y = pd.Series([1, 2, 3])
    model = DecisionTreeRegressor()
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)

    # Create new Explanation with data=None to test features=None path
    from shap import Explanation

    explanation_no_features = Explanation(
        values=explanation[0].values,
        base_values=explanation[0].base_values,
        data=None,
        feature_names=explanation[0].feature_names,
    )

    fig = plt.figure()
    shap.plots.waterfall(explanation_no_features, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare(tolerance=3)
def test_waterfall_string_features():
    """Test waterfall plot with non-numeric (string) feature values."""
    X = pd.DataFrame(
        {
            "Category_A": [1, 0, 0, 1, 0],
            "Category_B": [0, 1, 0, 0, 1],
            "Category_C": [0, 0, 1, 0, 0],
            "Type_X": [1, 0, 1, 0, 1],
            "Type_Y": [0, 1, 0, 1, 0],
            "Value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    y = pd.Series([1, 2, 3, 4, 5])

    model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)

    # Create new Explanation with string features
    from shap import Explanation

    explanation_strings = Explanation(
        values=explanation[0].values,
        base_values=explanation[0].base_values,
        data=np.array(["Cat_A", "Cat_B", "Cat_C", "Type_X", "Type_Y", 1.0], dtype=object),
        feature_names=explanation[0].feature_names,
    )

    fig = plt.figure()
    shap.plots.waterfall(explanation_strings, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare(tolerance=3)
def test_waterfall_pandas_series_features():
    """Test waterfall plot with pandas Series as features."""
    X = pd.DataFrame({"A": [1, 2, 3], "B": [2, 1, 3], "C": [3, 3, 1]})
    y = pd.Series([1, 2, 3])
    model = DecisionTreeRegressor()
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)

    # Create new Explanation with pandas Series features
    from shap import Explanation

    explanation_series = Explanation(
        values=explanation[0].values,
        base_values=explanation[0].base_values,
        data=pd.Series(explanation[0].data, index=X.columns),
        feature_names=explanation[0].feature_names,
    )

    fig = plt.figure()
    shap.plots.waterfall(explanation_series, show=False)
    plt.tight_layout()
    return fig


def test_waterfall_return_value():
    """Test that waterfall returns axes when show=False."""
    X = pd.DataFrame({"A": [1, 2, 3], "B": [2, 1, 3]})
    y = pd.Series([1, 2, 3])
    model = DecisionTreeRegressor()
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)

    ax = shap.plots.waterfall(explanation[0], show=False)
    assert ax is not None
    assert hasattr(ax, "plot")  # Check it's a matplotlib axes
    plt.close()


@pytest.mark.mpl_image_compare(tolerance=3)
def test_waterfall_legacy_max_display():
    """Test legacy waterfall plot with max_display parameter."""
    np.random.seed(42)
    expected_value = 0.5
    shap_values = np.random.randn(15) * 0.1
    features = np.random.randn(15)
    feature_names = [f"Feature {i}" for i in range(15)]

    fig = plt.figure()
    shap.plots._waterfall.waterfall_legacy(
        expected_value, shap_values, features, feature_names, max_display=10, show=False
    )
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare(tolerance=3)
def test_waterfall_legacy_no_features():
    """Test legacy waterfall plot without feature values."""
    np.random.seed(42)
    expected_value = 0.5
    shap_values = np.random.randn(8) * 0.2
    feature_names = [f"Feature {i}" for i in range(8)]

    fig = plt.figure()
    shap.plots._waterfall.waterfall_legacy(
        expected_value, shap_values, features=None, feature_names=feature_names, show=False
    )
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare(tolerance=3)
def test_waterfall_legacy_pandas_series():
    """Test legacy waterfall plot with pandas Series features."""
    np.random.seed(42)
    expected_value = 0.5
    shap_values = np.random.randn(5) * 0.2
    features = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=["A", "B", "C", "D", "E"])

    fig = plt.figure()
    shap.plots._waterfall.waterfall_legacy(expected_value, shap_values, features=features, show=False)
    plt.tight_layout()
    return fig


def test_waterfall_legacy_return_value():
    """Test that waterfall_legacy returns figure when show=False."""
    np.random.seed(42)
    expected_value = 0.5
    shap_values = np.random.randn(5) * 0.2
    features = np.random.randn(5)
    feature_names = [f"Feature {i}" for i in range(5)]

    fig = shap.plots._waterfall.waterfall_legacy(expected_value, shap_values, features, feature_names, show=False)
    assert fig is not None
    assert hasattr(fig, "savefig")  # Check it's a matplotlib figure
    plt.close()
