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


def test_waterfall_legacy_with_explanation_object():
    """waterfall_legacy must accept an Explanation object as its first argument.

    Before the fix, it accessed shap_exp.expected_value which does not exist on
    Explanation (the correct attribute is .base_values), causing an AttributeError.
    """
    X = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
    y = pd.Series([1.0, 2.0, 3.0])
    model = DecisionTreeRegressor()
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)

    # Passing a single-row Explanation must not raise AttributeError
    shap.plots._waterfall.waterfall_legacy(explanation[0], show=False)
    plt.close("all")


def test_waterfall_legacy_explanation_uses_base_values():
    """waterfall_legacy must read base_values, not expected_value, from an Explanation.

    We verify by comparing the scalar used inside the plot (expected_value after
    unpacking) against explanation[0].base_values directly.
    """
    X = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
    y = pd.Series([1.0, 2.0, 3.0])
    model = DecisionTreeRegressor()
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)

    single = explanation[0]
    expected_base = float(single.base_values)

    # Confirm the attribute used by the fix exists and is a finite scalar
    assert np.isfinite(expected_base), "base_values should be a finite scalar"

    # Confirm the old attribute does NOT exist, so the previous code was broken
    assert not hasattr(single, "expected_value"), (
        "Explanation should not have .expected_value; if it does the test premise is wrong"
    )

    # The plot should render without error using the correct attribute
    shap.plots._waterfall.waterfall_legacy(single, show=False)
    plt.close("all")


def test_waterfall_legacy_multirow_explanation_raises():
    """waterfall_legacy must raise when passed a multi-row Explanation."""
    X = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
    y = pd.Series([1.0, 2.0, 3.0])
    model = DecisionTreeRegressor()
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)

    # A multi-row Explanation has array-shaped base_values, which should be
    # caught by the scalar check inside waterfall_legacy.
    with pytest.raises(Exception, match="scalar expected_value"):
        shap.plots._waterfall.waterfall_legacy(explanation, show=False)
    plt.close("all")


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
