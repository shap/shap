import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
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
