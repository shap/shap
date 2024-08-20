import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

import shap


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
    shap_values = explainer(explainer.data)
    shap.plots.waterfall(shap_values[0], show=False)
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


def test_waterfall_plot_for_decision_tree_explanation():
    # Regression tests for GH #3129
    X = pd.DataFrame({"A": [1, 2, 3], "B": [2, 1, 3]})
    y = pd.Series([1, 2, 3])
    model = DecisionTreeRegressor()
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)
    shap.plots.waterfall(explanation[0], show=False)
