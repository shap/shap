import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

import shap
from shap.plots import WaterfallColorConfig, colors


def test_waterfall_input_is_explanation():
    """Checks an error is raised if a non-Explanation object is passed as input.
    """
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
    """ Test the new waterfall plot.
    """
    fig = plt.figure()
    shap_values = explainer(explainer.data)
    shap.plots.waterfall(shap_values[0])
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare(tolerance=3)
def test_waterfall_legacy(explainer):
    """ Test the old waterfall plot.
    """
    shap_values = explainer.shap_values(explainer.data)
    fig = plt.figure()
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0])
    plt.tight_layout()
    return fig
# todo: use parametrize to use all possible config options
@pytest.mark.parametrize("color_config", [{"positive_arrow": colors.red_rgb,
                                           "negative_arrow": colors.blue_rgb,
                                           "default_positive_color": colors.light_red_rgb,
                                           "default_negative_color": colors.light_blue_rgb
                                           },
                                          {
                                           'positive_arrow': np.array([1., 0., 0.31796406]),
                                           'negative_arrow': np.array([0., 0.54337757, 0.98337906]),
                                           'default_positive_color': np.array([1., 0.49803922, 0.65490196]),
                                           'default_negative_color': np.array([0.49803922, 0.76862745, 0.98823529])
                                           },
                                          WaterfallColorConfig(
                                           positive_arrow=np.array([1., 0., 0.31796406]),
                                           negative_arrow=np.array([0., 0.54337757, 0.98337906]),
                                           default_positive_color=np.array([1., 0.49803922, 0.65490196]),
                                           default_negative_color=np.array([0.49803922, 0.76862745, 0.98823529])
                                          ),
                                          [[1., 0., 0.31796406], [0., 0.54337757, 0.98337906], [1., 0.49803922, 0.65490196], [0.49803922, 0.76862745, 0.98823529]],
                                          ['#FF0051', '#008BFB', '#FF7FB1', '#7FC4FC'],
                                          "kgry",
                                          ["black", "green", "red", "yellow"],
                                          ])
@pytest.mark.mpl_image_compare(tolerance=3)
def test_waterfall_color_config_default(explainer, color_config):
    """Test waterfall config options."""
    fig = plt.figure()
    shap_values = explainer(explainer.data)
    shap.plots.waterfall(shap_values[0], plot_cmap=color_config)
    plt.tight_layout()
    return fig

def test_waterfall_plot_for_decision_tree_explanation():
    # Regression tests for GH issue #3129
    X = pd.DataFrame({"A": [1, 2, 3], "B": [2, 1, 3]})
    y = pd.Series([1, 2, 3])
    model = DecisionTreeRegressor()
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X)
    shap.plots.waterfall(explanation[0], show=False)
