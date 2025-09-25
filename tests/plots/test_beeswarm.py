import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap
from shap.plots.colors import (
    blue_rgb,
    gray_rgb,
    light_blue_rgb,
    light_red_rgb,
    red_blue,
    red_blue_circle,
    red_blue_no_bounds,
    red_blue_transparent,
    red_rgb,
    red_transparent_blue,
    red_white_blue,
    transparent_blue,
    transparent_red,
)
from shap.utils._exceptions import DimensionError


@pytest.fixture(
    params=[
        blue_rgb,
        gray_rgb,
        light_blue_rgb,
        light_red_rgb,
        red_blue,
        red_blue_circle,
        red_blue_no_bounds,
        red_blue_transparent,
        red_rgb,
        red_transparent_blue,
        red_white_blue,
        transparent_blue,
        transparent_red,
    ]
)
def color(request):
    return request.param


def test_beeswarm_input_is_explanation():
    """Checks an error is raised if a non-Explanation object is passed as input."""
    with pytest.raises(
        TypeError,
        match="beeswarm plot requires an `Explanation` object",
    ):
        _ = shap.plots.beeswarm(np.random.randn(20, 5), show=False)  # type: ignore


def test_beeswarm_wrong_features_shape():
    """Checks that DimensionError is raised if the features data matrix
    has an incompatible shape with the shap_values matrix.
    """
    rs = np.random.RandomState(42)

    emsg = (
        "The shape of the shap_values matrix does not match the shape of "
        "the provided data matrix. Perhaps the extra column"
    )
    with pytest.raises(DimensionError, match=emsg):
        expln = shap.Explanation(
            values=rs.randn(20, 5),
            data=rs.randn(20, 4),
        )
        shap.plots.beeswarm(expln, show=False)

    emsg = "The shape of the shap_values matrix does not match the shape of the provided data matrix."
    with pytest.raises(DimensionError, match=emsg):
        expln = shap.Explanation(
            values=rs.randn(20, 5),
            data=rs.randn(20, 1),
        )
        shap.plots.beeswarm(expln, show=False)


@pytest.mark.mpl_image_compare
def test_beeswarm(explainer):
    """Check a beeswarm chart renders correctly with shap_values as an Explanation
    object (default settings).
    """
    fig = plt.figure()
    shap_values = explainer(explainer.data)
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_beeswarm_no_group_remaining(explainer):
    """Beeswarm with group_remaining_features=False."""
    fig = plt.figure()
    shap_values = explainer(explainer.data)
    shap.plots.beeswarm(shap_values, show=False, group_remaining_features=False)
    plt.tight_layout()
    return fig


def test_beeswarm_basic_explanation_works():
    # GH 3901
    explanation = shap.Explanation([[1.0, 2.0, 3.0]])
    shap.plots.beeswarm(explanation, show=False)


def test_beeswarm_works_with_colors(color):
    # GH 3901
    explanation = shap.Explanation([[1.0, 2.0, 3.0]])
    shap.plots.beeswarm(explanation, show=False, color_bar=True, color=color)


def test_beeswarm_colors_values_with_data(color):
    np.random.seed(42)

    explanation = shap.Explanation(
        values=np.random.randn(100, 5),
        data=np.array([["cat"] * 5] * 100),
    )
    shap.plots.beeswarm(explanation, show=False, color_bar=True, color=color)
