import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

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


@pytest.fixture()
def binary_clf_explainer():
    # A binary classifier explainer that returns shap values of shape (n_samples, n_features, 2)    # get a dataset on income prediction
    X, y = shap.datasets.adult()
    X = X.iloc[:100]
    y = y[:100]

    model = RandomForestClassifier(n_estimators=3, random_state=0)
    model.fit(X, y)
    return shap.TreeExplainer(model, X)


@pytest.mark.filterwarnings("ignore:The passed Explanation object has two classes")
@pytest.mark.mpl_image_compare
def test_beeswarm_binary_classifier(binary_clf_explainer):
    # GH 3951
    # Ensure the plot works for explanations of shape (n_samples, n_features, 2)
    fig = plt.figure()
    explanation = binary_clf_explainer(binary_clf_explainer.data)

    sv_shape = explanation.values.shape
    assert len(sv_shape) == 3
    assert sv_shape[2] == 2
    shap.plots.beeswarm(explanation, show=False)
    plt.tight_layout()
    return fig


def test_beeswarm_basic_explanation_works():
    # GH 3901
    explanation = shap.Explanation([[1.0, 2.0, 3.0]])
    shap.plots.beeswarm(explanation, show=False)


@pytest.mark.parametrize(
    "color",
    [
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
    ],
)
def test_beeswarm_works_with_colors(color):
    # GH 3901
    explanation = shap.Explanation([[1.0, 2.0, 3.0]])
    shap.plots.beeswarm(explanation, show=False, color_bar=True, color=color)
