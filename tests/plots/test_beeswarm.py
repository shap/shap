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


def test_beeswarm_with_directionality():
    """Test that beeswarm plot with show_directionality=True works correctly."""
    np.random.seed(42)

    # Create synthetic data with clear positive and negative correlations
    n_samples = 100
    n_features = 3

    # Feature 0: positive correlation with SHAP values
    feature_0 = np.random.randn(n_samples)
    shap_0 = feature_0 * 2 + np.random.randn(n_samples) * 0.5

    # Feature 1: negative correlation with SHAP values
    feature_1 = np.random.randn(n_samples)
    shap_1 = -feature_1 * 2 + np.random.randn(n_samples) * 0.5

    # Feature 2: no correlation
    feature_2 = np.random.randn(n_samples)
    shap_2 = np.random.randn(n_samples)

    data = np.column_stack([feature_0, feature_1, feature_2])
    values = np.column_stack([shap_0, shap_1, shap_2])

    explanation = shap.Explanation(
        values=values, data=data, feature_names=["Positive Feature", "Negative Feature", "No Correlation"]
    )

    # Test without directionality (should work as before)
    fig1 = plt.figure()
    shap.plots.beeswarm(explanation, show=False, show_directionality=False)
    plt.close(fig1)

    # Test with directionality (should add +/- symbols)
    fig2 = plt.figure()
    ax = shap.plots.beeswarm(explanation, show=False, show_directionality=True)

    # Verify that the plot was created successfully
    assert ax is not None

    # Get the y-tick labels
    yticklabels = [label.get_text() for label in ax.get_yticklabels()]

    # Check that at least one label has a directionality symbol
    has_plus = any("+" in label for label in yticklabels)
    has_minus = any("−" in label or "-" in label for label in yticklabels)

    # At least one of the features should have a directionality indicator
    assert has_plus or has_minus, "Expected at least one directionality indicator in labels"

    plt.close(fig2)


def test_beeswarm_directionality_with_nan():
    """Test that directionality handles NaN values gracefully."""
    np.random.seed(42)

    n_samples = 50
    feature_vals = np.random.randn(n_samples)
    shap_vals = feature_vals * 2 + np.random.randn(n_samples) * 0.5

    # Add some NaN values
    feature_vals[0:5] = np.nan

    data = feature_vals.reshape(-1, 1)
    values = shap_vals.reshape(-1, 1)

    explanation = shap.Explanation(values=values, data=data, feature_names=["Feature with NaN"])

    # Should not raise an error
    fig = plt.figure()
    shap.plots.beeswarm(explanation, show=False, show_directionality=True)
    plt.close(fig)


def test_beeswarm_directionality_without_features():
    """Test that directionality is skipped when features are not provided."""
    np.random.seed(42)

    # Create explanation without feature data
    explanation = shap.Explanation(values=np.random.randn(50, 3), feature_names=["Feature 1", "Feature 2", "Feature 3"])

    # Should work without errors (directionality will be skipped)
    fig = plt.figure()
    shap.plots.beeswarm(explanation, show=False, show_directionality=True)
    plt.close(fig)
