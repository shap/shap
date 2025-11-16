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


def test_beeswarm_single_instance_raises_error():
    """Test that passing single instance raises ValueError."""
    explanation = shap.Explanation(values=np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="does not support plotting a single instance"):
        shap.plots.beeswarm(explanation, show=False)


def test_beeswarm_3d_raises_error():
    """Test that passing 3D explanation raises ValueError."""
    explanation = shap.Explanation(values=np.random.randn(10, 5, 3))
    with pytest.raises(ValueError, match="more than one dimension"):
        shap.plots.beeswarm(explanation, show=False)


def test_beeswarm_ax_and_plot_size_raises_error(explainer):
    """Test that passing both ax and plot_size raises ValueError."""
    shap_values = explainer(explainer.data)
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="does not support passing an axis and adjusting the plot size"):
        shap.plots.beeswarm(shap_values, ax=ax, plot_size=(10, 8), show=False)
    plt.close()


@pytest.mark.mpl_image_compare
def test_beeswarm_with_max_display(explainer):
    """Test beeswarm plot with max_display parameter."""
    shap_values = explainer(explainer.data)
    shap.plots.beeswarm(shap_values, max_display=5, show=False)
    plt.tight_layout()
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_beeswarm_with_custom_ax(explainer):
    """Test beeswarm plot with custom axes."""
    fig, ax = plt.subplots()
    shap_values = explainer(explainer.data)
    returned_ax = shap.plots.beeswarm(shap_values, ax=ax, plot_size=None, show=False)
    assert returned_ax == ax
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_beeswarm_with_alpha(explainer):
    """Test beeswarm plot with custom alpha."""
    shap_values = explainer(explainer.data)
    shap.plots.beeswarm(shap_values, alpha=0.5, show=False)
    plt.tight_layout()
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_beeswarm_with_custom_s(explainer):
    """Test beeswarm plot with custom marker size."""
    shap_values = explainer(explainer.data)
    shap.plots.beeswarm(shap_values, s=30, show=False)
    plt.tight_layout()
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_beeswarm_with_log_scale(explainer):
    """Test beeswarm plot with log scale."""
    shap_values = explainer(explainer.data)
    shap.plots.beeswarm(shap_values, log_scale=True, show=False)
    plt.tight_layout()
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_beeswarm_no_color_bar(explainer):
    """Test beeswarm plot without color bar."""
    shap_values = explainer(explainer.data)
    shap.plots.beeswarm(shap_values, color_bar=False, show=False)
    plt.tight_layout()
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_beeswarm_with_custom_color_bar_label(explainer):
    """Test beeswarm plot with custom color bar label."""
    shap_values = explainer(explainer.data)
    shap.plots.beeswarm(shap_values, color_bar_label="Custom Label", show=False)
    plt.tight_layout()
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_beeswarm_with_axis_color(explainer):
    """Test beeswarm plot with custom axis color."""
    shap_values = explainer(explainer.data)
    shap.plots.beeswarm(shap_values, axis_color="#FF0000", show=False)
    plt.tight_layout()
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_beeswarm_with_plot_size_float(explainer):
    """Test beeswarm plot with plot_size as float."""
    shap_values = explainer(explainer.data)
    shap.plots.beeswarm(shap_values, plot_size=0.5, show=False)
    plt.tight_layout()
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_beeswarm_with_plot_size_tuple(explainer):
    """Test beeswarm plot with plot_size as tuple."""
    shap_values = explainer(explainer.data)
    shap.plots.beeswarm(shap_values, plot_size=(10, 8), show=False)
    plt.tight_layout()
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_beeswarm_with_plot_size_none(explainer):
    """Test beeswarm plot with plot_size=None."""
    shap_values = explainer(explainer.data)
    shap.plots.beeswarm(shap_values, plot_size=None, show=False)
    plt.tight_layout()
    return plt.gcf()


def test_beeswarm_show_true(explainer, monkeypatch):
    """Test beeswarm plot with show=True."""
    shap_values = explainer(explainer.data)
    show_called = []
    monkeypatch.setattr(plt, "show", lambda: show_called.append(True))
    shap.plots.beeswarm(shap_values, show=True)
    assert len(show_called) == 1
    plt.close()


@pytest.mark.mpl_image_compare
def test_beeswarm_with_sparse_features(explainer):
    """Test beeswarm plot with sparse feature matrix."""
    shap_values = explainer(explainer.data)
    # Convert data to sparse matrix
    import scipy.sparse

    sparse_data = scipy.sparse.csr_matrix(shap_values.data)
    shap_values_sparse = shap.Explanation(
        values=shap_values.values, data=sparse_data, feature_names=shap_values.feature_names
    )
    shap.plots.beeswarm(shap_values_sparse, show=False)
    plt.tight_layout()
    return plt.gcf()
