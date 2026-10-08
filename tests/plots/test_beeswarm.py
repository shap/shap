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


# -----------------------------------------------------------------------
# API consistency tests
# -----------------------------------------------------------------------


def _make_explanation(n=50, n_features=5, seed=0):
    """Create a simple multi-column Explanation for unit tests."""
    rs = np.random.RandomState(seed)
    values = rs.randn(n, n_features)
    data = rs.randn(n, n_features)
    feature_names = [f"feat{i}" for i in range(n_features)]
    return shap.Explanation(values=values, data=data, feature_names=feature_names)


def test_show_false_returns_axes():
    """show=False must return a matplotlib Axes object."""
    plt.close("all")
    expln = _make_explanation()
    result = shap.plots.beeswarm(expln, show=False)
    assert isinstance(result, plt.Axes), f"Expected Axes, got {type(result)}"
    plt.close("all")


def test_ax_parameter_draws_on_given_axes():
    """When ax is provided the plot must be drawn on that exact Axes."""
    plt.close("all")
    fig, ax = plt.subplots()
    expln = _make_explanation()
    result = shap.plots.beeswarm(expln, ax=ax, show=False)
    assert result is ax, "beeswarm() must return the axes it was given"
    plt.close("all")


def test_subplot_isolation():
    """Drawing on one subplot must not modify sibling subplots."""
    plt.close("all")
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax2_title_before = ax2.get_title()
    ax2_xlabel_before = ax2.get_xlabel()
    ax2_n_lines_before = len(ax2.lines)
    ax2_n_collections_before = len(ax2.collections)

    expln = _make_explanation()
    shap.plots.beeswarm(expln, ax=ax1, show=False)

    assert ax2.get_title() == ax2_title_before, "sibling ax2 title was modified"
    assert ax2.get_xlabel() == ax2_xlabel_before, "sibling ax2 xlabel was modified"
    assert len(ax2.lines) == ax2_n_lines_before, "sibling ax2 gained unexpected line artists"
    assert len(ax2.collections) == ax2_n_collections_before, "sibling ax2 gained unexpected scatter/collection artists"
    plt.close("all")


def test_figure_size_unchanged_when_ax_provided():
    """When an ax is provided, the figure size must not be altered by beeswarm()."""
    plt.close("all")
    original_size = (4.0, 3.0)
    fig, ax = plt.subplots(figsize=original_size)
    expln = _make_explanation()
    # plot_size must be None when ax is provided (otherwise ValueError is raised)
    shap.plots.beeswarm(expln, ax=ax, plot_size=None, show=False)
    actual_size = tuple(fig.get_size_inches())
    assert actual_size == pytest.approx(original_size, rel=1e-3), (
        f"Figure size changed from {original_size} to {actual_size} when ax was provided"
    )
    plt.close("all")


def test_multiple_calls_on_same_ax():
    """Calling beeswarm() multiple times on the same Axes must not crash,
    and each call must add artists to the axes."""
    plt.close("all")
    fig, ax = plt.subplots()
    expln = _make_explanation()

    shap.plots.beeswarm(expln, ax=ax, show=False)
    n_collections_after_first = len(ax.collections)
    assert n_collections_after_first > 0, "First beeswarm() call produced no scatter collections on the axes"

    shap.plots.beeswarm(expln, ax=ax, show=False)
    n_collections_after_second = len(ax.collections)
    assert n_collections_after_second > n_collections_after_first, (
        "Second beeswarm() call did not add new artists to the axes"
    )
    plt.close("all")


def test_colorbar_compatibility():
    """beeswarm() with color_bar=True must not crash."""
    plt.close("all")
    expln = _make_explanation()
    result = shap.plots.beeswarm(expln, color_bar=True, show=False)
    assert isinstance(result, plt.Axes)
    plt.close("all")


def test_colorbar_with_explicit_ax():
    """Colorbar must attach to the provided ax without error or leaking to other axes."""
    plt.close("all")
    fig, (ax1, ax2) = plt.subplots(1, 2)

    expln = _make_explanation()
    result = shap.plots.beeswarm(expln, color_bar=True, ax=ax1, show=False)

    assert result is ax1, "beeswarm() must return the axes it was given"

    # Ensure sibling axes untouched
    assert ax2.get_xlabel() == "", "colorbar should not have written an xlabel onto sibling ax2"
    assert len(ax2.collections) == 0, "sibling ax2 should have no scatter artists after colorbar"

    # Validate any new axes belong to ax1 region (colorbar inset behavior)
    new_axes = [a for a in fig.axes if a is not ax1 and a is not ax2]

    for cb_ax in new_axes:
        ax1_x0 = ax1.get_position().x0
        cb_x0 = cb_ax.get_position().x0

        assert cb_x0 >= ax1_x0, f"Colorbar axes at x0={cb_x0:.3f} appears to be left of ax1 (x0={ax1_x0:.3f})"

    plt.close("all")


def test_backward_compatibility_no_ax():
    """Calling without ax must still work and return Axes when show=False."""
    plt.close("all")
    expln = _make_explanation()
    result = shap.plots.beeswarm(expln, show=False)
    assert isinstance(result, plt.Axes)
    plt.close("all")


def test_ax_with_plot_size_raises():
    """Passing ax and an explicit numeric plot_size must raise ValueError."""
    plt.close("all")
    fig, ax = plt.subplots()
    expln = _make_explanation()
    with pytest.raises(ValueError, match="does not support passing an axis and adjusting the plot size"):
        shap.plots.beeswarm(expln, ax=ax, plot_size=0.5, show=False)
    plt.close("all")


def test_pyplot_current_axes_not_clobbered():
    """When ax is provided, plt.gca() must still point to whatever it was before the call,
    not be silently redirected to the axes we passed in."""
    plt.close("all")
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # Make ax2 the active axes
    plt.sca(ax2)
    assert plt.gca() is ax2, "pre-condition: ax2 must be current before the call"

    expln = _make_explanation()
    shap.plots.beeswarm(expln, ax=ax1, show=False)

    assert plt.gca() is ax2, "beeswarm() must not change plt.gca() when an explicit ax is supplied"
    plt.close("all")


def test_rng_parameter_produces_deterministic_output():
    """Passing rng must make dot positions reproducible across calls."""
    plt.close("all")
    expln = _make_explanation(seed=7)
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)

    fig1, ax1 = plt.subplots()
    shap.plots.beeswarm(expln, ax=ax1, show=False, rng=rng_a)
    offsets_first = [c.get_offsets().data.copy() for c in ax1.collections]

    fig2, ax2 = plt.subplots()
    shap.plots.beeswarm(expln, ax=ax2, show=False, rng=rng_b)
    offsets_second = [c.get_offsets().data.copy() for c in ax2.collections]

    assert len(offsets_first) == len(offsets_second), "Number of scatter collections differs"
    for o1, o2 in zip(offsets_first, offsets_second):
        np.testing.assert_array_equal(o1, o2, err_msg="Dot positions differ despite identical rng seeds")
    plt.close("all")
