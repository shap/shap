"""This file contains tests for the bar plot."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap
from shap.utils._exceptions import DimensionError


@pytest.mark.parametrize(
    "unsupported_inputs",
    [
        [1, 2, 3],
        (1, 2, 3),
        np.array([1, 2, 3]),
        {"a": 1, "b": 2},
    ],
)
def test_input_shap_values_type(unsupported_inputs):
    """Check that a TypeError is raised when shap_values is not a valid input type."""
    emsg = (
        "The shap_values argument must be an Explanation object, Cohorts object, or dictionary of Explanation objects!"
    )
    with pytest.raises(TypeError, match=emsg):
        shap.plots.bar(unsupported_inputs, show=False)


def test_input_shap_values_type_2():
    """Check that a DimensionError is raised if the cohort Explanation objects have different shape."""
    rs = np.random.RandomState(42)
    emsg = "When passing several Explanation objects, they must all have the same number of feature columns!"
    with pytest.raises(DimensionError, match=emsg):
        shap.plots.bar(
            {
                "t1": shap.Explanation(
                    values=rs.randn(40, 10),
                    base_values=np.ones(40) * 0.5,
                ),
                "t2": shap.Explanation(
                    values=rs.randn(20, 5),
                    base_values=np.ones(20) * 0.5,
                ),
            },
            show=False,
        )


@pytest.mark.mpl_image_compare
def test_bar(explainer):
    """Check that the bar plot is unchanged."""
    shap_values = explainer(explainer.data)
    fig = plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_bar_with_cohorts_dict():
    """Ensure that bar plots supports dictionary of Explanations as input."""
    rs = np.random.RandomState(42)
    fig = plt.figure()
    shap.plots.bar(
        {
            "t1": shap.Explanation(
                values=rs.randn(40, 5),
                base_values=np.ones(40) * 0.5,
            ),
            "t2": shap.Explanation(
                values=rs.randn(20, 5),
                base_values=np.ones(20) * 0.5,
            ),
        },
        show=False,
    )
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_bar_local_feature_importance(explainer):
    """Bar plot with single row of SHAP values"""
    shap_values = explainer(explainer.data)
    fig = plt.figure()
    shap.plots.bar(shap_values[0], show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_bar_with_clustering(explainer):
    """Bar plot with clustering"""
    shap_values = explainer(explainer.data)
    clustering = shap.utils.hclust(explainer.data, metric="cosine")
    fig = plt.figure()
    shap.plots.bar(shap_values, clustering=clustering, show=False)
    plt.tight_layout()
    return fig


def test_bar_raises_error_for_invalid_clustering(explainer):
    shap_values = explainer(explainer.data)
    clustering = np.array([1, 2, 3])
    with pytest.raises(TypeError, match="does not seem to be a partition tree"):
        shap.plots.bar(shap_values, clustering=clustering, show=False)


def test_bar_raises_error_for_empty_explanation(explainer):
    shap_values = explainer(explainer.data)
    with pytest.raises(ValueError, match="The passed Explanation is empty"):
        shap.plots.bar(shap_values[0:0], show=False)


def _make_explanation(n=50, n_features=5, seed=0):
    """Create a simple multi-row Explanation for unit tests."""
    rs = np.random.RandomState(seed)
    values = rs.randn(n, n_features)
    data = rs.randn(n, n_features)
    feature_names = [f"feat{i}" for i in range(n_features)]
    return shap.Explanation(
        values=values,
        data=data,
        feature_names=feature_names,
        base_values=np.zeros(n),
    )


def test_show_false_returns_axes():
    """show=False must return a matplotlib Axes object (not None, not some other type)."""
    plt.close("all")
    expln = _make_explanation()
    result = shap.plots.bar(expln, show=False)
    assert isinstance(result, plt.Axes), f"Expected plt.Axes, got {type(result)}"
    plt.close("all")


def test_show_true_returns_none():
    """show=True must return None (pyplot handles display; caller gets nothing back)."""
    plt.close("all")
    expln = _make_explanation()
    result = shap.plots.bar(expln, show=True)
    assert result is None, f"Expected None when show=True, got {type(result)}"
    plt.close("all")


def test_ax_parameter_draws_on_given_axes():
    """When ax is provided the plot must be drawn on that exact Axes object."""
    plt.close("all")
    fig, ax = plt.subplots()
    expln = _make_explanation()
    result = shap.plots.bar(expln, ax=ax, show=False)
    assert result is ax, f"bar() must return the axes it was given, got {type(result)}"
    plt.close("all")


def test_subplot_isolation():
    """Drawing on one subplot must not modify sibling subplots."""
    plt.close("all")
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax2_title_before = ax2.get_title()
    ax2_xlabel_before = ax2.get_xlabel()

    expln = _make_explanation()
    shap.plots.bar(expln, ax=ax1, show=False)

    assert ax2.get_title() == ax2_title_before, "sibling ax2 title was modified by bar()"
    assert ax2.get_xlabel() == ax2_xlabel_before, "sibling ax2 xlabel was modified by bar()"
    plt.close("all")


def test_figure_size_unchanged_when_ax_provided():
    """When an ax is provided, bar() must not alter the figure size."""
    plt.close("all")
    original_size = (4.0, 3.0)
    fig, ax = plt.subplots(figsize=original_size)
    expln = _make_explanation()
    shap.plots.bar(expln, ax=ax, show=False)
    actual_size = tuple(fig.get_size_inches())
    assert actual_size == pytest.approx(original_size, rel=1e-3), (
        f"Figure size changed from {original_size} to {actual_size} when ax was provided"
    )
    plt.close("all")


def test_figure_size_set_when_no_ax_provided():
    """When no ax is provided, bar() must resize the figure to fit the features."""
    plt.close("all")
    expln = _make_explanation(n_features=5)
    result = shap.plots.bar(expln, show=False)
    fig = result.get_figure()
    w, h = fig.get_size_inches()
    # Default logic: 8 wide, height depends on num_features — just verify it's not the mpl default (6.4 x 4.8)
    assert w == pytest.approx(8.0, rel=1e-3), f"Expected width 8.0, got {w}"
    assert h != pytest.approx(4.8, rel=0.1), "Figure height was not resized from the mpl default"
    plt.close("all")


def test_multiple_calls_on_same_ax():
    """Calling bar() multiple times on the same Axes must not crash."""
    plt.close("all")
    fig, ax = plt.subplots()
    expln = _make_explanation()
    shap.plots.bar(expln, ax=ax, show=False)
    shap.plots.bar(expln, ax=ax, show=False)
    plt.close("all")


def test_backward_compatibility_no_ax():
    """Calling without ax must still work and return Axes when show=False."""
    plt.close("all")
    expln = _make_explanation()
    result = shap.plots.bar(expln, show=False)
    assert isinstance(result, plt.Axes), f"Expected plt.Axes, got {type(result)}"
    plt.close("all")


def test_pyplot_current_axes_not_leaked():
    """bar() with an explicit ax must not change matplotlib's current-axes state."""
    plt.close("all")
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.sca(ax2)  # set ax2 as the "current" axes

    expln = _make_explanation()
    shap.plots.bar(expln, ax=ax1, show=False)

    # matplotlib's current axes must still be ax2, not ax1
    assert plt.gca() is ax2, "bar() must not change plt.gca() when ax is provided"
    plt.close("all")
