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
    emsg = "The shap_values argument must be an Explanation object, Cohorts object, or dictionary of Explanation objects!"
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


# --- vertical=True tests ---


@pytest.mark.mpl_image_compare
def test_bar_vertical():
    """Check that the vertical bar plot is unchanged."""
    rs = np.random.RandomState(42)
    feature_names = [f"feature_{i}" for i in range(12)]
    fig = plt.figure()
    shap.plots.bar(
        shap.Explanation(
            values=rs.randn(50, 12),
            base_values=np.zeros(50),
            feature_names=feature_names,
        ),
        vertical=True,
        show=False,
    )
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_bar_vertical_with_cohorts_dict():
    """Ensure vertical bar plot supports a dictionary of Explanations as input."""
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
        vertical=True,
        show=False,
    )
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_bar_vertical_local_feature_importance():
    """Vertical bar plot with a single row of SHAP values (local explanation)."""
    rs = np.random.RandomState(0)
    feature_names = [f"feature_{i}" for i in range(8)]
    fig = plt.figure()
    shap.plots.bar(
        shap.Explanation(
            values=rs.randn(8),
            base_values=0.5,
            feature_names=feature_names,
        ),
        vertical=True,
        show=False,
    )
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_bar_vertical_negative_values():
    """Vertical bar plot with mixed positive and negative values exercises the axhline(0) branch."""
    rs = np.random.RandomState(7)
    values = rs.randn(30, 5)  # mix of positive and negative
    fig = plt.figure()
    shap.plots.bar(
        shap.Explanation(values=values, base_values=np.zeros(30)),
        vertical=True,
        show=False,
    )
    plt.tight_layout()
    return fig


def test_bar_vertical_returns_ax():
    """Ensure that vertical=True returns the Axes object when show=False."""
    rs = np.random.RandomState(0)
    ax = shap.plots.bar(
        shap.Explanation(values=rs.randn(20, 5), base_values=np.zeros(20)),
        vertical=True,
        show=False,
    )
    assert ax is not None
    assert hasattr(ax, "get_figure")


def test_bar_vertical_only_positive_values():
    """Vertical bar plot with all-positive values should not draw an axhline at 0."""
    rs = np.random.RandomState(1)
    values = np.abs(rs.randn(20, 4))
    ax = shap.plots.bar(
        shap.Explanation(values=values, base_values=np.zeros(20)),
        vertical=True,
        show=False,
    )
    # no horizontal line at y=0 should have been added (only the bar containers)
    assert len(ax.containers) > 0
    hlines = [
        line
        for line in ax.lines
        if line.get_ydata()[0] == 0 and line.get_xdata()[0] == ax.get_xlim()[0]
    ]
    assert len(hlines) == 0
