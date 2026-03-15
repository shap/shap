"""This file contains tests for the bar plot."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.cluster.hierarchy

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
    hlines = [line for line in ax.lines if line.get_ydata()[0] == 0 and line.get_xdata()[0] == ax.get_xlim()[0]]
    assert len(hlines) == 0


def test_bar_vertical_show_true_returns_none():
    """vertical=True with show=True should return None (not the Axes object)."""
    rs = np.random.RandomState(2)
    result = shap.plots.bar(
        shap.Explanation(values=rs.randn(20, 5), base_values=np.zeros(20)),
        vertical=True,
        show=True,
    )
    assert result is None
    plt.close("all")


def test_bar_vertical_with_external_ax():
    """Passing an existing Axes with vertical=True skips the fig-sizing block."""
    rs = np.random.RandomState(3)
    fig, ax = plt.subplots()
    returned_ax = shap.plots.bar(
        shap.Explanation(values=rs.randn(20, 5), base_values=np.zeros(20)),
        vertical=True,
        ax=ax,
        show=False,
    )
    assert returned_ax is ax
    plt.close("all")


def test_bar_vertical_max_display_none():
    """max_display=None with vertical=True should display all features."""
    rs = np.random.RandomState(4)
    n_features = 6
    ax = shap.plots.bar(
        shap.Explanation(
            values=rs.randn(20, n_features),
            base_values=np.zeros(20),
            feature_names=[f"f{i}" for i in range(n_features)],
        ),
        vertical=True,
        max_display=None,
        show=False,
    )
    assert len(ax.containers) == 1  # one bar group
    assert len(ax.get_xticks()) == n_features
    plt.close("all")


def test_bar_vertical_with_cohorts_object():
    """vertical=True with a Cohorts object (not a dict) exercises the Cohorts branch."""
    rs = np.random.RandomState(5)
    exp = shap.Explanation(
        values=rs.randn(40, 5),
        base_values=np.zeros(40),
        feature_names=[f"f{i}" for i in range(5)],
    )
    cohorts = shap.Cohorts(**{"group_a": exp[:20], "group_b": exp[20:]})
    ax = shap.plots.bar(cohorts, vertical=True, show=False)
    assert ax is not None
    plt.close("all")


def test_bar_vertical_clustering_false():
    """clustering=False with vertical=True sets partition_tree to None (lines 121-122)."""
    rs = np.random.RandomState(6)
    ax = shap.plots.bar(
        shap.Explanation(values=rs.randn(20, 5), base_values=np.zeros(20)),
        vertical=True,
        clustering=False,
        show=False,
    )
    assert ax is not None
    plt.close("all")


def test_bar_vertical_invalid_clustering_raises():
    """Passing an invalid clustering array with vertical=True raises TypeError (lines 123-127)."""
    rs = np.random.RandomState(6)
    with pytest.raises(TypeError, match="does not seem to be a partition tree"):
        shap.plots.bar(
            shap.Explanation(values=rs.randn(20, 5), base_values=np.zeros(20)),
            vertical=True,
            clustering=np.array([1, 2, 3]),
            show=False,
        )
    plt.close("all")


def test_bar_vertical_empty_explanation_raises():
    """Passing an empty Explanation with vertical=True raises ValueError (line 135)."""
    with pytest.raises(ValueError, match="The passed Explanation is empty"):
        shap.plots.bar(
            shap.Explanation(values=np.array([]), base_values=np.array([])),
            vertical=True,
            show=False,
        )


def test_bar_vertical_string_feature_names():
    """String feature_names triggers ordinal label generation (line 144)."""
    rs = np.random.RandomState(7)
    ax = shap.plots.bar(
        shap.Explanation(
            values=rs.randn(20, 4),
            base_values=np.zeros(20),
            feature_names="token",
        ),
        vertical=True,
        show=False,
    )
    assert ax is not None
    plt.close("all")


def test_bar_vertical_pandas_series_features():
    """pd.Series data triggers the unwrap branch (lines 166-168)."""
    rs = np.random.RandomState(8)
    n = 4
    ax = shap.plots.bar(
        shap.Explanation(
            values=rs.randn(n),
            base_values=0.0,
            data=pd.Series(rs.rand(n), index=[f"f{i}" for i in range(n)]),
        ),
        vertical=True,
        show=False,
    )
    assert ax is not None
    plt.close("all")


def test_bar_vertical_show_data_with_features():
    """show_data=True with feature data covers the value-in-label branch (line 241)."""
    rs = np.random.RandomState(9)
    n = 4
    ax = shap.plots.bar(
        shap.Explanation(
            values=rs.randn(n),
            base_values=0.0,
            data=np.array([1.0, 2.5, 0.0, -1.0]),
            feature_names=[f"f{i}" for i in range(n)],
        ),
        vertical=True,
        show_data=True,
        show=False,
    )
    assert ax is not None
    plt.close("all")


def test_bar_vertical_with_valid_clustering():
    """Valid partition tree with vertical=True exercises the clustering while loop (lines 190-210)."""
    rs = np.random.RandomState(10)
    n_features = 6
    values = rs.randn(30, n_features)
    # Build a valid linkage/partition tree from the feature value matrix
    partition_tree = scipy.cluster.hierarchy.linkage(values.T, method="ward")
    ax = shap.plots.bar(
        shap.Explanation(
            values=values,
            base_values=np.zeros(30),
            feature_names=[f"f{i}" for i in range(n_features)],
        ),
        vertical=True,
        clustering=partition_tree,
        max_display=4,  # fewer than n_features to force the merge loop
        show=False,
    )
    assert ax is not None
    plt.close("all")


def _make_merge_partition_tree():
    """Return a 4-feature partition tree where all cross-cluster distances equal 0.5.

    This guarantees the while-loop merge condition triggers when max_display=3.
    Tree: features 0+1 merge at 0.1, features 2+3 merge at 0.2, both groups merge at 0.5.
    """
    return np.array([
        [0.0, 1.0, 0.1, 2.0],
        [2.0, 3.0, 0.2, 2.0],
        [4.0, 5.0, 0.5, 4.0],
    ])


def test_bar_vertical_clustering_merge_loop():
    """Partition tree + max_display triggers node-merging in the while loop (lines 203-208).

    Features 0 and 1 are nearly zero (tiny SHAP effect), so they land at the end of
    feature_order. Because all cross-cluster cophenet distances are exactly 0.5 (==
    the default clustering_cutoff), the merge condition is True and merge_nodes is called.
    After one merge the loop has only 3 features left and exits.
    """
    rs = np.random.RandomState(11)
    n = 20
    values = np.column_stack([
        rs.randn(n) * 0.05,  # f0 – tiny effect
        rs.randn(n) * 0.05,  # f1 – tiny effect, clusters with f0
        rs.randn(n) * 2.0,   # f2 – large effect
        rs.randn(n) * 1.5,   # f3 – medium effect
    ])
    ax = shap.plots.bar(
        shap.Explanation(
            values=values,
            base_values=np.zeros(n),
            feature_names=["f0", "f1", "f2", "f3"],
        ),
        vertical=True,
        clustering=_make_merge_partition_tree(),
        max_display=3,
        show=False,
    )
    assert ax is not None
    plt.close("all")


def test_bar_vertical_clustering_merged_short_name():
    """Merged feature with short combined name covers feature_names_new branch (line 224)."""
    rs = np.random.RandomState(12)
    n = 20
    values = np.column_stack([
        rs.randn(n) * 0.05,
        rs.randn(n) * 0.05,
        rs.randn(n) * 2.0,
        rs.randn(n) * 1.5,
    ])
    # short names -> "f0 + f1" (7 chars) <= 40 -> line 224
    ax = shap.plots.bar(
        shap.Explanation(
            values=values,
            base_values=np.zeros(n),
            feature_names=["f0", "f1", "f2", "f3"],
        ),
        vertical=True,
        clustering=_make_merge_partition_tree(),
        max_display=3,
        show=False,
    )
    assert ax is not None
    plt.close("all")


def test_bar_vertical_clustering_merged_long_name():
    """Merged feature with long combined name covers the truncated-name branch (lines 226-227)."""
    rs = np.random.RandomState(13)
    n = 20
    values = np.column_stack([
        rs.randn(n) * 0.05,
        rs.randn(n) * 0.05,
        rs.randn(n) * 2.0,
        rs.randn(n) * 1.5,
    ])
    # "very_long_feature_name_alpha + very_long_feature_name_beta" = 58 chars > 40 -> line 226-227
    long_names = [
        "very_long_feature_name_alpha",
        "very_long_feature_name_beta_",
        "short_f2",
        "short_f3",
    ]
    ax = shap.plots.bar(
        shap.Explanation(
            values=values,
            base_values=np.zeros(n),
            feature_names=long_names,
        ),
        vertical=True,
        clustering=_make_merge_partition_tree(),
        max_display=3,
        show=False,
    )
    assert ax is not None
    plt.close("all")
