import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap
from shap.utils._exceptions import DimensionError


def test_violin_with_invalid_plot_type():
    with pytest.raises(ValueError, match="plot_type: Expected one of "):
        shap.plots.violin(np.random.randn(20, 5), plot_type="nonsense")


def test_violin_wrong_features_shape():
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
        shap.plots.violin(expln, show=False)
    # legacy API
    with pytest.raises(DimensionError, match=emsg):
        shap.plots.violin(
            shap_values=rs.randn(20, 5),
            features=rs.randn(20, 4),
            show=False,
        )

    emsg = "The shape of the shap_values matrix does not match the shape of the provided data matrix."
    with pytest.raises(DimensionError, match=emsg):
        expln = shap.Explanation(
            values=rs.randn(20, 5),
            data=rs.randn(20, 1),
        )
        shap.plots.violin(expln, show=False)
    # legacy API
    with pytest.raises(DimensionError, match=emsg):
        shap.plots.violin(
            shap_values=rs.randn(20, 5),
            features=rs.randn(20, 1),
            show=False,
        )


def test_violin_returns_axes():
    """The violin plot returns the Axes it drew onto."""
    rs = np.random.RandomState(0)
    plt.figure()
    ax = shap.plots.violin(rs.standard_normal(size=(20, 5)), show=False)
    assert ax is plt.gca()
    plt.close("all")


@pytest.mark.parametrize(
    ("plot_type", "with_features"),
    [
        ("violin", True),
        ("violin", False),
        # layered_violin indexes into `features`, so it requires feature values
        ("layered_violin", True),
    ],
)
def test_violin_draws_onto_supplied_axes(plot_type, with_features):
    """The violin plot draws onto, and returns, a caller-supplied Axes."""
    rs = np.random.RandomState(0)
    features = rs.standard_normal(size=(20, 5)) if with_features else None
    fig, ax = plt.subplots()
    assert shap.plots.violin(rs.standard_normal(size=(20, 5)), features, plot_type=plot_type, show=False, ax=ax) is ax
    # something was actually drawn onto the supplied axes
    assert ax.collections or ax.patches
    plt.close("all")


def test_violin_does_not_resize_supplied_figure():
    """A caller-supplied Axes keeps its own figure size, unlike the implicit-axes path."""
    rs = np.random.RandomState(0)
    fig, ax = plt.subplots(figsize=(3, 3))
    size_before = tuple(fig.get_size_inches())
    shap.plots.violin(rs.standard_normal(size=(20, 5)), show=False, ax=ax)
    assert tuple(fig.get_size_inches()) == size_before
    plt.close("all")


def test_violin_use_log_scale():
    """use_log_scale=True applies a symmetric log scale to the x-axis."""
    rs = np.random.RandomState(0)
    plt.figure()
    ax = shap.plots.violin(rs.standard_normal(size=(20, 5)), use_log_scale=True, show=False)
    assert ax.get_xscale() == "symlog"
    plt.close("all")


@pytest.mark.parametrize(
    ("plot_size", "expected"),
    [
        # "auto" scales the height by row_height (0.4) for the 5 features plotted
        ("auto", (8, 5 * 0.4 + 1.5)),
        # a (width, height) pair is used as-is
        ((7, 2), (7, 2)),
        # a single float is the height of each row
        (0.5, (8, 5 * 0.5 + 1.5)),
    ],
)
def test_violin_plot_size(plot_size, expected):
    """plot_size resizes the figure when the plot picks the Axes itself."""
    rs = np.random.RandomState(0)
    fig = plt.figure()
    shap.plots.violin(rs.standard_normal(size=(20, 5)), plot_size=plot_size, show=False)
    assert tuple(fig.get_size_inches()) == pytest.approx(expected)
    plt.close("all")


def test_violin_plot_size_none_leaves_figure_alone():
    """plot_size=None leaves the size of the figure unchanged."""
    rs = np.random.RandomState(0)
    fig = plt.figure(figsize=(3, 3))
    size_before = tuple(fig.get_size_inches())
    shap.plots.violin(rs.standard_normal(size=(20, 5)), plot_size=None, show=False)
    assert tuple(fig.get_size_inches()) == size_before
    plt.close("all")


@pytest.mark.mpl_image_compare
def test_violin(explainer):
    """Make sure the violin plot is unchanged."""
    fig = plt.figure()
    shap_values = explainer.shap_values(explainer.data)
    shap.plots.violin(shap_values, show=False)
    plt.tight_layout()
    return fig


# FIXME: remove once we migrate violin completely to the Explanation object
# ------ "legacy" violin plots -------
# Currently using the same files as the `test_summary.py` violin tests for comparison
@pytest.mark.mpl_image_compare(
    filename="test_summary_violin_with_data.png",
    tolerance=5,
)
def test_summary_violin_with_data2():
    """Check a violin chart with shap_values as a np.array."""
    rs = np.random.RandomState(0)
    fig = plt.figure()
    shap.plots.violin(
        rs.standard_normal(size=(20, 5)),
        rs.standard_normal(size=(20, 5)),
        plot_type="violin",
        show=False,
    )
    fig.set_layout_engine("tight")
    return fig


# Currently using the same files as the `test_summary.py` violin tests for comparison
@pytest.mark.mpl_image_compare(
    filename="test_summary_layered_violin_with_data.png",
    tolerance=5,
)
def test_summary_layered_violin_with_data2():
    """Check a layered violin chart with shap_values as a np.array."""
    rs = np.random.RandomState(0)
    fig = plt.figure()
    shap_values = rs.randn(200, 5)
    feats = rs.randn(200, 5)
    shap.plots.violin(
        shap_values,
        feats,
        plot_type="layered_violin",
        show=False,
    )
    fig.set_layout_engine("tight")
    return fig
