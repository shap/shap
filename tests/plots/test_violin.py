import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import shap
from shap.plots._violin import _trim_crange, shorten_text
from shap.utils._exceptions import DimensionError


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture()
def rng():
    return np.random.RandomState(42)


@pytest.fixture()
def shap_vals(rng):
    return rng.randn(30, 5)


@pytest.fixture()
def feature_data(rng):
    return rng.randn(30, 5)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


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


def test_violin_multi_output_raises(rng):
    """List of shap_values (multi-output) should raise TypeError."""
    sv = [rng.randn(20, 5), rng.randn(20, 5)]
    with pytest.raises(TypeError, match="multi-output"):
        shap.plots.violin(sv)


def test_violin_1d_shap_values_raises(rng):
    """1D shap_values should trigger assertion error."""
    with pytest.raises(AssertionError):
        shap.plots.violin(rng.randn(20), show=False)


def test_violin_title_deprecation(rng, shap_vals):
    """Passing title should emit DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="title"):
        shap.plots.violin(shap_vals, title="unused title", show=False)


# ---------------------------------------------------------------------------
# Explanation object interface
# ---------------------------------------------------------------------------


def test_violin_explanation_object(rng):
    """Explanation objects should be accepted as input."""
    expln = shap.Explanation(
        values=rng.randn(20, 4),
        data=rng.randn(20, 4),
        feature_names=["a", "b", "c", "d"],
    )
    shap.plots.violin(expln, show=False)
    plt.close("all")


def test_violin_explanation_without_feature_names(rng):
    """Explanation without feature_names uses auto-generated labels."""
    expln = shap.Explanation(
        values=rng.randn(20, 4),
        data=rng.randn(20, 4),
    )
    shap.plots.violin(expln, show=False)
    plt.close("all")


# ---------------------------------------------------------------------------
# plot_type="violin" with/without features
# ---------------------------------------------------------------------------


def test_violin_no_features(shap_vals):
    """Default violin plot with no feature data draws violinplot."""
    shap.plots.violin(shap_vals, show=False)
    plt.close("all")


def test_violin_with_features(shap_vals, feature_data):
    """Violin plot with feature data exercises scatter + fill_between path."""
    shap.plots.violin(shap_vals, features=feature_data, show=False)
    plt.close("all")


def test_violin_with_features_and_nan(rng):
    """NaN values in features should be handled and drawn in grey."""
    sv = rng.randn(30, 3)
    feat = rng.randn(30, 3)
    feat[0, 0] = np.nan
    feat[5, 1] = np.nan
    shap.plots.violin(sv, features=feat, show=False)
    plt.close("all")


def test_violin_with_dataframe_features(rng):
    """Features as pd.DataFrame should be accepted and extract column names."""
    sv = rng.randn(30, 4)
    feat_df = pd.DataFrame(rng.randn(30, 4), columns=["w", "x", "y", "z"])
    shap.plots.violin(sv, features=feat_df, show=False)
    plt.close("all")


def test_violin_features_as_list_of_names(shap_vals):
    """Passing a list as features should be used as feature_names shorthand."""
    names = ["f0", "f1", "f2", "f3", "f4"]
    shap.plots.violin(shap_vals, features=names, show=False)
    plt.close("all")


def test_violin_features_1d_as_names(shap_vals, rng):
    """Passing a 1-D numpy array as features should be used as feature_names."""
    names = np.array(["a", "b", "c", "d", "e"])
    shap.plots.violin(shap_vals, features=names, show=False)
    plt.close("all")


# ---------------------------------------------------------------------------
# plot_type="layered_violin"
# ---------------------------------------------------------------------------


def test_layered_violin(shap_vals, feature_data):
    """Basic layered_violin plot should not raise."""
    shap.plots.violin(shap_vals, features=feature_data, plot_type="layered_violin", show=False)
    plt.close("all")


def test_layered_violin_with_few_bins(rng):
    """Layered violin with very few bins should still render."""
    sv = rng.randn(40, 3)
    feat = rng.randn(40, 3)
    shap.plots.violin(sv, features=feat, plot_type="layered_violin", layered_violin_max_num_bins=3, show=False)
    plt.close("all")


def test_layered_violin_with_binary_feature(rng):
    """Feature with only 2 unique values should trigger the unique-value binning path."""
    sv = rng.randn(40, 3)
    feat = np.column_stack([rng.randint(0, 2, size=40), rng.randn(40, 2)])
    shap.plots.violin(sv, features=feat, plot_type="layered_violin", show=False)
    plt.close("all")


def test_layered_violin_singleton_bin_warning(rng):
    """A bin with only one element should emit a UserWarning."""
    sv = rng.randn(5, 2)
    # Two features, using layered_violin_max_num_bins large enough to create singleton bins.
    feat = rng.randn(5, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap.plots.violin(sv, features=feat, plot_type="layered_violin", layered_violin_max_num_bins=20, show=False)
    plt.close("all")


# ---------------------------------------------------------------------------
# Parameter combinations
# ---------------------------------------------------------------------------


def test_violin_max_display(shap_vals):
    """max_display limits the number of displayed features."""
    shap.plots.violin(shap_vals, max_display=3, show=False)
    ax = plt.gca()
    assert len(ax.get_yticklabels()) == 3
    plt.close("all")


def test_violin_sort_false(shap_vals):
    """sort=False should produce a plot without sorting features."""
    shap.plots.violin(shap_vals, sort=False, show=False)
    plt.close("all")


def test_violin_color_bar_false(shap_vals, feature_data):
    """color_bar=False should suppress the colorbar."""
    shap.plots.violin(shap_vals, features=feature_data, color_bar=False, show=False)
    plt.close("all")


def test_violin_plot_size_tuple(shap_vals):
    """plot_size as (width, height) tuple should resize the figure."""
    shap.plots.violin(shap_vals, plot_size=(6, 4), show=False)
    fig = plt.gcf()
    w, h = fig.get_size_inches()
    assert abs(w - 6) < 0.5 and abs(h - 4) < 0.5
    plt.close("all")


def test_violin_plot_size_float(shap_vals):
    """plot_size as a float should scale row height."""
    shap.plots.violin(shap_vals, plot_size=0.6, show=False)
    plt.close("all")


def test_violin_plot_size_none(shap_vals):
    """plot_size=None should leave figure size unchanged."""
    fig = plt.figure(figsize=(7, 7))
    shap.plots.violin(shap_vals, plot_size=None, show=False)
    w, h = fig.get_size_inches()
    assert abs(w - 7) < 0.5 and abs(h - 7) < 0.5
    plt.close("all")


def test_violin_use_log_scale(shap_vals):
    """use_log_scale=True should set symlog x-scale."""
    shap.plots.violin(shap_vals, use_log_scale=True, show=False)
    ax = plt.gca()
    assert ax.get_xscale() == "symlog"
    plt.close("all")


def test_violin_custom_feature_names(shap_vals):
    """Custom feature_names should appear as y-tick labels."""
    names = ["feat_A", "feat_B", "feat_C", "feat_D", "feat_E"]
    shap.plots.violin(shap_vals, feature_names=names, show=False)
    ax = plt.gca()
    labels = [t.get_text() for t in ax.get_yticklabels()]
    for name in names:
        assert name in labels
    plt.close("all")


def test_violin_custom_color(shap_vals):
    """Passing a custom color string should not raise."""
    shap.plots.violin(shap_vals, color="steelblue", show=False)
    plt.close("all")


def test_violin_alpha(shap_vals, feature_data):
    """Custom alpha should not raise."""
    shap.plots.violin(shap_vals, features=feature_data, alpha=0.5, show=False)
    plt.close("all")


def test_violin_axis_color(shap_vals):
    """Custom axis_color should not raise."""
    shap.plots.violin(shap_vals, axis_color="#111111", show=False)
    plt.close("all")


def test_violin_plot_type_none_defaults_to_violin(shap_vals):
    """plot_type=None should default to 'violin' without error."""
    shap.plots.violin(shap_vals, plot_type=None, show=False)
    plt.close("all")


def test_violin_auto_feature_names(shap_vals):
    """No feature names provided should generate auto Feature N labels."""
    shap.plots.violin(shap_vals, show=False)
    ax = plt.gca()
    tick_texts = [t.get_text() for t in ax.get_yticklabels()]
    assert any("Feature" in t for t in tick_texts)
    plt.close("all")


# ---------------------------------------------------------------------------
# Image-comparison tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _trim_crange helper
# ---------------------------------------------------------------------------


def test_trim_crange_normal():
    """_trim_crange returns vmin < vmax for normal data."""
    rng = np.random.RandomState(7)
    values = rng.randn(100)
    nan_mask = np.zeros(100, dtype=bool)
    vmin, vmax, cvals = _trim_crange(values, nan_mask)
    assert vmin <= vmax
    assert cvals.shape[0] == 100


def test_trim_crange_constant():
    """_trim_crange handles constant arrays (all values equal)."""
    values = np.ones(50)
    nan_mask = np.zeros(50, dtype=bool)
    vmin, vmax, cvals = _trim_crange(values, nan_mask)
    assert vmin <= vmax
    assert cvals.shape[0] == 50


def test_trim_crange_with_nans():
    """_trim_crange excludes NaN-masked values from cvals."""
    values = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    nan_mask = np.array([False, False, True, False, False])
    vmin, vmax, cvals = _trim_crange(values, nan_mask)
    assert cvals.shape[0] == 4  # only non-nan values


def test_trim_crange_all_same_percentile():
    """_trim_crange falls back to min/max when 1st/99th percentiles also equal."""
    values = np.array([1.0, 1.0, 1.0, 1.0, 2.0])
    nan_mask = np.zeros(5, dtype=bool)
    vmin, vmax, cvals = _trim_crange(values, nan_mask)
    assert vmin <= vmax


# ---------------------------------------------------------------------------
# shorten_text helper
# ---------------------------------------------------------------------------


def test_shorten_text_short():
    assert shorten_text("hello", 10) == "hello"


def test_shorten_text_long():
    result = shorten_text("hello world", 8)
    assert len(result) == 8
    assert result.endswith("...")


def test_shorten_text_exact_limit():
    assert shorten_text("hello", 5) == "hello"
