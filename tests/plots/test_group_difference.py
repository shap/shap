import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap
from shap.plots._group_difference import group_difference


@pytest.mark.mpl_image_compare
def test_group_difference(explainer):
    """Check that the group_difference plot is unchanged."""
    np.random.seed(0)

    shap_values = explainer(explainer.data).values
    group_mask = np.random.randint(2, size=shap_values.shape[0])
    feature_names = explainer.data_feature_names
    fig, ax = plt.subplots()
    shap.plots.group_difference(shap_values, group_mask, feature_names, show=False, ax=ax)
    plt.tight_layout()
    return fig


@pytest.fixture()
def basic_data():
    """Return a deterministic 2-D SHAP matrix, boolean group mask, and feature names.

    Produces 20 samples × 4 features split evenly (10 per group).
    A fixed RandomState is used so results are stable regardless of the
    global seed set by the autouse fixture in conftest.py.
    """
    rs = np.random.RandomState(42)
    shap_values = rs.randn(20, 4)
    group_mask = np.array([True] * 10 + [False] * 10)
    feature_names = ["feat_0", "feat_1", "feat_2", "feat_3"]
    return shap_values, group_mask, feature_names


@pytest.fixture()
def basic_1d_data():
    """Return a deterministic 1-D SHAP array and a boolean group mask.

    Represents the single-model-output case where shap_values is a vector
    rather than a matrix.
    """
    rs = np.random.RandomState(7)
    shap_values = rs.randn(30)
    group_mask = np.array([True] * 15 + [False] * 15)
    return shap_values, group_mask


def test_basic_call_does_not_raise(basic_data):
    """Check that group_difference completes without error on well-formed 2-D input."""
    shap_values, group_mask, feature_names = basic_data
    fig, ax = plt.subplots()
    group_difference(shap_values, group_mask, feature_names=feature_names, show=False, ax=ax)


def test_returns_none(basic_data):
    """Check that group_difference returns None (no explicit return value)."""
    shap_values, group_mask, feature_names = basic_data
    fig, ax = plt.subplots()
    result = group_difference(shap_values, group_mask, feature_names=feature_names, show=False, ax=ax)
    assert result is None


def test_axes_object_is_populated(basic_data):
    """Check that group_difference draws at least one bar on the provided axes."""
    shap_values, group_mask, feature_names = basic_data
    fig, ax = plt.subplots()
    group_difference(shap_values, group_mask, feature_names=feature_names, show=False, ax=ax)
    assert len(ax.patches) > 0


def test_1d_shap_values_does_not_raise(basic_1d_data):
    """Check that a 1-D shap_values array (single model output) is handled without error."""
    shap_values, group_mask = basic_1d_data
    fig, ax = plt.subplots()
    group_difference(shap_values, group_mask, show=False, ax=ax)


def test_1d_shap_values_auto_feature_name(basic_1d_data):
    """Check that 1-D input with feature_names=None produces a single empty-string tick label.

    The source sets feature_names = [""] for the 1-D branch when no names are supplied.
    """
    shap_values, group_mask = basic_1d_data
    fig, ax = plt.subplots()
    group_difference(shap_values, group_mask, feature_names=None, show=False, ax=ax)
    tick_labels = [t.get_text() for t in ax.get_yticklabels()]
    assert "" in tick_labels


def test_1d_shap_values_with_explicit_feature_name(basic_1d_data):
    """Check that a caller-supplied feature name is used for 1-D input."""
    shap_values, group_mask = basic_1d_data
    fig, ax = plt.subplots()
    group_difference(shap_values, group_mask, feature_names=["my_output"], show=False, ax=ax)
    tick_labels = [t.get_text() for t in ax.get_yticklabels()]
    assert "my_output" in tick_labels


def test_auto_feature_names_2d(basic_data):
    """Check that feature_names=None on 2-D input generates 'Feature N' labels."""
    shap_values, group_mask, _ = basic_data
    fig, ax = plt.subplots()
    group_difference(shap_values, group_mask, feature_names=None, show=False, ax=ax)
    tick_labels = [t.get_text() for t in ax.get_yticklabels()]
    assert any(label.startswith("Feature ") for label in tick_labels)


def test_auto_feature_names_count(basic_data):
    """Check that the number of auto-generated feature names matches the feature count."""
    shap_values, group_mask, _ = basic_data
    n_features = shap_values.shape[1]
    fig, ax = plt.subplots()
    group_difference(shap_values, group_mask, feature_names=None, show=False, ax=ax)
    tick_labels = [t.get_text() for t in ax.get_yticklabels()]
    auto_labels = [label for label in tick_labels if label.startswith("Feature ")]
    assert len(auto_labels) == n_features


def test_default_xlabel(basic_data):
    """Check that xlabel=None falls back to the hard-coded default string."""
    shap_values, group_mask, feature_names = basic_data
    fig, ax = plt.subplots()
    group_difference(shap_values, group_mask, feature_names=feature_names, show=False, ax=ax)
    assert ax.get_xlabel() == "Group SHAP value difference"


def test_custom_xlabel(basic_data):
    """Check that a caller-supplied xlabel appears on the x-axis."""
    shap_values, group_mask, feature_names = basic_data
    fig, ax = plt.subplots()
    group_difference(
        shap_values,
        group_mask,
        feature_names=feature_names,
        xlabel="Fairness gap",
        show=False,
        ax=ax,
    )
    assert ax.get_xlabel() == "Fairness gap"


def test_max_display_limits_tick_count(basic_data):
    """Check that max_display=2 results in exactly 2 y-tick labels."""
    shap_values, group_mask, feature_names = basic_data
    fig, ax = plt.subplots()
    group_difference(shap_values, group_mask, feature_names=feature_names, max_display=2, show=False, ax=ax)
    assert len(ax.get_yticklabels()) == 2


def test_max_display_larger_than_features_shows_all(basic_data):
    """Check that max_display larger than n_features shows all features without error."""
    shap_values, group_mask, feature_names = basic_data
    fig, ax = plt.subplots()
    group_difference(shap_values, group_mask, feature_names=feature_names, max_display=999, show=False, ax=ax)
    assert len(ax.get_yticklabels()) == shap_values.shape[1]


def test_xmin_xmax_applied(basic_data):
    """Check that xmin and xmax are forwarded to ax.set_xlim."""
    shap_values, group_mask, feature_names = basic_data
    fig, ax = plt.subplots()
    group_difference(
        shap_values,
        group_mask,
        feature_names=feature_names,
        xmin=-5.0,
        xmax=5.0,
        show=False,
        ax=ax,
    )
    lo, hi = ax.get_xlim()
    assert lo == pytest.approx(-5.0)
    assert hi == pytest.approx(5.0)


def test_no_ax_creates_new_figure(basic_data):
    """Check that ax=None (default) causes the function to create a new figure."""
    shap_values, group_mask, feature_names = basic_data
    before = set(plt.get_fignums())
    group_difference(shap_values, group_mask, feature_names=feature_names, show=False)
    assert set(plt.get_fignums()) - before


def test_diff_positive_when_group1_higher():
    """Check that the bar is positive when group-1 mean exceeds group-2 mean.

    Group 1 (mask=True) has value 5.0; group 2 has value 1.0.
    Expected diff = 5.0 - 1.0 = +4.0.
    """
    sv = np.array([[5.0]] * 10 + [[1.0]] * 10)
    mask = np.array([True] * 10 + [False] * 10)
    fig, ax = plt.subplots()
    group_difference(sv, mask, show=False, ax=ax)
    assert ax.patches[0].get_width() > 0


def test_diff_zero_for_equal_groups():
    """Check that all bars have width ≈ 0 when both groups share identical values."""
    sv = np.ones((20, 3)) * 2.5
    mask = np.array([True] * 10 + [False] * 10)
    fig, ax = plt.subplots()
    group_difference(sv, mask, show=False, ax=ax)
    for patch in ax.patches:
        assert abs(patch.get_width()) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("spine", ["right", "top", "left"])
def test_spines_hidden(basic_data, spine):
    """Check that right, top, and left spines are all invisible after the call."""
    shap_values, group_mask, feature_names = basic_data
    fig, ax = plt.subplots()
    group_difference(shap_values, group_mask, feature_names=feature_names, show=False, ax=ax)
    assert not ax.spines[spine].get_visible()


def test_accessible_via_shap_plots_namespace(basic_data):
    """Check that shap.plots.group_difference resolves to the same function."""
    shap_values, group_mask, feature_names = basic_data
    fig, ax = plt.subplots()
    shap.plots.group_difference(shap_values, group_mask, feature_names=feature_names, show=False, ax=ax)
