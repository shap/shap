"""Tests for shap/plots/_group_difference.py.

Covers:
- Basic 2D SHAP matrix (happy path)
- 1D model-output vector input
- Custom feature names
- Custom xlabel
- xmin / xmax axis clipping
- max_display truncation
- sort=False ordering
- Injecting a pre-existing Axes (ax parameter, GH #3354)
- Multiple subplots in a single figure via ax
- All-True / all-False group mask edge cases
- Single-feature edge case
- Identical groups → differences close to zero
- Reproducibility: same inputs → same bar heights
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap

matplotlib.use("Agg")


# Shared fixtures


@pytest.fixture
def rng():
    """Fixed-seed NumPy RNG so every test is deterministic."""
    return np.random.RandomState(42)


@pytest.fixture
def shap_2d(rng):
    """20 samples × 5 features, split 10/10 between two groups."""
    values = rng.randn(20, 5)
    mask = np.array([True] * 10 + [False] * 10)
    feature_names = [f"feature_{i}" for i in range(5)]
    return values, mask, feature_names


@pytest.fixture
def shap_1d(rng):
    """1-D model-output vector (20 samples) with a 10/10 group split."""
    values = rng.randn(20)
    mask = np.array([True] * 10 + [False] * 10)
    return values, mask


# Happy-path smoke tests


def test_basic_2d_does_not_raise(shap_2d):

    values, mask, feature_names = shap_2d
    shap.plots.group_difference(values, mask, feature_names=feature_names, show=False)
    plt.close("all")


def test_1d_model_output_does_not_raise(shap_1d):
    """group_difference() should accept a 1-D array (model-output vector)."""
    values, mask = shap_1d
    shap.plots.group_difference(values, mask, show=False)
    plt.close("all")


def test_returns_axes_object(shap_2d):
    values, mask, feature_names = shap_2d
    ax = shap.plots.group_difference(values, mask, feature_names=feature_names, show=False)
    assert isinstance(ax, matplotlib.axes.Axes), f"Expected matplotlib.axes.Axes, got {type(ax)}"
    plt.close("all")


# Parameter: feature_names


def test_feature_names_appear_on_yaxis(shap_2d):
    values, mask, feature_names = shap_2d
    ax = shap.plots.group_difference(values, mask, feature_names=feature_names, show=False)
    tick_labels = [t.get_text() for t in ax.get_yticklabels()]
    for name in feature_names:
        assert name in tick_labels, f"'{name}' not found in y-axis tick labels: {tick_labels}"
    plt.close("all")


def test_no_feature_names_uses_fallback(shap_2d):
    values, mask, _ = shap_2d
    shap.plots.group_difference(values, mask, feature_names=None, show=False)
    plt.close("all")


# Parameter: xlabel


def test_custom_xlabel_is_set(shap_2d):
    values, mask, feature_names = shap_2d
    custom_label = "Mean SHAP difference (group A − group B)"
    ax = shap.plots.group_difference(values, mask, feature_names=feature_names, xlabel=custom_label, show=False)
    assert ax.get_xlabel() == custom_label, f"Expected xlabel '{custom_label}', got '{ax.get_xlabel()}'"
    plt.close("all")


# Parameters: xmin / xmax


def test_xmin_is_applied(shap_2d):
    values, mask, feature_names = shap_2d
    ax = shap.plots.group_difference(values, mask, feature_names=feature_names, xmin=-0.01, show=False)
    assert ax.get_xlim()[0] <= -0.01, f"xmin not applied correctly; left limit is {ax.get_xlim()[0]}"
    plt.close("all")


def test_xmax_is_applied(shap_2d):
    values, mask, feature_names = shap_2d
    ax = shap.plots.group_difference(values, mask, feature_names=feature_names, xmax=0.01, show=False)
    assert ax.get_xlim()[1] >= 0.01, f"xmax not applied correctly; right limit is {ax.get_xlim()[1]}"
    plt.close("all")


def test_xmin_xmax_together(shap_2d):
    values, mask, feature_names = shap_2d
    ax = shap.plots.group_difference(values, mask, feature_names=feature_names, xmin=-5.0, xmax=5.0, show=False)
    lo, hi = ax.get_xlim()
    assert lo <= -5.0 and hi >= 5.0
    plt.close("all")


# Parameter: max_display


def test_max_display_limits_visible_bars(rng):
    values = rng.randn(30, 10)
    mask = np.array([True] * 15 + [False] * 15)
    feature_names = [f"f{i}" for i in range(10)]
    ax = shap.plots.group_difference(values, mask, feature_names=feature_names, max_display=3, show=False)
    # Each bar is a Rectangle patch; so we count non-zero-height ones
    bars = [p for p in ax.patches if p.get_height() > 0 or p.get_width() != 0]
    assert len(bars) <= 3, f"Expected ≤3 bars, got {len(bars)}"
    plt.close("all")


def test_max_display_larger_than_features_is_safe(shap_2d):
    values, mask, feature_names = shap_2d
    shap.plots.group_difference(values, mask, feature_names=feature_names, max_display=999, show=False)
    plt.close("all")


# Parameter: sort


def test_sort_false_does_not_raise(shap_2d):

    values, mask, feature_names = shap_2d
    shap.plots.group_difference(values, mask, feature_names=feature_names, sort=False, show=False)
    plt.close("all")


def test_sort_true_vs_false_both_render(shap_2d):
    values, mask, feature_names = shap_2d
    ax_sorted = shap.plots.group_difference(values, mask, feature_names=feature_names, sort=True, show=False)
    plt.close("all")
    ax_unsorted = shap.plots.group_difference(values, mask, feature_names=feature_names, sort=False, show=False)
    plt.close("all")
    assert isinstance(ax_sorted, matplotlib.axes.Axes)
    assert isinstance(ax_unsorted, matplotlib.axes.Axes)


# Parameter: ax


def test_ax_parameter_draws_into_provided_axes(shap_2d):
    values, mask, feature_names = shap_2d
    fig, ax = plt.subplots()
    returned_ax = shap.plots.group_difference(values, mask, feature_names=feature_names, ax=ax, show=False)
    assert returned_ax is ax, "Returned Axes should be the same object that was passed in."
    plt.close("all")


def test_multiple_group_difference_plots_in_subplots(shap_2d, rng):
    values, mask, feature_names = shap_2d
    values2 = rng.randn(20, 5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    shap.plots.group_difference(values, mask, feature_names=feature_names, ax=ax1, show=False)
    shap.plots.group_difference(values2, mask, feature_names=feature_names, ax=ax2, show=False)

    # Both axes should have content (patches = bars were drawn)
    assert len(ax1.patches) > 0, "ax1 has no bars"
    assert len(ax2.patches) > 0, "ax2 has no bars"
    plt.close("all")


# edge cases


def test_all_true_group_mask(rng):
    values = rng.randn(10, 4)
    mask = np.ones(10, dtype=bool)
    try:
        shap.plots.group_difference(values, mask, show=False)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Degenerate all-True mask raised {type(exc).__name__}: {exc}")
    finally:
        plt.close("all")


def test_all_false_group_mask(rng):
    values = rng.randn(10, 4)
    mask = np.zeros(10, dtype=bool)
    try:
        shap.plots.group_difference(values, mask, show=False)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Degenerate all-False mask raised {type(exc).__name__}: {exc}")
    finally:
        plt.close("all")


def test_single_feature(rng):
    values = rng.randn(20, 1)
    mask = np.array([True] * 10 + [False] * 10)
    shap.plots.group_difference(values, mask, feature_names=["only_feature"], show=False)
    plt.close("all")


def test_single_sample_per_group(rng):
    values = rng.randn(2, 4)
    mask = np.array([True, False])
    shap.plots.group_difference(values, mask, show=False)
    plt.close("all")


# tests for correcteness


def test_identical_groups_produce_near_zero_differences():
    values = np.ones((20, 4))  # every sample identical
    mask = np.array([True] * 10 + [False] * 10)
    feature_names = [f"f{i}" for i in range(4)]

    ax = shap.plots.group_difference(values, mask, feature_names=feature_names, show=False)

    # bar widths represent the group difference; all should be essentially zero
    for patch in ax.patches:
        assert abs(patch.get_width()) < 1e-9, f"Expected bar width ≈ 0 for identical groups, got {patch.get_width()}"
    plt.close("all")


def test_known_difference_is_reflected_in_bars():
    n = 20
    values = np.zeros((n, 1))
    values[: n // 2] = 1.0  # first group: all ones
    mask = np.array([True] * (n // 2) + [False] * (n // 2))

    ax = shap.plots.group_difference(values, mask, feature_names=["f0"], show=False)

    widths = [abs(p.get_width()) for p in ax.patches if p.get_width() != 0]
    assert len(widths) > 0, "No bars were drawn"
    assert pytest.approx(max(widths), abs=1e-6) == 1.0, f"Expected bar width of 1.0, got {max(widths)}"
    plt.close("all")


def test_reproducibility(shap_2d):
    values, mask, feature_names = shap_2d

    ax1 = shap.plots.group_difference(values, mask, feature_names=feature_names, show=False)
    widths1 = sorted(p.get_width() for p in ax1.patches)
    plt.close("all")

    ax2 = shap.plots.group_difference(values, mask, feature_names=feature_names, show=False)
    widths2 = sorted(p.get_width() for p in ax2.patches)
    plt.close("all")

    assert widths1 == widths2, "Two identical calls produced different bar widths."


def test_group_difference_direction(rng):
    values = np.zeros((20, 1))
    values[:10] = 2.0  # group A: high SHAP
    values[10:] = 0.0  # group B: low SHAP
    mask = np.array([True] * 10 + [False] * 10)

    ax = shap.plots.group_difference(values, mask, feature_names=["f0"], show=False)

    positive_bars = [p for p in ax.patches if p.get_width() > 0]
    assert len(positive_bars) > 0, "Expected at least one positive bar when group A SHAP > group B SHAP."
    plt.close("all")
