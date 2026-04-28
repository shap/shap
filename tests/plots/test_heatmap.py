"""Tests for shap.plots.heatmap — API consistency suite.

Covers:
  - Image-comparison regression tests (unchanged visual output)
  - ax= parameter contract (return value, identity, isolation, resize guard,
    repeated calls, colorbar placement, backward compatibility)
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap

matplotlib.use("Agg")


def _make_explanation(n_samples=40, n_features=6, seed=0):
    """Return a small synthetic Explanation object sufficient for heatmap."""
    rng = np.random.RandomState(seed)
    values = rng.randn(n_samples, n_features)
    data = rng.randn(n_samples, n_features)
    feature_names = [f"feature_{i}" for i in range(n_features)]
    return shap.Explanation(
        values=values,
        data=data,
        feature_names=feature_names,
    )


@pytest.mark.mpl_image_compare
def test_heatmap(explainer):
    """Make sure the heatmap plot is unchanged."""
    fig = plt.figure()
    shap_values = explainer(explainer.data)
    shap.plots.heatmap(shap_values, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_heatmap_feature_order(explainer):
    """Make sure the heatmap plot is unchanged when we apply a feature ordering."""
    fig = plt.figure()
    shap_values = explainer(explainer.data)
    shap.plots.heatmap(
        shap_values,
        max_display=5,
        feature_order=np.array(range(shap_values.shape[1]))[::-1],
        show=False,
    )
    plt.tight_layout()
    return fig


class TestHeatmapReturnValue:
    """show=False must return an Axes object, never None."""

    def test_returns_axes_when_show_false(self):
        sv = _make_explanation()
        fig, ax = plt.subplots()
        result = shap.plots.heatmap(sv, ax=ax, show=False)
        assert isinstance(result, matplotlib.axes.Axes), (
            f"heatmap(show=False) must return an Axes object, got {type(result)!r}"
        )
        plt.close(fig)

    def test_returns_axes_without_explicit_ax(self):
        sv = _make_explanation()
        fig = plt.figure()
        result = shap.plots.heatmap(sv, show=False)
        assert isinstance(result, matplotlib.axes.Axes), "heatmap(show=False) without ax= must return an Axes object"
        plt.close(fig)


class TestHeatmapAxIdentity:
    """The returned Axes must be the same object that was passed in."""

    def test_ax_identity_preserved(self):
        sv = _make_explanation()
        fig, ax = plt.subplots()
        result = shap.plots.heatmap(sv, ax=ax, show=False)
        assert result is ax, "heatmap must return the exact Axes object that was supplied via ax="
        plt.close(fig)


class TestHeatmapSubplotIsolation:
    """Drawing on one subplot must not modify sibling axes."""

    def test_sibling_axes_not_modified(self):
        sv = _make_explanation()
        fig, (ax1, ax2) = plt.subplots(1, 2)

        # Record sibling state before
        xlim_before = ax2.get_xlim()
        ylim_before = ax2.get_ylim()
        title_before = ax2.get_title()
        n_lines_before = len(ax2.lines)

        shap.plots.heatmap(sv, ax=ax1, show=False)

        # Sibling must be untouched
        assert ax2.get_xlim() == xlim_before, "sibling ax xlim changed"
        assert ax2.get_ylim() == ylim_before, "sibling ax ylim changed"
        assert ax2.get_title() == title_before, "sibling ax title changed"
        assert len(ax2.lines) == n_lines_before, "sibling ax gained unexpected lines"
        plt.close(fig)


class TestHeatmapResizeGuard:
    """Figure size must not change when an explicit ax is provided."""

    def test_figure_size_unchanged_when_ax_provided(self):
        sv = _make_explanation()
        fig, ax = plt.subplots()
        original_size = fig.get_size_inches().tolist()

        shap.plots.heatmap(sv, ax=ax, show=False)

        assert fig.get_size_inches().tolist() == original_size, (
            "heatmap must not resize the figure when ax= is explicitly provided"
        )
        plt.close(fig)

    def test_figure_resized_when_ax_not_provided(self):
        """Without ax= the implementation IS allowed to resize (not a failure)."""
        sv = _make_explanation()
        fig = plt.figure(figsize=(4, 4))
        shap.plots.heatmap(sv, show=False)

        plt.close(fig)


class TestHeatmapRepeatedCalls:
    """Calling heatmap multiple times on the same axes must not raise."""

    def test_multiple_calls_on_same_ax(self):
        sv = _make_explanation()
        fig, ax = plt.subplots()
        shap.plots.heatmap(sv, ax=ax, show=False)
        # Second call: should not crash or raise any exception
        shap.plots.heatmap(sv, ax=ax, show=False)
        plt.close(fig)

    def test_multiple_calls_return_same_ax(self):
        sv = _make_explanation()
        fig, ax = plt.subplots()
        result1 = shap.plots.heatmap(sv, ax=ax, show=False)
        result2 = shap.plots.heatmap(sv, ax=ax, show=False)
        assert result1 is ax
        assert result2 is ax
        plt.close(fig)


class TestHeatmapColorbar:
    """Colorbar must be attached to the supplied axes, not a sibling."""

    def test_colorbar_attaches_to_provided_ax(self):
        sv = _make_explanation()
        fig, (ax1, ax2) = plt.subplots(1, 2)

        shap.plots.heatmap(sv, ax=ax1, show=False)

        new_axes = [a for a in fig.axes if a not in (ax1, ax2)]
        assert len(new_axes) >= 1, "expected at least one colorbar axes to be created"

        # ax2 must not have been consumed / stolen as a colorbar axes
        assert ax2 in fig.axes, "sibling ax2 was incorrectly consumed by the colorbar"
        plt.close(fig)

    def test_colorbar_present_without_explicit_ax(self):
        sv = _make_explanation()
        fig = plt.figure()
        shap.plots.heatmap(sv, show=False)
        # At least two axes should exist: the main heatmap axes + colorbar axes
        assert len(fig.axes) >= 2, "expected heatmap axes and colorbar axes when no ax= provided"
        plt.close(fig)


class TestHeatmapBackwardCompatibility:
    """Calling heatmap without the ax= kwarg must still work correctly."""

    def test_no_ax_kwarg_does_not_raise(self):
        sv = _make_explanation()
        fig = plt.figure()
        result = shap.plots.heatmap(sv, show=False)
        assert result is not None
        plt.close(fig)

    def test_no_ax_kwarg_returns_axes(self):
        sv = _make_explanation()
        fig = plt.figure()
        result = shap.plots.heatmap(sv, show=False)
        assert isinstance(result, matplotlib.axes.Axes)
        plt.close(fig)

    def test_explicit_ax_none_equivalent_to_omitting_ax(self):
        """ax=None is the public default and must behave identically to omitting ax."""
        sv = _make_explanation()
        fig = plt.figure()
        result = shap.plots.heatmap(sv, ax=None, show=False)
        assert isinstance(result, matplotlib.axes.Axes)
        plt.close(fig)


class TestHeatmapMaxDisplay:
    """max_display grouping should work correctly with ax= parameter."""

    def test_max_display_with_explicit_ax(self):
        sv = _make_explanation(n_features=12)
        fig, ax = plt.subplots()
        result = shap.plots.heatmap(sv, max_display=5, ax=ax, show=False)
        assert result is ax
        plt.close(fig)

    def test_custom_feature_order_with_explicit_ax(self):
        sv = _make_explanation(n_features=6)
        fig, ax = plt.subplots()
        order = np.arange(sv.shape[1])[::-1]
        result = shap.plots.heatmap(sv, feature_order=order, ax=ax, show=False)
        assert result is ax
        plt.close(fig)
