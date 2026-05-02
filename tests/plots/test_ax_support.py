"""Tests for explicit `ax` parameter support across matplotlib plot wrappers.

Validates that `waterfall`, `violin`, and `decision` plots correctly render
on a user-supplied Axes instead of implicitly relying on ``plt.gca()``.

Closes #3411
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _mpl_backend():
    """Use a non-interactive backend so tests run headlessly."""
    matplotlib.use("Agg")
    yield
    plt.close("all")


@pytest.fixture()
def linear_explanation():
    """Build a minimal Explanation from a linear model for fast testing."""
    np.random.seed(0)
    X = np.random.randn(50, 4)
    coef = np.array([1.0, -2.0, 0.5, 0.0])
    y = X @ coef

    explainer = shap.Explainer(
        lambda x: x @ coef,
        X,
        feature_names=["A", "B", "C", "D"],
    )
    return explainer(X)


# ---------------------------------------------------------------------------
# waterfall – explicit ax
# ---------------------------------------------------------------------------

class TestWaterfallAx:
    """Verify waterfall() respects an explicitly passed Axes."""

    def test_renders_on_custom_ax(self, linear_explanation):
        fig, ax = plt.subplots()
        returned_ax = shap.plots.waterfall(linear_explanation[0], show=False, ax=ax)
        assert returned_ax is ax, "waterfall should return the same Axes it was given"

    def test_does_not_create_new_figure(self, linear_explanation):
        fig, ax = plt.subplots()
        shap.plots.waterfall(linear_explanation[0], show=False, ax=ax)
        # Only the one figure we created should exist
        assert len(plt.get_fignums()) == 1

    def test_subplot_isolation(self, linear_explanation):
        """Two waterfall plots on a 1×2 subplot grid should not clobber each other."""
        fig, (ax1, ax2) = plt.subplots(1, 2)
        shap.plots.waterfall(linear_explanation[0], show=False, ax=ax1)
        shap.plots.waterfall(linear_explanation[1], show=False, ax=ax2)
        # Each axis should have independent children
        assert len(ax1.get_children()) > 0
        assert len(ax2.get_children()) > 0


# ---------------------------------------------------------------------------
# violin – explicit ax
# ---------------------------------------------------------------------------

class TestViolinAx:
    """Verify violin() respects an explicitly passed Axes."""

    def test_renders_on_custom_ax(self, linear_explanation):
        fig, ax = plt.subplots()
        returned_ax = shap.plots.violin(linear_explanation, show=False, ax=ax)
        assert returned_ax is ax, "violin should return the same Axes it was given"

    def test_does_not_create_new_figure(self, linear_explanation):
        fig, ax = plt.subplots()
        shap.plots.violin(linear_explanation, show=False, ax=ax)
        assert len(plt.get_fignums()) == 1

    def test_log_scale_on_custom_ax(self, linear_explanation):
        """use_log_scale should apply to the supplied ax, not plt.gca()."""
        fig, ax = plt.subplots()
        shap.plots.violin(
            linear_explanation, show=False, ax=ax, use_log_scale=True,
        )
        assert ax.get_xscale() == "symlog"


# ---------------------------------------------------------------------------
# decision – explicit ax
# ---------------------------------------------------------------------------

class TestDecisionAx:
    """Verify decision() respects an explicitly passed Axes."""

    def test_renders_on_custom_ax(self, linear_explanation):
        fig, ax = plt.subplots()
        shap.plots.decision(
            linear_explanation.base_values[0],
            linear_explanation.values[0],
            feature_names=["A", "B", "C", "D"],
            show=False,
            ax=ax,
        )
        # The ax should have been drawn on
        assert len(ax.get_children()) > 0

    def test_does_not_create_new_figure(self, linear_explanation):
        fig, ax = plt.subplots()
        shap.plots.decision(
            linear_explanation.base_values[0],
            linear_explanation.values[0],
            feature_names=["A", "B", "C", "D"],
            show=False,
            ax=ax,
        )
        assert len(plt.get_fignums()) == 1


# ---------------------------------------------------------------------------
# Default behaviour (no ax) should still work
# ---------------------------------------------------------------------------

class TestDefaultAxBehavior:
    """Ensure that when ax is *not* passed, the functions still work."""

    def test_waterfall_default(self, linear_explanation):
        ax = shap.plots.waterfall(linear_explanation[0], show=False)
        assert ax is not None

    def test_violin_default(self, linear_explanation):
        ax = shap.plots.violin(linear_explanation, show=False)
        assert ax is not None
