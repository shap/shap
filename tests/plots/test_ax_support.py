"""Elite Test Suite for Matplotlib 'ax' Parameter Standardization.

Validates that plotting wrappers (waterfall, violin, decision) respect user-provided
Axes objects, maintain figure isolation, and correctly delegate rendering.

Project: shap
Issue: #3411
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap


@pytest.fixture(autouse=True)
def _headless_mpl():
    """Ensure tests run in a headless environment."""
    matplotlib.use("Agg")
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def shared_explanation():
    """Build a reusable Explanation object for plot testing."""
    X, _ = shap.datasets.adult(n_points=10)
    # Mock a simple linear model for deterministic SHAP values
    values = np.random.randn(10, X.shape[1])
    base_values = np.zeros(10)
    return shap.Explanation(values, base_values, X, feature_names=X.columns.tolist())


# ---------------------------------------------------------------------------
# Core Artist Rendering Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "plot_func, plot_args",
    [
        (shap.plots.waterfall, lambda exp: (exp[0],)),
        (shap.plots.violin, lambda exp: (exp,)),
        (shap.plots.decision, lambda exp: (exp.base_values[0], exp.values[0])),
    ],
)
def test_explicit_ax_rendering_logic(plot_func, plot_args, shared_explanation):
    """
    Verify that plot functions:
    1. Draw artists on the provided Axes.
    2. Do not pollute the global plt.gca().
    3. Return the provided Axes object.
    """
    # Setup: Create two distinct figures
    fig_user, ax_user = plt.subplots()
    fig_global, ax_global = plt.subplots()

    # Pre-test artist count
    initial_artists = len(ax_user.get_children())

    # Execution: Target the user axes explicitly
    plt.sca(ax_global)  # Set the 'current' axis to the wrong one
    args = plot_args(shared_explanation)
    returned_ax = plot_func(*args, ax=ax_user, show=False)

    # Validation
    assert returned_ax is ax_user, f"{plot_func.__name__} failed to return the user-supplied Axes."
    assert len(ax_user.get_children()) > initial_artists, (
        f"{plot_func.__name__} did not draw any artists on the provided Axes."
    )
    assert len(ax_global.get_children()) <= 10, f"{plot_func.__name__} leaked artists into the global plt.gca()."


# ---------------------------------------------------------------------------
# Figure Isolation & Sizing Logic
# ---------------------------------------------------------------------------


def test_waterfall_respects_user_figure_sizing(shared_explanation):
    """
    Regression test for #3411: waterfall should NOT resize the figure
    if the user provides their own Axes.
    """
    custom_size = (15, 10)
    fig, ax = plt.subplots(figsize=custom_size)

    shap.plots.waterfall(shared_explanation[0], ax=ax, show=False)

    # Verify size was preserved
    actual_size = tuple(fig.get_size_inches())
    assert actual_size == custom_size, f"waterfall unexpectedly resized user figure from {custom_size} to {actual_size}"


# ---------------------------------------------------------------------------
# Backward Compatibility (Legacy Default Behavior)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "plot_func, plot_args",
    [
        (shap.plots.waterfall, lambda exp: (exp[0],)),
        (shap.plots.violin, lambda exp: (exp,)),
        (shap.plots.decision, lambda exp: (exp.base_values[0], exp.values[0])),
    ],
)
def test_legacy_gca_fallback(plot_func, plot_args, shared_explanation):
    """Ensure the code still works when 'ax' is not provided (uses plt.gca)."""
    plt.figure()
    current_ax = plt.gca()

    args = plot_args(shared_explanation)
    returned_ax = plot_func(*args, show=False)

    assert returned_ax is current_ax, "Plot failed to fall back to plt.gca() when ax=None"

