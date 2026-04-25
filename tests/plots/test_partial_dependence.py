"""Tests for shap.partial_dependence_plot."""

import matplotlib

matplotlib.use("Agg")  # headless backend so tests don't pop windows
import matplotlib.pyplot as plt
import numpy as np

import shap


def test_partial_dependence_does_not_crash():
    """Smoke test: the basic call path renders without error."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3))

    def model(x):
        return x[:, 0] + 0.5 * x[:, 1]

    shap.partial_dependence_plot(0, model, X, ice=False, show=False)
    plt.close("all")


def test_partial_dependence_respects_custom_ax():
    """Regression test for GH #3206.

    Passing ``ax=`` must draw onto that axis. Previously the function
    called ``plt.gca()`` in the ``else`` branch and ignored the caller's
    axis, so multiple plots in a loop all stacked onto the last subplot.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3))

    def model(x):
        return x[:, 0] + 0.5 * x[:, 1]

    fig, axes = plt.subplots(1, 2)
    try:
        shap.partial_dependence_plot(
            0,
            model,
            X,
            ice=False,
            show=False,
            ax=axes[0],
        )
        shap.partial_dependence_plot(
            1,
            model,
            X,
            ice=False,
            show=False,
            ax=axes[1],
        )
        # Both axes must have been drawn into. With the bug, only one would.
        assert axes[0].has_data(), "ax=axes[0] should have been drawn into"
        assert axes[1].has_data(), "ax=axes[1] should have been drawn into"
    finally:
        plt.close("all")
