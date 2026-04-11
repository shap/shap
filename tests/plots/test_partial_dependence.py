"""Tests for shap.plots.partial_dependence."""

import matplotlib.pyplot as plt
import numpy as np

import shap


def test_partial_dependence_respects_ax_keyword():
    """Passing ax= must draw on that axes (regression for ignored ax / plt.gca())."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(25, 3))

    def model(x):
        return x[:, 0] * 2.0 + x[:, 1]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(8, 3))
    plt.sca(ax_right)

    out = shap.plots.partial_dependence(0, model, X, ax=ax_left, ice=False, hist=False, show=False)

    assert out is ax_left
    assert len(ax_left.get_lines()) >= 1
    assert len(ax_right.get_lines()) == 0

    plt.close(fig)


def test_partial_dependence_show_false_returns_axes():
    """With show=False, return the primary Axes (not a figure tuple)."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(15, 2))

    def model(x):
        return x[:, 0]

    fig, ax = plt.subplots()
    result = shap.plots.partial_dependence(0, model, X, ax=ax, ice=False, hist=False, show=False)
    assert not isinstance(result, tuple)
    assert result is ax
    plt.close(fig)
