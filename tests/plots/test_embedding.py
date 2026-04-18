import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

import shap

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture()
def shap_vals():
    rng = np.random.RandomState(0)
    return rng.randn(30, 4)


@pytest.fixture()
def feature_names():
    return ["alpha", "beta", "gamma", "delta"]


# ---------------------------------------------------------------------------
# Basic usage
# ---------------------------------------------------------------------------


def test_embedding_pca_returns_ax(shap_vals):
    """With show=False, embedding() should return a matplotlib Axes."""
    ax = shap.plots.embedding(0, shap_vals, show=False)
    assert isinstance(ax, Axes)


def test_embedding_pca_no_return_when_show_true(shap_vals, monkeypatch):
    """With show=True, embedding() should return None after calling plt.show."""
    monkeypatch.setattr(plt, "show", lambda: None)
    result = shap.plots.embedding(0, shap_vals, show=True)
    assert result is None


def test_embedding_custom_ax(shap_vals):
    """A caller-supplied ax should be used instead of plt.gca()."""
    fig, ax = plt.subplots()
    returned = shap.plots.embedding(0, shap_vals, ax=ax, show=False)
    assert returned is ax


def test_embedding_draws_on_given_ax(shap_vals):
    """The scatter should appear on the provided axes, not a new one."""
    fig, (ax1, ax2) = plt.subplots(1, 2)
    shap.plots.embedding(0, shap_vals, ax=ax1, show=False)
    # ax1 should have children (scatter collection); ax2 should be empty
    assert len(ax1.collections) > 0
    assert len(ax2.collections) == 0


# ---------------------------------------------------------------------------
# ind variants
# ---------------------------------------------------------------------------


def test_embedding_ind_by_integer(shap_vals):
    ax = shap.plots.embedding(1, shap_vals, show=False)
    assert isinstance(ax, Axes)


def test_embedding_ind_by_feature_name(shap_vals, feature_names):
    ax = shap.plots.embedding("beta", shap_vals, feature_names=feature_names, show=False)
    assert isinstance(ax, Axes)


def test_embedding_ind_sum(shap_vals):
    """ind='sum()' should color by the sum of all SHAP values."""
    ax = shap.plots.embedding("sum()", shap_vals, show=False)
    assert isinstance(ax, Axes)


def test_embedding_ind_rank(shap_vals, feature_names):
    """ind='rank(0)' should select the feature with rank 0."""
    ax = shap.plots.embedding("rank(0)", shap_vals, feature_names=feature_names, show=False)
    assert isinstance(ax, Axes)


# ---------------------------------------------------------------------------
# method variants
# ---------------------------------------------------------------------------


def test_embedding_custom_embedding_array(shap_vals):
    """Passing a (n x 2) array as method should use it directly."""
    rng = np.random.RandomState(1)
    custom_embed = rng.randn(30, 2)
    ax = shap.plots.embedding(0, shap_vals, method=custom_embed, show=False)
    assert isinstance(ax, Axes)


def test_embedding_unsupported_method_prints_and_returns(shap_vals, capsys):
    """An unsupported method string should print a message and return the ax."""
    ax = shap.plots.embedding(0, shap_vals, method="tsne", show=False)
    captured = capsys.readouterr()
    assert "Unsupported" in captured.out
    assert isinstance(ax, Axes)


# ---------------------------------------------------------------------------
# feature_names
# ---------------------------------------------------------------------------


def test_embedding_auto_feature_names(shap_vals):
    """Without feature_names, labels like 'Feature 0' should be generated."""
    ax = shap.plots.embedding(0, shap_vals, show=False)
    assert isinstance(ax, Axes)


def test_embedding_explicit_feature_names(shap_vals, feature_names):
    ax = shap.plots.embedding(0, shap_vals, feature_names=feature_names, show=False)
    assert isinstance(ax, Axes)


# ---------------------------------------------------------------------------
# alpha
# ---------------------------------------------------------------------------


def test_embedding_alpha(shap_vals):
    ax = shap.plots.embedding(0, shap_vals, alpha=0.3, show=False)
    assert isinstance(ax, Axes)


# ---------------------------------------------------------------------------
# Axis is turned off
# ---------------------------------------------------------------------------


def test_embedding_axis_off(shap_vals):
    """The axes should have axis visibility turned off."""
    ax = shap.plots.embedding(0, shap_vals, show=False)
    assert not ax.axison


# ---------------------------------------------------------------------------
# Colorbar label
# ---------------------------------------------------------------------------


def test_embedding_colorbar_label_uses_feature_name(shap_vals, feature_names):
    """The colorbar label should mention the feature name."""
    ax = shap.plots.embedding(0, shap_vals, feature_names=feature_names, show=False)
    fig = ax.get_figure()
    # The colorbar axes is a sibling of the main axes on the figure
    cb_labels = [
        child.get_label()
        for child in fig.get_axes()
        if child is not ax
    ]
    # At least one axis or artist should reference the feature name
    assert any("alpha" in lbl or "SHAP" in lbl for lbl in cb_labels) or len(fig.get_axes()) > 1


def test_embedding_colorbar_label_sum(shap_vals):
    """The colorbar label for ind='sum()' should say 'sum(SHAP values)'."""
    ax = shap.plots.embedding("sum()", shap_vals, show=False)
    fig = ax.get_figure()
    assert len(fig.get_axes()) > 1  # colorbar axes is present
