import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")

import shap


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


@pytest.fixture
def shap_values_2d():
    np.random.seed(42)
    return np.random.randn(100, 5)


@pytest.fixture
def group_mask():
    mask = np.zeros(100, dtype=bool)
    mask[:50] = True
    return mask


def test_1d_input_no_feature_names(group_mask):
    """1D input with no feature_names should set label to empty string."""
    np.random.seed(0)
    shap_values_1d = np.random.randn(100)
    ax = plt.subplots()[1]
    shap.plots.group_difference(shap_values_1d, group_mask, show=False, ax=ax)
    tick_labels = [t.get_text() for t in ax.get_yticklabels()]
    assert "" in tick_labels


def test_auto_feature_names(shap_values_2d, group_mask):
    """Auto-generate Feature i names when none provided."""
    ax = plt.subplots()[1]
    shap.plots.group_difference(shap_values_2d, group_mask, feature_names=None, show=False, ax=ax)
    tick_labels = [t.get_text() for t in ax.get_yticklabels()]
    assert any(label.startswith("Feature") for label in tick_labels)


def test_sort_false(shap_values_2d, group_mask):
    """sort=False should preserve original feature order."""
    feature_names = ["a", "b", "c", "d", "e"]
    ax = plt.subplots()[1]
    shap.plots.group_difference(shap_values_2d, group_mask, feature_names=feature_names, sort=False, show=False, ax=ax)
    tick_labels = [t.get_text() for t in ax.get_yticklabels()]
    assert set(tick_labels) == set(feature_names)


def test_max_display(shap_values_2d, group_mask):
    """max_display should limit the number of bars shown."""
    ax = plt.subplots()[1]
    shap.plots.group_difference(shap_values_2d, group_mask, max_display=3, show=False, ax=ax)
    assert len(ax.patches) == 3


def test_no_ax_creates_figure(shap_values_2d, group_mask):
    """When no ax is provided, function should create its own figure."""
    shap.plots.group_difference(shap_values_2d, group_mask, show=False)
    assert plt.get_fignums()


def test_show_true_calls_plt_show(shap_values_2d, group_mask, monkeypatch):
    """show=True without ax should call plt.show()."""
    called = []
    monkeypatch.setattr(plt, "show", lambda: called.append(1))
    shap.plots.group_difference(shap_values_2d, group_mask, show=True)
    assert len(called) == 1
