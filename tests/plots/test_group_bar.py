"""tests for the group_bar plot."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

import shap


@pytest.fixture
def coalition_setup():
    """shared setup for coalition explainer tests."""
    X, y = load_iris(return_X_y=True)
    X, y = X[:50], y[:50]
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)

    feature_names = ["sepal length", "sepal width", "petal length", "petal width"]
    masker = shap.maskers.Partition(X)
    masker.feature_names = feature_names

    tree = {
        "Sepal": ["sepal length", "sepal width"],
        "Petal": ["petal length", "petal width"],
    }

    explainer = shap.CoalitionExplainer(
        model.predict,
        masker,
        partition_tree=tree,
        feature_names=feature_names,
    )
    sv = explainer(X[:10])
    return sv, tree


def test_group_bar_wrong_input_type():
    """should raise TypeError if shap_values is not an Explanation object."""
    with pytest.raises(TypeError, match="group_bar requires an Explanation object"):
        shap.plots.group_bar([1, 2, 3], partition_tree={"g": ["a"]}, show=False)


def test_group_bar_missing_feature_in_tree(coalition_setup):
    """should raise ValueError if partition_tree references a feature not in shap_values."""
    sv, _ = coalition_setup
    bad_tree = {"Sepal": ["sepal length", "does not exist"]}
    with pytest.raises(ValueError, match="these features appear in partition_tree but not in shap_values"):
        shap.plots.group_bar(sv, partition_tree=bad_tree, show=False)


def test_group_bar_returns_ax(coalition_setup):
    """should return a matplotlib Axes object when show=False."""
    sv, tree = coalition_setup
    ax = shap.plots.group_bar(sv, partition_tree=tree, show=False)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_group_bar_single_row(coalition_setup):
    """should work with a single-row Explanation (1d values)."""
    sv, tree = coalition_setup
    ax = shap.plots.group_bar(sv[0], partition_tree=tree, show=False)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_group_bar_multi_row(coalition_setup):
    """should work with multi-row Explanation and collapse to mean."""
    sv, tree = coalition_setup
    ax = shap.plots.group_bar(sv, partition_tree=tree, show=False)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")


def test_group_bar_show_individual_false(coalition_setup):
    """show_individual=False should produce fewer bars (only group rows)."""
    sv, tree = coalition_setup

    ax_full = shap.plots.group_bar(sv, partition_tree=tree, show_individual=True, show=False)
    n_bars_full = len(ax_full.patches)
    plt.close("all")

    ax_groups = shap.plots.group_bar(sv, partition_tree=tree, show_individual=False, show=False)
    n_bars_groups = len(ax_groups.patches)
    plt.close("all")

    assert n_bars_groups < n_bars_full


def test_group_bar_max_display(coalition_setup):
    """max_display should limit the number of groups shown."""
    sv, _ = coalition_setup
    big_tree = {
        "Sepal": ["sepal length", "sepal width"],
        "Petal": ["petal length", "petal width"],
    }
    ax = shap.plots.group_bar(sv, partition_tree=big_tree, max_display=1, show=False)
    # only 1 group shown, so ytick labels should have exactly 1 bold label
    tick_labels = ax.yaxis.get_majorticklabels()
    bold_labels = [t for t in tick_labels if t.get_fontweight() in ("bold", 700)]
    assert len(bold_labels) == 1
    plt.close("all")


def test_group_bar_nested_tree():
    """should handle nested partition_tree without crashing."""
    rs = np.random.RandomState(0)
    feature_names = ["a", "b", "c", "d"]
    sv = shap.Explanation(
        values=rs.randn(5, 4),
        base_values=np.zeros(5),
        data=rs.randn(5, 4),
        feature_names=feature_names,
    )
    nested_tree = {
        "Group1": {"SubA": ["a", "b"]},
        "Group2": ["c", "d"],
    }
    ax = shap.plots.group_bar(sv, partition_tree=nested_tree, show=False)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close("all")
