import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap


@pytest.fixture
def group_difference_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    shap_values = np.linspace(-0.5, 0.5, 60).reshape(20, 3)
    group_mask = np.array([True] * 10 + [False] * 10)
    feature_names = ["Feature A", "Feature B", "Feature C"]
    return shap_values, group_mask, feature_names


def test_group_difference_returns_axes(group_difference_data):
    shap_values, group_mask, feature_names = group_difference_data
    np.random.seed(0)

    returned_ax = shap.plots.group_difference(shap_values, group_mask, feature_names, show=False)

    assert isinstance(returned_ax, plt.Axes)


def test_group_difference_returns_user_axes(group_difference_data):
    shap_values, group_mask, feature_names = group_difference_data
    fig, ax = plt.subplots()
    np.random.seed(0)

    returned_ax = shap.plots.group_difference(shap_values, group_mask, feature_names, show=False, ax=ax)

    assert returned_ax is ax
    assert returned_ax.get_figure() is fig


@pytest.mark.filterwarnings("ignore:Conversion of an array with ndim > 0 to a scalar is deprecated:DeprecationWarning")
def test_group_difference_accepts_output_vector(monkeypatch):
    model_outputs = np.linspace(-0.5, 0.5, 20)
    group_mask = np.array([True] * 10 + [False] * 10)
    show_called = False

    def mock_show():
        nonlocal show_called
        show_called = True

    monkeypatch.setattr(plt, "show", mock_show)
    np.random.seed(0)

    returned_ax = shap.plots.group_difference(
        model_outputs,
        group_mask,
        xlabel="Mean output difference",
        xmin=-1,
        xmax=1,
        max_display=1,
        sort=False,
    )

    assert isinstance(returned_ax, plt.Axes)
    assert returned_ax.get_xlabel() == "Mean output difference"
    assert returned_ax.get_xlim() == (-1, 1)
    assert show_called


@pytest.mark.mpl_image_compare
def test_group_difference(group_difference_data):
    """Check that the group_difference plot renders on a custom Axes."""
    shap_values, group_mask, feature_names = group_difference_data
    np.random.seed(0)

    fig, ax = plt.subplots()
    returned_ax = shap.plots.group_difference(shap_values, group_mask, feature_names, show=False, ax=ax)
    assert returned_ax is ax
    plt.tight_layout()
    return fig
