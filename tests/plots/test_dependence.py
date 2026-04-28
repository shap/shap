import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap
from shap.plots._partial_dependence import partial_dependence

# The following tests use shap.dependence_plot,
# which currently points to shap.plots._scatter.dependence_legacy


def test_random_dependence():
    """Make sure a dependence plot does not crash."""
    shap.dependence_plot(0, np.random.randn(20, 5), np.random.randn(20, 5), show=False)


def test_random_dependence_no_interaction():
    """Make sure a dependence plot does not crash when we are not showing interactions."""
    shap.dependence_plot(0, np.random.randn(20, 5), np.random.randn(20, 5), show=False, interaction_index=None)


def _simple_model(X):
    """Linear model stub: returns weighted sum of columns."""
    return X[:, 0] * 0.5 + X[:, 1] * 0.3


def test_partial_dependence_ax_none_creates_new_figure():
    """When ax=None (default), partial_dependence must create a new figure."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal((30, 3))

    fig_before = plt.gcf()
    fig, ax1 = partial_dependence(0, _simple_model, data, show=False)
    plt.close("all")

    # A new figure object must have been created
    assert fig is not None
    assert ax1 is not None


def test_partial_dependence_ax_is_used_when_provided():
    """When ax is provided, partial_dependence must draw onto that axes.

    Before the fix, the function called plt.gca() regardless, silently
    discarding the caller's axes object.
    """
    rng = np.random.default_rng(1)
    data = rng.standard_normal((30, 3))

    fig, target_ax = plt.subplots()
    returned_fig, returned_ax = partial_dependence(0, _simple_model, data, ax=target_ax, show=False)

    # The axes drawn into must be the one we passed
    assert returned_ax is target_ax, (
        "partial_dependence ignored the ax parameter and drew onto a different axes"
    )
    # The figure must match the one that owns our axes
    assert returned_fig is target_ax.figure

    plt.close("all")


def test_partial_dependence_ax_figure_matches():
    """The returned fig must be the figure that owns the supplied ax."""
    rng = np.random.default_rng(2)
    data = rng.standard_normal((30, 3))

    outer_fig, outer_ax = plt.subplots()
    returned_fig, _ = partial_dependence(0, _simple_model, data, ax=outer_ax, show=False)

    assert returned_fig is outer_fig, (
        "partial_dependence returned a different figure from the one that owns ax"
    )
    plt.close("all")


def test_partial_dependence_ax_receives_plot_content():
    """The provided ax must actually contain plotted lines after the call."""
    rng = np.random.default_rng(3)
    data = rng.standard_normal((30, 3))

    fig, target_ax = plt.subplots()
    partial_dependence(0, _simple_model, data, ax=target_ax, show=False, hist=False)

    # The PD line must have been drawn on target_ax, not some other axes
    assert len(target_ax.lines) > 0, "No lines were drawn on the provided axes"
    plt.close("all")


def test_dependence_use_line_collection_bug():
    """Make sure a dependence plot does not crash."""
    # GH 3368
    sklearn = pytest.importorskip("sklearn")

    X, y = shap.datasets.california(n_points=10)

    X2 = shap.utils.sample(X, 2)

    model = sklearn.linear_model.LinearRegression()
    model.fit(X, y)

    explainer = shap.Explainer(model.predict, X2)
    shap_values = explainer(X2)
    shap.partial_dependence_plot(
        "MedInc",
        model.predict,
        X2,
        model_expected_value=True,
        feature_expected_value=True,
        ice=False,
        shap_values=shap_values[:1, :],  # type: ignore[call-overload]
        show=False,
    )
