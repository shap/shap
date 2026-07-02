import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import sklearn

import shap

matplotlib.use("Agg")


@pytest.fixture
def simple_data():
    """Minimal synthetic SHAP values for fast unit tests."""
    rs = np.random.RandomState(42)
    base = 0.0
    shap_values = rs.standard_normal(size=(10, 5))
    features = rs.standard_normal(size=(10, 5))
    return base, shap_values, features


@pytest.fixture
def values_features():
    X, y = shap.datasets.adult(n_points=10)
    rfc = sklearn.ensemble.RandomForestClassifier(random_state=0)
    rfc.fit(X, y)
    ex = shap.TreeExplainer(rfc)
    shap_values = ex(X)
    return shap_values, X


def test_random_decision(random_seed):
    """Make sure the decision plot does not crash on random data (legacy call)."""
    rs = np.random.RandomState(random_seed)
    shap.decision_plot(0, rs.standard_normal(size=(20, 5)), rs.standard_normal(size=(20, 5)), show=False)
    plt.close("all")


def test_backward_compat_no_ax(simple_data):
    """Calling without ax still works and does not raise."""
    base, sv, feat = simple_data
    result = shap.plots.decision(base, sv, show=False)
    assert result is not None, "show=False without ax must return an Axes"
    plt.close("all")


def test_show_false_returns_axes(simple_data):
    """show=False returns a matplotlib Axes object."""
    base, sv, _ = simple_data
    fig, ax = plt.subplots()
    result = shap.plots.decision(base, sv, ax=ax, show=False)
    assert isinstance(result, matplotlib.axes.Axes), f"Expected Axes, got {type(result)}"
    plt.close("all")


def test_show_false_no_ax_returns_axes(simple_data):
    """show=False without explicit ax still returns Axes."""
    base, sv, _ = simple_data
    result = shap.plots.decision(base, sv, show=False)
    assert isinstance(result, matplotlib.axes.Axes)
    plt.close("all")


def test_show_true_returns_none(simple_data, monkeypatch):
    """show=True returns None (plt.show is monkeypatched to a no-op)."""
    monkeypatch.setattr(plt, "show", lambda: None)
    base, sv, _ = simple_data
    result = shap.plots.decision(base, sv, show=True)
    assert result is None
    plt.close("all")


def test_return_objects_precedence(simple_data):
    """return_objects=True returns DecisionPlotResult even with show=False."""
    base, sv, _ = simple_data
    fig, ax = plt.subplots()
    result = shap.plots.decision(base, sv, ax=ax, show=False, return_objects=True)
    assert isinstance(result, shap.plots._decision.DecisionPlotResult)
    plt.close("all")


def test_ax_identity_preserved(simple_data):
    """The returned Axes is the same object that was passed in."""
    base, sv, _ = simple_data
    fig, ax = plt.subplots()
    result = shap.plots.decision(base, sv, ax=ax, show=False)
    assert result is ax, "Returned Axes must be the same object passed via ax="
    plt.close("all")


def test_subplot_isolation(simple_data):
    """Drawing on ax1 must not alter xlim/ylim/title of sibling ax2."""
    base, sv, _ = simple_data
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # record sibling state before plot
    xlim_before = ax2.get_xlim()
    ylim_before = ax2.get_ylim()
    title_before = ax2.get_title()

    shap.plots.decision(base, sv, ax=ax1, show=False)

    assert ax2.get_xlim() == xlim_before, "Sibling ax2 xlim was modified"
    assert ax2.get_ylim() == ylim_before, "Sibling ax2 ylim was modified"
    assert ax2.get_title() == title_before, "Sibling ax2 title was modified"
    plt.close("all")


def test_figure_size_unchanged_when_ax_provided(simple_data):
    """auto_size_plot must not resize the figure when ax is explicitly provided."""
    base, sv, _ = simple_data
    fig, ax = plt.subplots(figsize=(3, 3))
    w_before, h_before = fig.get_size_inches()

    shap.plots.decision(base, sv, ax=ax, auto_size_plot=True, show=False)

    w_after, h_after = fig.get_size_inches()
    assert (w_after, h_after) == (w_before, h_before), (
        f"Figure was resized from {(w_before, h_before)} to {(w_after, h_after)} even though ax was provided"
    )
    plt.close("all")


def test_figure_resized_when_no_ax(simple_data):
    """auto_size_plot=True should resize when no ax is provided."""
    base, sv, _ = simple_data
    fig = plt.figure(figsize=(3, 3))
    w_before, h_before = fig.get_size_inches()

    shap.plots.decision(base, sv, auto_size_plot=True, show=False)

    w_after, h_after = plt.gcf().get_size_inches()
    # The default resize produces an 8-inch wide figure
    assert w_after == pytest.approx(8.0), "Figure width not resized to 8 as expected"
    plt.close("all")


def test_multiple_calls_same_ax(simple_data):
    """Calling decision_plot twice on the same ax must not raise."""
    base, sv, _ = simple_data
    fig, ax = plt.subplots()
    r1 = shap.plots.decision(base, sv, ax=ax, show=False)
    r2 = shap.plots.decision(base, sv, ax=ax, show=False)
    assert r1 is ax
    assert r2 is ax
    plt.close("all")


def test_legend_attaches_to_correct_ax(simple_data):
    """When legend_labels are given, the legend must be on the target ax."""
    base, sv, _ = simple_data
    fig, (ax1, ax2) = plt.subplots(1, 2)
    labels = [f"obs {i}" for i in range(sv.shape[0])]

    shap.plots.decision(base, sv, ax=ax1, show=False, legend_labels=labels)

    assert ax1.get_legend() is not None, "Legend missing on target ax1"
    assert ax2.get_legend() is None, "Legend incorrectly placed on sibling ax2"
    plt.close("all")


def test_no_pyplot_state_leakage(simple_data):
    """After plotting with an explicit ax, the pyplot current axis must not
    have been silently changed to some unrelated new axes."""
    base, sv, _ = simple_data
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.sca(ax2)  # user sets ax2 as current

    shap.plots.decision(base, sv, ax=ax1, show=False)

    # After the call, both original axes must still be present — no spurious new axes.
    assert ax1 in fig.axes, "ax1 is missing from the figure after decision plot"
    assert ax2 in fig.axes, "ax2 is missing from the figure after decision plot"
    # The colorbar inset is allowed (created inside ax1's data space), so we permit
    # len == 2 (no colorbar) or len == 3 (with inset colorbar axes).
    assert len(fig.axes) in (2, 3), f"Unexpected number of axes: {len(fig.axes)}"
    plt.close("all")


@pytest.mark.mpl_image_compare
def test_decision_plot(values_features):
    fig = plt.figure()
    shap_values, _X = values_features

    shap.decision_plot(
        shap_values.base_values[0, 1],
        shap_values.values[:, :, 1],
        show=False,
        return_objects=True,
        title="Decision Plot",
        link="identity",
    )
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_decision_plot_single_instance(values_features):
    fig = plt.figure()
    shap_values, X = values_features

    shap.decision_plot(
        shap_values.base_values[0, 1],
        shap_values.values[0, :, 1],
        features=X.iloc[0],
        show=False,
        new_base_value=0,
        return_objects=True,
    )
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_decision_plot_interactions():
    fig = plt.figure()

    X, y = shap.datasets.adult(n_points=10)
    rfc = sklearn.ensemble.RandomForestClassifier(random_state=0)
    rfc.fit(X, y)
    ex = shap.TreeExplainer(rfc)
    result_values = ex(X, interactions=True)
    shap.decision_plot(
        result_values.base_values[0, 1],
        result_values.values[:, :, :, 1],
        features=X,
        show=False,
    )
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
def test_decision_multioutput(values_features):
    adult_rfc_shap_values, X = values_features
    fig = plt.figure()
    adult_rfc_shap_values_list = [adult_rfc_shap_values.values[:, :, i] for i in range(adult_rfc_shap_values.shape[2])]
    base_values_list = list(adult_rfc_shap_values.base_values[0, :])
    shap.multioutput_decision_plot(base_values_list, adult_rfc_shap_values_list, row_index=0, features=X, show=False)
    plt.tight_layout()
    return fig


def test_multioutput_decision_raises(values_features):
    adult_rfc_shap_values, X = values_features
    with pytest.raises(ValueError, match="The base_values and shap_values args expect lists."):
        shap.multioutput_decision_plot(
            adult_rfc_shap_values.base_values[0, :],
            adult_rfc_shap_values.values[:, :, :],
            row_index=0,
            features=X,
        )
    with pytest.raises(
        ValueError, match="The shap_values arg should be a list of two or three dimensional SHAP arrays."
    ):
        adult_rfc_shap_values_list = [
            adult_rfc_shap_values.values[:, 0, i] for i in range(adult_rfc_shap_values.shape[2])
        ]
        base_values_list = list(adult_rfc_shap_values.base_values[0, :])
        shap.multioutput_decision_plot(
            base_values_list,
            adult_rfc_shap_values_list,
            row_index=0,
        )
    plt.close("all")
