import matplotlib.pyplot as plt
import numpy as np
import pytest

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_shap_values(n_samples=50, n_features=6, seed=0):
    rng = np.random.RandomState(seed)
    return rng.randn(n_samples, n_features)


def _make_group_mask(n_samples=50, seed=42):
    rng = np.random.RandomState(seed)
    return rng.rand(n_samples) > 0.5


# ---------------------------------------------------------------------------
# Behavioural tests
# ---------------------------------------------------------------------------


def test_group_difference_returns_none_with_ax():
    """Function should return None and not call plt.show() when ax is provided."""
    shap_values = _make_shap_values()
    group_mask = _make_group_mask()
    _, ax = plt.subplots()
    result = shap.plots.group_difference(shap_values, group_mask, show=False, ax=ax)
    assert result is None


def test_group_difference_default_feature_names():
    """Feature names are auto-generated as 'Feature i' when not provided."""
    n_features = 5
    shap_values = _make_shap_values(n_features=n_features)
    group_mask = _make_group_mask()
    _, ax = plt.subplots()
    shap.plots.group_difference(shap_values, group_mask, show=False, ax=ax)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    for label in labels:
        assert label.startswith("Feature "), f"Unexpected label: {label!r}"


def test_group_difference_custom_feature_names():
    """Provided feature names appear on the y-axis."""
    n_features = 4
    feature_names = [f"col_{i}" for i in range(n_features)]
    shap_values = _make_shap_values(n_features=n_features)
    group_mask = _make_group_mask()
    _, ax = plt.subplots()
    shap.plots.group_difference(shap_values, group_mask, feature_names=feature_names, show=False, ax=ax)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    for label in labels:
        assert label in feature_names, f"Unexpected label: {label!r}"


def test_group_difference_max_display():
    """max_display limits the number of features shown on the plot."""
    n_features = 10
    max_display = 4
    shap_values = _make_shap_values(n_features=n_features)
    group_mask = _make_group_mask()
    _, ax = plt.subplots()
    shap.plots.group_difference(
        shap_values, group_mask, max_display=max_display, show=False, ax=ax
    )
    assert len(ax.get_yticklabels()) == max_display


def test_group_difference_sort_false_preserves_order():
    """With sort=False features appear in their original index order."""
    n_features = 5
    feature_names = [f"f{i}" for i in range(n_features)]
    shap_values = _make_shap_values(n_features=n_features)
    group_mask = _make_group_mask()
    _, ax = plt.subplots()
    shap.plots.group_difference(
        shap_values, group_mask, feature_names=feature_names, sort=False, show=False, ax=ax
    )
    labels = [t.get_text() for t in ax.get_yticklabels()]
    # Labels are returned in the order they were assigned (feature index order)
    assert labels == feature_names


def test_group_difference_sort_true_orders_by_abs_diff():
    """With sort=True the feature with the largest absolute difference appears first."""
    rng = np.random.RandomState(0)
    n_samples, n_features = 60, 5
    shap_values = rng.randn(n_samples, n_features)
    # Make feature 3 dominate with a large group difference (+10 shift)
    group_mask = np.zeros(n_samples, dtype=bool)
    group_mask[:30] = True
    shap_values[group_mask, 3] += 10.0

    feature_names = [f"f{i}" for i in range(n_features)]
    _, ax = plt.subplots()
    shap.plots.group_difference(
        shap_values, group_mask, feature_names=feature_names, sort=True, show=False, ax=ax
    )
    labels = [t.get_text() for t in ax.get_yticklabels()]
    # Labels are returned in inds order: labels[0] = highest-ranked feature
    assert labels[0] == "f3"


def test_group_difference_custom_xlabel():
    """Custom xlabel is applied to the x-axis."""
    shap_values = _make_shap_values()
    group_mask = _make_group_mask()
    _, ax = plt.subplots()
    shap.plots.group_difference(
        shap_values, group_mask, xlabel="My custom label", show=False, ax=ax
    )
    assert ax.get_xlabel() == "My custom label"


def test_group_difference_default_xlabel():
    """Default xlabel is 'Group SHAP value difference'."""
    shap_values = _make_shap_values()
    group_mask = _make_group_mask()
    _, ax = plt.subplots()
    shap.plots.group_difference(shap_values, group_mask, show=False, ax=ax)
    assert ax.get_xlabel() == "Group SHAP value difference"


def test_group_difference_xlim():
    """xmin and xmax are applied to the axis limits."""
    shap_values = _make_shap_values()
    group_mask = _make_group_mask()
    _, ax = plt.subplots()
    shap.plots.group_difference(
        shap_values, group_mask, xmin=-5.0, xmax=5.0, show=False, ax=ax
    )
    assert ax.get_xlim() == (-5.0, 5.0)


def test_group_difference_1d_shap_values():
    """A 1-D shap_values vector (model output) is handled without error."""
    rng = np.random.RandomState(0)
    n_samples = 40
    shap_values_1d = rng.randn(n_samples)
    group_mask = rng.rand(n_samples) > 0.5
    _, ax = plt.subplots()
    # Should not raise
    shap.plots.group_difference(shap_values_1d, group_mask, show=False, ax=ax)
    # Single feature → single tick
    assert len(ax.get_yticklabels()) == 1


def test_group_difference_1d_default_feature_name():
    """A 1-D input with no feature_names gets an empty-string label."""
    rng = np.random.RandomState(1)
    n_samples = 40
    shap_values_1d = rng.randn(n_samples)
    group_mask = rng.rand(n_samples) > 0.5
    _, ax = plt.subplots()
    shap.plots.group_difference(shap_values_1d, group_mask, show=False, ax=ax)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == [""]


def test_group_difference_all_same_group():
    """When all samples are in one group the diff is zero for all features."""
    rng = np.random.RandomState(0)
    n_samples, n_features = 30, 4
    shap_values = rng.randn(n_samples, n_features)
    group_mask = np.ones(n_samples, dtype=bool)  # all True → ~group_mask is empty
    _, ax = plt.subplots()
    # Should not raise (mean of empty slice produces nan, bar chart handles it)
    shap.plots.group_difference(shap_values, group_mask, show=False, ax=ax)


def test_group_difference_boolean_group_mask():
    """Boolean group_mask produces the correct group difference values."""
    rng = np.random.RandomState(0)
    n_samples, n_features = 50, 4
    shap_values = rng.randn(n_samples, n_features)
    bool_mask = rng.rand(n_samples) > 0.5

    # Compute expected diff manually using boolean indexing
    expected_diff = shap_values[bool_mask].mean(0) - shap_values[~bool_mask].mean(0)

    _, ax = plt.subplots()
    shap.plots.group_difference(shap_values, bool_mask, sort=False, show=False, ax=ax)
    bars = [p.get_width() for p in ax.patches]

    np.testing.assert_allclose(bars, expected_diff, rtol=1e-6)


def test_group_difference_no_ax_creates_figure():
    """Without a provided ax the function creates its own figure."""
    shap_values = _make_shap_values()
    group_mask = _make_group_mask()
    before = plt.get_fignums()
    shap.plots.group_difference(shap_values, group_mask, show=False)
    after = plt.get_fignums()
    assert len(after) == len(before) + 1
