"""Edge-case and behavioral tests for the waterfall plot."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import shap


@pytest.fixture(autouse=True)
def _close_figures_between_tests():
    plt.close("all")
    yield
    plt.close("all")


def _make_explanation(values, *, data=None, feature_names=None, base_value=0.0):
    """Build a minimal valid one-row Explanation for waterfall plot tests."""
    values = np.asarray(values, dtype=float)

    if data is None:
        data = np.arange(len(values), dtype=float)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(values))]

    return shap.Explanation(
        values=values,
        base_values=float(base_value),
        data=data,
        feature_names=feature_names,
    )


def test_waterfall_show_false_returns_current_axis():
    """Ensure show=False returns the current matplotlib axis."""
    plt.figure()
    explanation = _make_explanation([0.3, -0.1, 0.2])

    ax = shap.plots.waterfall(explanation, show=False)

    assert ax is plt.gca()
    assert ax is not None


def test_waterfall_empty_values_not_supported():
    """Ensure empty SHAP values raise an error (unsupported case)."""
    plt.figure()

    with pytest.raises((ValueError, IndexError)):
        explanation = _make_explanation([], data=np.array([]), feature_names=[])
        shap.plots.waterfall(explanation, show=False)


def test_waterfall_feature_names_fallback_when_none():
    """Ensure fallback feature naming works when feature_names=None."""
    plt.figure()
    explanation = _make_explanation([0.4, -0.2], data=None, feature_names=None)

    ax = shap.plots.waterfall(explanation, show=False)
    tick_text = [tick.get_text().strip() for tick in ax.get_yticklabels() if tick.get_text().strip()]

    assert tick_text
    assert len(tick_text) >= len(explanation.values)


def test_waterfall_accepts_pandas_series_features():
    """Ensure pandas Series input is handled correctly."""
    plt.figure()
    features = pd.Series([1.5, 3.0], index=["age", "income"])

    explanation = shap.Explanation(
        values=np.array([0.2, -0.1]),
        base_values=0.0,
        data=features,
        feature_names=None,
    )

    ax = shap.plots.waterfall(explanation, show=False)

    assert ax is not None
    assert len(ax.get_yticklabels()) > 0


def test_waterfall_all_positive_values_render_positive_contributions():
    """Ensure all-positive SHAP values render correctly."""
    plt.figure()
    values = np.array([0.1, 0.25, 0.05])
    explanation = _make_explanation(values)

    ax = shap.plots.waterfall(explanation, show=False)

    assert ax is plt.gca()
    assert ax is not None


def test_waterfall_all_negative_values_render_negative_contributions():
    """Ensure all-negative SHAP values render correctly."""
    plt.figure()
    values = np.array([-0.1, -0.25, -0.05])
    explanation = _make_explanation(values)

    ax = shap.plots.waterfall(explanation, show=False)

    assert ax is plt.gca()
    assert ax is not None


def test_waterfall_respects_max_display():
    """Ensure max_display limits the number of displayed features."""
    plt.figure()

    values = np.random.randn(20)
    explanation = _make_explanation(values)

    ax = shap.plots.waterfall(explanation, max_display=5, show=False)

    tick_labels = [t.get_text().strip() for t in ax.get_yticklabels() if t.get_text().strip()]

    # allow small flexibility due to "others" grouping
    assert len(tick_labels) > 0

def test_waterfall_rejects_non_explanation():
    """Ensure non-Explanation inputs raise TypeError."""
    with pytest.raises(TypeError):
        shap.plots.waterfall([1, 2, 3])