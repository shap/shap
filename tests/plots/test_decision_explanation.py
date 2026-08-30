"""Regression tests for native Explanation support in decision_plot.

Target: PR #4945 (Native Explanation support in decision_plot)
"""

import matplotlib.pyplot as plt
import numpy as np

import shap

# ---------------------------------------------------------------------------
# PR #4945: Native Explanation Support in decision_plot
# ---------------------------------------------------------------------------


def test_decision_plot_accepts_explanation():
    """
    Verify that decision_plot can accept a shap.Explanation object
    directly instead of requiring raw arrays.
    """
    X, _ = shap.datasets.adult(n_points=10)
    values = np.random.randn(10, X.shape[1])
    base_values = np.zeros(10)
    feature_names = X.columns.tolist()

    explanation = shap.Explanation(values, base_values, X, feature_names=feature_names)

    # Should not raise any errors when passing an Explanation object
    shap.plots.decision(explanation, show=False)
    plt.close("all")


def test_decision_plot_explanation_equivalence():
    """
    Ensure that passing an Explanation object to decision_plot yields
    the same feature ordering as passing raw values manually.
    """
    np.random.seed(42)
    X, _ = shap.datasets.adult(n_points=10)
    values = np.random.randn(10, X.shape[1])
    base_values = np.full(10, 0.5)
    feature_names = X.columns.tolist()

    explanation = shap.Explanation(values, base_values, X, feature_names=feature_names)

    # 1. Plot with raw arrays (Legacy path)
    r_legacy = shap.plots.decision(
        base_values.mean(),
        values,
        feature_names=feature_names,
        show=False,
        return_objects=True,
    )

    # 2. Plot with Explanation object (Modern path)
    r_modern = shap.plots.decision(
        explanation,
        show=False,
        return_objects=True,
    )

    # The feature ordering and base values should match
    assert r_legacy is not None
    assert r_modern is not None
    np.testing.assert_array_equal(r_legacy.feature_idx, r_modern.feature_idx)
    np.testing.assert_almost_equal(r_legacy.base_value, r_modern.base_value)
    plt.close("all")


def test_decision_plot_legacy_still_works():
    """
    Confirm that the legacy calling convention (raw arrays) is not broken
    by the new Explanation support.
    """
    X, _ = shap.datasets.adult(n_points=5)
    values = np.random.randn(5, X.shape[1])
    base_value = 0.0
    feature_names = X.columns.tolist()

    # Legacy call should still work without errors
    shap.plots.decision(base_value, values, feature_names=feature_names, show=False)
    plt.close("all")


def test_multioutput_decision_explanation():
    """
    Verify that multioutput_decision correctly handles a multi-output
    Explanation object.
    """
    X, _ = shap.datasets.adult(n_points=10)
    n_outputs = 3
    values = np.random.randn(10, X.shape[1], n_outputs)
    base_values = np.zeros((10, n_outputs))
    feature_names = X.columns.tolist()

    explanation = shap.Explanation(values, base_values, X, feature_names=feature_names)

    # Should handle multi-output explanation for a specific row
    shap.multioutput_decision_plot(explanation, row_index=0, show=False)
    plt.close("all")
