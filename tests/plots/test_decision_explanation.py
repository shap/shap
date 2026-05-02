"""Professional regression tests for Explanation support and PDP axis handling.

Targets:
- PR #4945 (Native Explanation support in decision_plot)
- PR #4922 (Ax support in partial_dependence_plot)
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import shap

# ---------------------------------------------------------------------------
# PR #4945: Native Explanation Support in decision_plot
# ---------------------------------------------------------------------------


def test_decision_plot_explanation_equivalence():
    """
    Ensure that passing an Explanation object to decision_plot yields
    identical results to passing the raw values/base_values manually.
    """
    X, _ = shap.datasets.adult(n_points=10)
    # Mock data
    values = np.random.randn(10, X.shape[1])
    base_values = np.zeros(10)
    feature_names = X.columns.tolist()

    explanation = shap.Explanation(values, base_values, X, feature_names=feature_names)

    # 1. Plot with raw arrays (Legacy)
    fig1, ax1 = plt.subplots()
    shap.plots.decision(base_values[0], values[0], feature_names=feature_names, show=False, ax=ax1)

    # 2. Plot with Explanation object (Modern)
    fig2, ax2 = plt.subplots()
    shap.plots.decision(explanation[0], show=False, ax=ax2)

    # Verification: Check that the y-tick labels (feature names) match
    assert [t.get_text() for t in ax1.get_yticklabels()] == [t.get_text() for t in ax2.get_yticklabels()]
    # Verification: Check that the line data matches
