import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap

# The following tests use shap.dependence_plot,
# which currently points to shap.plots._scatter.dependence_legacy


def test_random_dependence():
    """Make sure a dependence plot does not crash."""
    shap.dependence_plot(0, np.random.randn(20, 5), np.random.randn(20, 5), show=False)


def test_random_dependence_no_interaction():
    """Make sure a dependence plot does not crash when we are not showing interactions."""
    shap.dependence_plot(0, np.random.randn(20, 5), np.random.randn(20, 5), show=False, interaction_index=None)


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


def test_partial_dependence_custom_ax():
    """Passing a custom ax should plot on that axes, not create a new one.

    Regression test for GH #3206.
    """
    sklearn = pytest.importorskip("sklearn")

    X, y = shap.datasets.california(n_points=20)
    model = sklearn.linear_model.LinearRegression()
    model.fit(X, y)

    fig, axes = plt.subplots(1, 2)

    # Plot on the first axes
    shap.partial_dependence_plot(
        "MedInc",
        model.predict,
        X,
        ice=False,
        show=False,
        ax=axes[0],
    )

    # Plot on the second axes
    shap.partial_dependence_plot(
        "HouseAge",
        model.predict,
        X,
        ice=False,
        show=False,
        ax=axes[1],
    )

    # Verify that each axes received its own plot (has lines drawn on it)
    assert len(axes[0].lines) > 0, "First axes should have plot lines"
    assert len(axes[1].lines) > 0, "Second axes should have plot lines"

    # Verify that the axes labels are different (each feature plotted separately)
    assert axes[0].get_xlabel() == "MedInc"
    assert axes[1].get_xlabel() == "HouseAge"

    plt.close(fig)
