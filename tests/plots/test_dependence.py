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


@pytest.mark.mpl_image_compare
def test_partial_dependence_custom_ax():
    """Ensure the ax parameter is respected — GH #3206."""
    sklearn = pytest.importorskip("sklearn")

    X, y = shap.datasets.california(n_points=10)
    X2 = shap.utils.sample(X, 2)

    model = sklearn.linear_model.LinearRegression()
    model.fit(X, y)

    fig, expected_ax = plt.subplots()
    _, returned_ax = shap.partial_dependence_plot(
        "MedInc",
        model.predict,
        X2,
        ice=False,
        ax=expected_ax,
        show=False,
    )
    assert returned_ax is expected_ax, "partial_dependence_plot must draw on the provided ax"
    return fig


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
