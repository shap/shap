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


@pytest.mark.parametrize("interaction_index", ["auto", None])
@pytest.mark.parametrize(("ind", "n_outputs"), [(0, 3), ("rank(0)", 2)])
def test_dependence_multioutput_raises(ind, n_outputs, interaction_index):
    """Make sure multi-output SHAP values give a clear error instead of crashing in approximate_interactions."""
    # GH 5110
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match=r"shap_values\[:, :, 0\]"):
        shap.dependence_plot(
            ind,
            rng.standard_normal((20, 5, n_outputs)),
            rng.standard_normal((20, 5)),
            show=False,
            interaction_index=interaction_index,
        )


def test_dependence_interaction_values():
    """Make sure 3D interaction values still plot correctly with a pair of indices."""
    rng = np.random.default_rng(0)
    interaction_values = rng.standard_normal((20, 5, 5))
    _, ax = plt.subplots()
    shap.dependence_plot((0, 1), interaction_values, rng.standard_normal((20, 5)), ax=ax, show=False)
    plotted = np.asarray(ax.collections[0].get_offsets())[:, 1]
    np.testing.assert_allclose(np.sort(plotted), np.sort(2 * interaction_values[:, 1, 0]))


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
