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


def _rank_fixture():
    """Model and Explanation where feature "c" ranks first and "a" second by mean |SHAP|."""
    X = np.random.default_rng(0).normal(size=(50, 3))
    values = X * np.array([1.0, 0.01, 5.0])

    def model(features):
        return features[:, 2] * 2.0

    return X, shap.Explanation(values=values, data=X, base_values=0.0, feature_names=["a", "b", "c"]), model


def test_partial_dependence_rank_based_indexing():
    """rank(N) selectors resolve against the Explanation supplied as data."""
    # GH #2920
    _, exp, model = _rank_fixture()

    fig, ax = shap.plots.partial_dependence("rank(0)", model, exp, feature_names=["a", "b", "c"], ice=False, show=False)
    assert ax.get_xlabel() == "c"
    plt.close(fig)

    fig, ax = shap.plots.partial_dependence("rank(1)", model, exp, feature_names=["a", "b", "c"], ice=False, show=False)
    assert ax.get_xlabel() == "a"
    plt.close(fig)


def test_partial_dependence_rank_based_indexing_with_explicit_shap_values():
    """rank(N) selectors resolve against explicitly supplied SHAP values."""
    # GH #2920
    X, exp, model = _rank_fixture()

    fig, ax = shap.plots.partial_dependence(
        "rank(1)", model, X, shap_values=exp, feature_names=["a", "b", "c"], ice=False, show=False
    )
    assert ax.get_xlabel() == "a"
    plt.close(fig)


def test_partial_dependence_two_feature_rank_based_indexing():
    """rank(N) selectors resolve in the two-feature (3D) plot."""
    # GH #2920
    _, exp, model = _rank_fixture()

    fig, ax = shap.plots.partial_dependence(
        ("rank(0)", "rank(1)"), model, exp, feature_names=["a", "b", "c"], show=False
    )
    assert ax.get_xlabel() == "c"
    assert ax.get_ylabel() == "a"
    plt.close(fig)


def test_partial_dependence_rank_without_shap_values_raises():
    """Without any SHAP values, rank-based indexing keeps its informative error."""
    # GH #2920
    X, _, model = _rank_fixture()

    with pytest.raises(ValueError, match="shap_values must be provided for rank-based indexing"):
        shap.plots.partial_dependence("rank(0)", model, X, ice=False, show=False)


def test_partial_dependence_ordinary_selectors():
    """Integer and name selectors keep working alongside rank-based indexing."""
    # GH #2920
    X, exp, model = _rank_fixture()

    fig, ax = shap.plots.partial_dependence("c", model, exp, feature_names=["a", "b", "c"], ice=False, show=False)
    assert ax.get_xlabel() == "c"
    plt.close(fig)

    fig, ax = shap.plots.partial_dependence(0, model, X, ice=False, show=False)
    assert ax.get_xlabel() == "Feature 0"
    plt.close(fig)
