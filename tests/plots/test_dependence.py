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


@pytest.fixture
def partial_dependence_explanation():
    data = np.array([[0, 10, 100], [2, 11, 105], [4, 12, 110]], dtype=float)
    values = (data - data.mean(0)) * np.array([3, 1, 2])
    return shap.Explanation(values, base_values=227.0, data=data)


@pytest.mark.parametrize("explanation_as_data", [False, True])
@pytest.mark.parametrize(
    "ind, feature_name, slope, intercept",
    [("rank(0)", "C", 2, 17), ("rank(1)", "A", 3, 221), (2, "C", 2, 17), ("C", "C", 2, 17)],
)
def test_partial_dependence_feature_selection(
    partial_dependence_explanation, explanation_as_data, ind, feature_name, slope, intercept
):
    explanation = partial_dependence_explanation
    fig, ax = shap.plots.partial_dependence(
        ind,
        lambda data: data @ np.array([3, 1, 2]),
        explanation if explanation_as_data else explanation.data,
        shap_values=None if explanation_as_data else explanation,
        feature_names=["A", "B", "C"],
        xmin=-1,
        xmax=1,
        npoints=3,
        ice=False,
        hist=False,
        show=False,
    )

    assert ax.get_xlabel() == feature_name
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [-1, 0, 1])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), slope * np.array([-1, 0, 1]) + intercept)
    fig.canvas.draw()


@pytest.mark.parametrize("explanation_as_data", [False, True])
def test_partial_dependence_rank_pair(partial_dependence_explanation, explanation_as_data):
    explanation = partial_dependence_explanation
    fig, ax = shap.plots.partial_dependence(
        ("rank(0)", "rank(1)"),
        lambda data: data @ np.array([3, 1, 2]),
        explanation if explanation_as_data else explanation.data,
        shap_values=None if explanation_as_data else explanation,
        feature_names=["A", "B", "C"],
        npoints=3,
        show=False,
    )

    assert ax.get_xlabel() == "C"
    assert ax.get_ylabel() == "A"
    fig.canvas.draw()


@pytest.mark.parametrize("ind", [(2, 0), ("C", "A")])
def test_partial_dependence_pair_with_shap_array(partial_dependence_explanation, ind):
    explanation = partial_dependence_explanation
    fig, ax = shap.plots.partial_dependence(
        ind,
        lambda data: data @ np.array([3, 1, 2]),
        explanation.data,
        shap_values=explanation.values,
        feature_names=["A", "B", "C"],
        npoints=3,
        show=False,
    )

    assert ax.get_xlabel() == "C"
    assert ax.get_ylabel() == "A"
    fig.canvas.draw()


@pytest.mark.parametrize("ind", ["rank(0)", ("rank(0)", 1), (0, "rank(1)")])
def test_partial_dependence_rank_requires_shap_values(partial_dependence_explanation, ind):
    with pytest.raises(ValueError, match="shap_values must be provided for rank-based indexing"):
        shap.plots.partial_dependence(
            ind,
            lambda data: data @ np.array([3, 1, 2]),
            partial_dependence_explanation.data,
            show=False,
        )
