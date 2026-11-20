import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap


@pytest.mark.mpl_image_compare
def test_scatter_single(explainer):
    explanation = explainer(explainer.data)
    shap.plots.scatter(explanation[:, "Age"], show=False)
    plt.tight_layout()
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_scatter_interaction(explainer):
    explanation = explainer(explainer.data)
    shap.plots.scatter(explanation[:, "Age"], color=explanation[:, "Workclass"], show=False)
    plt.tight_layout()
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_scatter_dotchain(explainer):
    explanation = explainer(explainer.data)
    shap.plots.scatter(explanation[:, explanation.abs.mean(0).argsort[-2]], show=False)
    plt.tight_layout()
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_scatter_multiple_cols_overlay(explainer):
    explanation = explainer(explainer.data)
    shap_values = explanation[:, ["Age", "Workclass"]]
    overlay = {
        "foo": [
            ([20, 40, 70], [0, 1, 2]),
            ([1, 4, 6], [2, 1, 0]),
        ],
    }
    shap.plots.scatter(shap_values, overlay=overlay, show=False)
    plt.tight_layout()
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_scatter_custom(explainer):
    # Test with custom x/y limits, alpha and colormap
    explanation = explainer(explainer.data)
    age = explanation[:, "Age"]
    shap.plots.scatter(
        age,
        color=explanation[:, "Workclass"],
        xmin=age.percentile(20),
        xmax=age.percentile(80),
        ymin=age.percentile(10),
        ymax=age.percentile(90),
        alpha=0.5,
        cmap=plt.get_cmap("cool"),
        show=False,
    )
    plt.tight_layout()
    return plt.gcf()


@pytest.fixture()
def categorical_explanation():
    """Adopted from explainer in conftest.py but using a categorical input."""
    xgboost = pytest.importorskip("xgboost")
    # get a dataset on income prediction
    X, y = shap.datasets.diabetes()

    # Swap the input data from a "float-category" to categorical
    # Note: XGBoost with enable_categorical=True requires integer categories
    # when using pandas 3.0+, so we use integer categories to test categorical handling
    X.loc[X["sex"] < 0, "sex"] = 0
    X.loc[X["sex"] > 0, "sex"] = 1
    X["sex"] = X["sex"].astype(int).astype("category")

    # train an XGBoost model (but any other model type would also work)
    model = xgboost.XGBRegressor(random_state=0, enable_categorical=True, max_cat_to_onehot=1, base_score=0.5)
    model.fit(X, y)
    # build an Exact explainer and explain the model predictions on the given dataset
    # We aren't providing masker directly because there appears
    # to be an error with categorical features when using masker like this
    # TODO: Investigate the error when this line is `return shap.Explainer(model, X)``
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    return shap_values


@pytest.mark.mpl_image_compare(tolerance=3)
def test_scatter_categorical(categorical_explanation):
    """Test the scatter plot with categorical data. See GH #3135"""
    fig, ax = plt.subplots()
    shap.plots.scatter(categorical_explanation[:, "sex"], ax=ax, show=False)
    plt.tight_layout()
    return fig


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("input", [np.array([[1], [1]]), np.array([[1e-10], [1e-9]]), np.array([[1]])])
def test_scatter_plot_value_input(input):
    """Test scatter plot with different input values. See GH #4037"""
    explanations = shap.Explanation(
        input,
        data=input,
        feature_names=["feature1"],
    )

    shap.plots.scatter(explanations, show=False)
    plt.tight_layout()
    return plt.gcf()


# -----------------------------------------------------------------------
# API consistency tests
# -----------------------------------------------------------------------


def _make_explanation(n=50, seed=0):
    """Create a simple single-column Explanation for unit tests."""
    rs = np.random.RandomState(seed)
    values = rs.randn(n)
    data = rs.randn(n)
    return shap.Explanation(values=values, data=data, feature_names="feature1")


def _make_explanation_2col(n=50, seed=0):
    """Create a two-column Explanation for interaction tests."""
    rs = np.random.RandomState(seed)
    values = rs.randn(n, 2)
    data = rs.randn(n, 2)
    return shap.Explanation(values=values, data=data, feature_names=["feat1", "feat2"])


def test_show_false_returns_axes():
    """show=False must return a matplotlib Axes object."""
    plt.close("all")
    expln = _make_explanation()
    result = shap.plots.scatter(expln, show=False)
    assert isinstance(result, plt.Axes), f"Expected Axes, got {type(result)}"
    plt.close("all")


def test_show_true_returns_none():
    """show=True must return None (pyplot handles display)."""
    plt.close("all")
    expln = _make_explanation()
    # Patch plt.show to be a no-op so the test doesn't open a window
    original_show = plt.show
    plt.show = lambda: None
    try:
        result = shap.plots.scatter(expln, show=True)
    finally:
        plt.show = original_show
    assert result is None, f"Expected None when show=True, got {type(result)}"
    plt.close("all")


def test_ax_parameter_draws_on_given_axes():
    """When ax is provided the plot must be drawn on that exact Axes."""
    plt.close("all")
    fig, ax = plt.subplots()
    expln = _make_explanation()
    result = shap.plots.scatter(expln, ax=ax, show=False)
    assert result is ax, "scatter() must return the axes it was given"
    plt.close("all")


def test_subplot_isolation():
    """Drawing on one subplot must not modify sibling subplots."""
    plt.close("all")
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Capture state of ax2 before drawing on ax1
    ax2_title_before = ax2.get_title()
    ax2_xlabel_before = ax2.get_xlabel()
    ax2_ylabel_before = ax2.get_ylabel()

    expln = _make_explanation()
    shap.plots.scatter(expln, ax=ax1, show=False)

    assert ax2.get_title() == ax2_title_before, "sibling ax2 title was modified"
    assert ax2.get_xlabel() == ax2_xlabel_before, "sibling ax2 xlabel was modified"
    assert ax2.get_ylabel() == ax2_ylabel_before, "sibling ax2 ylabel was modified"
    plt.close("all")


def test_figure_size_unchanged_when_ax_provided():
    """When an ax is provided, the figure size must not be altered by scatter()."""
    plt.close("all")
    original_size = (4.0, 3.0)
    fig, ax = plt.subplots(figsize=original_size)
    expln = _make_explanation()
    shap.plots.scatter(expln, ax=ax, show=False)
    actual_size = fig.get_size_inches()
    assert tuple(actual_size) == pytest.approx(original_size, rel=1e-3), (
        f"Figure size changed from {original_size} to {tuple(actual_size)} when ax was provided"
    )
    plt.close("all")


def test_figure_size_set_when_ax_not_provided():
    """When no ax is provided, scatter() must create a figure with a reasonable default size."""
    plt.close("all")
    expln = _make_explanation()
    result = shap.plots.scatter(expln, show=False)
    fig_size = result.get_figure().get_size_inches()
    # Default single-feature size should be approximately (6, 5)
    assert fig_size[0] == pytest.approx(6.0, rel=0.1), f"Unexpected figure width: {fig_size[0]}"
    assert fig_size[1] == pytest.approx(5.0, rel=0.1), f"Unexpected figure height: {fig_size[1]}"
    plt.close("all")


def test_multiple_calls_on_same_ax():
    """Calling scatter() multiple times on the same Axes must not crash."""
    plt.close("all")
    fig, ax = plt.subplots()
    expln = _make_explanation()
    result1 = shap.plots.scatter(expln, ax=ax, show=False)
    result2 = shap.plots.scatter(expln, ax=ax, show=False)
    assert result1 is ax, "First call must return the provided ax"
    assert result2 is ax, "Second call must return the provided ax"
    plt.close("all")


def test_colorbar_compatibility():
    """scatter() with an interaction feature (which draws a colorbar) must not crash."""
    plt.close("all")
    expln2 = _make_explanation_2col()
    col_expln = shap.Explanation(
        values=expln2.values[:, 1],
        data=expln2.data[:, 1],
        feature_names="feat2",
    )
    result = shap.plots.scatter(
        shap.Explanation(
            values=expln2.values[:, 0],
            data=expln2.data[:, 0],
            feature_names="feat1",
        ),
        color=col_expln,
        show=False,
    )
    assert isinstance(result, plt.Axes), f"Expected Axes with colorbar, got {type(result)}"
    plt.close("all")


def test_colorbar_compatibility_with_explicit_ax():
    """Colorbar must attach to the provided ax without error or leaking to sibling axes."""
    plt.close("all")
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Record ax2 state before
    ax2_title_before = ax2.get_title()
    ax2_xlabel_before = ax2.get_xlabel()

    expln2 = _make_explanation_2col()
    col_expln = shap.Explanation(
        values=expln2.values[:, 1],
        data=expln2.data[:, 1],
        feature_names="feat2",
    )
    result = shap.plots.scatter(
        shap.Explanation(
            values=expln2.values[:, 0],
            data=expln2.data[:, 0],
            feature_names="feat1",
        ),
        color=col_expln,
        ax=ax1,
        show=False,
    )
    assert result is ax1, "scatter() must return the provided ax1"
    # Sibling ax2 must remain untouched by colorbar creation
    assert ax2.get_title() == ax2_title_before, "sibling ax2 title was modified by colorbar"
    assert ax2.get_xlabel() == ax2_xlabel_before, "sibling ax2 xlabel was modified by colorbar"
    plt.close("all")


def test_ax_parameter_raises_for_multiple_features():
    """Passing ax= when plotting multiple features must raise ValueError."""
    plt.close("all")
    expln = _make_explanation_2col()
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="ax parameter is not supported"):
        shap.plots.scatter(expln, ax=ax, show=False)
    plt.close("all")


def test_no_pyplot_state_leak_when_ax_provided():
    """scatter() must not change the pyplot current axes when ax is provided."""
    plt.close("all")
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    # Set ax2 as the current active axes
    plt.sca(ax2)
    current_before = plt.gca()

    expln = _make_explanation()
    shap.plots.scatter(expln, ax=ax1, show=False)

    current_after = plt.gca()
    assert current_after is current_before, "scatter() must not mutate pyplot current axes when ax= is provided"
    plt.close("all")
