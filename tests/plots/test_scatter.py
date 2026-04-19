import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap
from shap.plots._scatter import dependence_legacy
from shap.plots._scatter import _plot_histogram
from shap.plots._scatter import _suggest_buffered_limits
from shap.plots._scatter import _suggest_x_jitter


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


def test_scatter_rejects_non_explanation_input():
    with pytest.raises(TypeError, match="The shap_values parameter must be a shap\\.Explanation object!"):
        shap.plots.scatter(np.array([1.0, 2.0, 3.0]), show=False)


def test_scatter_multi_column_with_ax_raises_value_error():
    explanation = shap.Explanation(
        values=np.array([[1.0, 2.0], [3.0, 4.0]]),
        data=np.array([[10.0, 20.0], [30.0, 40.0]]),
        feature_names=["feature_a", "feature_b"],
    )
    _, ax = plt.subplots()

    with pytest.raises(ValueError, match="The ax parameter is not supported when plotting multiple features"):
        shap.plots.scatter(explanation, ax=ax, show=False)


def test_scatter_single_column_show_false_returns_passed_axes():
    explanation = shap.Explanation(
        values=np.array([1.0, 2.0, 3.0]),
        data=np.array([10.0, 20.0, 30.0]),
        feature_names="feature_a",
    )
    _, ax = plt.subplots()

    returned_ax = shap.plots.scatter(explanation, ax=ax, show=False)

    assert returned_ax is ax


def test_scatter_sets_xticklabels_from_string_display_data():
    explanation = shap.Explanation(
        values=np.array([0.1, 0.2, 0.3]),
        data=np.array([0.0, 1.0, 2.0]),
        display_data=np.array(["low", "med", "high"]),
        feature_names="feature_a",
    )

    ax = shap.plots.scatter(explanation, show=False)

    assert sorted(tick.get_text() for tick in ax.get_xticklabels()) == sorted(["low", "med", "high"])


def test_suggest_buffered_limits_adds_five_percent_buffer():
    assert _suggest_buffered_limits(None, None, np.array([0.0, 10.0])) == (-0.5, 10.5)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (np.array([1, 1, 1]), 0.0),
        (np.array([1, 2, 1, 2]), 0),
        (np.tile(np.array([1, 2]), 25), 0.1),
        (np.tile(np.array([1, 2]), 250), 0.2),
    ],
)
def test_suggest_x_jitter_branches(values, expected):
    assert _suggest_x_jitter(values) == expected


def test_plot_histogram_uses_discrete_bins_and_updates_xlim():
    xv = np.tile(np.arange(5), 20)
    fig, ax = plt.subplots()
    axes_before = len(fig.axes)

    _plot_histogram(ax, xv, xv)

    assert len(fig.axes) == axes_before + 1
    assert ax.get_xlim() == (-0.5, 4.5)


def test_dependence_legacy_rejects_list_shap_values():
    with pytest.raises(TypeError, match="list not an array"):
        dependence_legacy(ind=0, shap_values=[np.array([0.1]), np.array([0.2])], features=np.array([[1.0], [2.0]]), show=False)


def test_dependence_legacy_sets_symmetric_ylim_when_only_ymax_given():
    _, ax = plt.subplots()

    dependence_legacy(
        ind=0,
        shap_values=np.array([[0.1], [0.2]]),
        features=np.array([[1.0], [2.0]]),
        ax=ax,
        ymin=None,
        ymax=2,
        show=False,
    )

    assert ax.get_ylim() == (-2.0, 2.0)


def test_suggest_x_jitter_handles_tiny_diffs():
    values = np.tile(np.array([1.0, 1.0 + 1e-10]), 250)

    assert _suggest_x_jitter(values) == 0.2


def test_scatter_uses_display_data_for_xticklabels():
    explanation = shap.Explanation(
        values=np.array([0.1, 0.2, 0.3]),
        data=np.array([0.0, 1.0, 2.0]),
        display_data=np.array(["low", "med", "high"]),
        feature_names="feature_a",
    )

    ax = shap.plots.scatter(explanation, show=False)

    assert sorted(ax.get_xticks().tolist()) == [0.0, 1.0, 2.0]
    assert sorted(tick.get_text() for tick in ax.get_xticklabels()) == sorted(["low", "med", "high"])


def test_scatter_auto_jitter_triggers_helper():
    values = np.tile(np.array([0.1, 0.2]), 250)
    data = np.tile(np.array([1.0, 2.0]), 250)
    explanation = shap.Explanation(values=values, data=data, feature_names="feature_a")

    ax = shap.plots.scatter(explanation, show=False)
    x_positions = ax.collections[0].get_offsets()[:, 0]

    assert isinstance(ax, plt.Axes)
    assert np.any(np.abs(x_positions - np.round(x_positions)) > 1e-12)


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
