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

    # Swap the input data from a "float-category"
    # To a string category in order to test scatter with
    # true non-float-castable category values
    X.loc[X["sex"] < 0, "sex"] = 0
    X.loc[X["sex"] > 0, "sex"] = 1
    X["sex"] = X["sex"].map({1: "Male", 0: "Female"}).astype("category")

    # train an XGBoost model (but any other model type would also work)
    model = xgboost.XGBRegressor(random_state=0, enable_categorical=True, max_cat_to_onehot=1, base_score=0.5)
    model.fit(X, y)
    # build an Exact explainer and explain the model predictions on the given dataset
    # We aren't providing masker directly because there appears
    # to be an error with string categories when using masker like this
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


@pytest.mark.mpl_image_compare
def test_scatter_no_histogram(explainer):
    """Test scatter plot without histogram."""
    explanation = explainer(explainer.data)
    shap.plots.scatter(explanation[:, "Age"], hist=False, show=False)
    plt.tight_layout()
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_scatter_with_title(explainer):
    """Test scatter plot with custom title."""
    explanation = explainer(explainer.data)
    shap.plots.scatter(explanation[:, "Age"], title="Custom Title", show=False)
    plt.tight_layout()
    return plt.gcf()


def test_scatter_show_true(explainer, monkeypatch):
    """Test scatter plot with show=True."""
    explanation = explainer(explainer.data)
    show_called = []
    monkeypatch.setattr(plt, "show", lambda: show_called.append(True))
    shap.plots.scatter(explanation[:, "Age"], show=True)
    assert len(show_called) == 1
    plt.close()


def test_scatter_multiple_features_with_ax_raises_error(explainer):
    """Test that passing ax with multiple features raises ValueError."""
    explanation = explainer(explainer.data)
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="The ax parameter is not supported when plotting multiple features"):
        shap.plots.scatter(explanation[:, ["Age", "Workclass"]], ax=ax, show=False)
    plt.close()


def test_scatter_invalid_type_raises_error():
    """Test that passing invalid type raises TypeError."""
    with pytest.raises(TypeError, match="The shap_values parameter must be a shap.Explanation object"):
        shap.plots.scatter([1, 2, 3], show=False)


def test_scatter_with_display_data(explainer):
    """Test scatter plot with display_data different from data."""
    explanation = explainer(explainer.data)
    age_explanation = explanation[:, "Age"]
    # Manually set display_data to be different
    age_explanation.display_data = age_explanation.data * 2  # Different display values
    shap.plots.scatter(age_explanation, show=False)
    plt.close()


def test_scatter_with_color_as_numpy_array(explainer):
    """Test scatter plot with color as numpy array."""
    explanation = explainer(explainer.data)
    age_explanation = explanation[:, "Age"]
    # Pass numpy array as color (will be wrapped as Explanation)
    color_values = np.random.randn(len(age_explanation))
    shap.plots.scatter(age_explanation, color=color_values, show=False)
    plt.close()


def test_scatter_with_multi_feature_color(explainer):
    """Test scatter plot with color as multi-feature Explanation."""
    explanation = explainer(explainer.data)
    age_explanation = explanation[:, "Age"]
    # Pass full explanation as color (will trigger approximate_interactions)
    shap.plots.scatter(age_explanation, color=explanation, show=False)
    plt.close()


def test_scatter_with_x_jitter_float(explainer):
    """Test scatter plot with explicit x_jitter float value."""
    explanation = explainer(explainer.data)
    age_explanation = explanation[:, "Age"]
    shap.plots.scatter(age_explanation, x_jitter=0.5, show=False)
    plt.close()


def test_scatter_with_x_jitter_above_1(explainer):
    """Test scatter plot with x_jitter > 1 (gets capped to 1)."""
    explanation = explainer(explainer.data)
    age_explanation = explanation[:, "Age"]
    shap.plots.scatter(age_explanation, x_jitter=1.5, show=False)
    plt.close()


def test_scatter_categorical_interaction_integer_range(explainer):
    """Test scatter plot with categorical interaction (integer values in small range)."""
    explanation = explainer(explainer.data)
    # Create explanation with small integer range for categorical detection
    rs = np.random.RandomState(42)
    color_data = rs.randint(0, 5, size=len(explanation))  # Small integer range
    color_explanation = shap.Explanation(values=color_data.astype(float), data=color_data)
    shap.plots.scatter(explanation[:, "Age"], color=color_explanation, show=False)
    plt.close()


def test_scatter_with_ylabel_parameter(explainer):
    """Test scatter plot with custom ylabel."""
    explanation = explainer(explainer.data)
    shap.plots.scatter(explanation[:, "Age"], ylabel="Custom SHAP", show=False)
    plt.close()


def test_scatter_with_dot_size_parameter(explainer):
    """Test scatter plot with custom dot_size."""
    explanation = explainer(explainer.data)
    shap.plots.scatter(explanation[:, "Age"], dot_size=30, show=False)
    plt.close()


def test_scatter_with_axis_color_parameter(explainer):
    """Test scatter plot with custom axis_color."""
    explanation = explainer(explainer.data)
    shap.plots.scatter(explanation[:, "Age"], axis_color="#FF0000", show=False)
    plt.close()


def test_scatter_with_custom_cmap(explainer):
    """Test scatter plot with custom colormap."""
    explanation = explainer(explainer.data)
    shap.plots.scatter(
        explanation[:, "Age"], color=explanation[:, "Workclass"], cmap=plt.get_cmap("viridis"), show=False
    )
    plt.close()
