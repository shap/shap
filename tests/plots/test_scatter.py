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
