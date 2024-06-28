import matplotlib.pyplot as plt
import pytest

import shap


@pytest.fixture()
def shap_values():
    """Adopted from explainer in conftest.py but using a categorical input."""
    xgboost = pytest.importorskip('xgboost')
    # get a dataset on income prediction
    X, y = shap.datasets.diabetes()

    # Swap the input data from a "float-category"
    # To a string category in order to test scatter with
    # true non-float-castable category values
    X.loc[X["sex"] < 0, "sex"] = 0
    X.loc[X["sex"] > 0, "sex"] = 1
    X["sex"] = X["sex"].map({
        1: "Male",
        0: "Female",
    }).astype("category")

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
def test_scatter(shap_values):
    """Test the scatter plot."""

    fig, ax = plt.subplots()
    shap.plots.scatter(shap_values[:, "sex"], ax=ax)
    plt.tight_layout()
    return fig
