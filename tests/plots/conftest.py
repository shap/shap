"""Shared pytest fixtures"""

import matplotlib.pyplot as plt
import pytest

import shap


@pytest.fixture(autouse=True)
def close_matplotlib_plots_after_tests():
    plt.close("all")


@pytest.fixture()
def explainer():
    """A simple explainer to be used as a test fixture."""
    xgboost = pytest.importorskip("xgboost")
    # get a dataset on income prediction
    X, y = shap.datasets.adult()
    X = X.iloc[:100]
    y = y[:100]

    # train an XGBoost model (but any other model type would also work)
    model = xgboost.XGBClassifier(random_state=0, tree_method="exact", base_score=0.5).fit(X, y)

    # build an Exact explainer and explain the model predictions on the given dataset
    return shap.TreeExplainer(model, X)
