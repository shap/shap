import matplotlib
import shap
import pytest
import numpy as np
from .test_tree import _brute_force_tree_shap


def test_front_page_xgboost():
    xgboost = pytest.importorskip("xgboost")

    # load JS visualization code to notebook
    shap.initjs()

    # train XGBoost model
    X, y = shap.datasets.boston()
    model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

    # explain the model's predictions using SHAP values
    explainer = shap.GPUTreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # visualize the first prediction's explaination
    shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])

    # visualize the training set predictions
    shap.force_plot(explainer.expected_value, shap_values, X)

    # create a SHAP dependence plot to show the effect of a single feature across the whole dataset
    shap.dependence_plot(5, shap_values, X, show=False)
    shap.dependence_plot("RM", shap_values, X, show=False)

    # summarize the effects of all the features
    shap.summary_plot(shap_values, X, show=False)


def test_xgboost_direct():
    xgboost = pytest.importorskip("xgboost")
    N = 100
    M = 4
    X = np.random.randn(N, M)
    y = np.random.randn(N)

    model = xgboost.XGBRegressor()
    model.fit(X, y)

    explainer = shap.GPUTreeExplainer(model)
    shap_values = explainer.shap_values(X)

    assert np.allclose(shap_values[0, :], _brute_force_tree_shap(explainer.model, X[0, :]))
