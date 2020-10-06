''' This file contains tests for the bar plot.
'''
import matplotlib
matplotlib.use('Agg')


def test_simple_bar():
    import shap
    import xgboost
    import numpy as np

    # get a dataset on income prediction
    X,y = shap.datasets.adult()
    X = X.iloc[:100]
    y = y[:100]
    
    # train an XGBoost model (but any other model type would also work)
    model = xgboost.XGBClassifier()
    model.fit(X, y)

    # build an Exact explainer and explain the model predictions on the given dataset
    explainer = shap.explainers.Permutation(model.predict, X)
    shap_values = explainer(X)

    shap.plots.bar(shap_values, show=False)
