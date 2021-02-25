import numpy as np
import pytest
import shap


@pytest.fixture()
def explainer():
    """ A simple explainer to be used as a test fixture.
    """
    xgboost = pytest.importorskip('xgboost')
    np.random.seed(0)
    # get a dataset on income prediction
    X, y = shap.datasets.adult()
    X = X.iloc[:100]
    y = y[:100]

    # train an XGBoost model (but any other model type would also work)
    model = xgboost.XGBClassifier().fit(X, y)

    # build an Exact explainer and explain the model predictions on the given dataset
    return shap.TreeExplainer(model, X)
