"""Shared pytest fixtures"""

from contextlib import ExitStack

import matplotlib.pyplot as plt
import pytest

import shap


@pytest.fixture(autouse=True)
def mpl_test_cleanup():
    """Run tests in a context manager and close figures after each test."""
    # Adapted from matplotlib test suite in cartopy
    with ExitStack() as stack:
        # At exit, close all open figures and switch backend back to original.
        stack.callback(plt.switch_backend, plt.get_backend())
        stack.callback(plt.close, "all")

        # Run each test in a context manager so that state does not leak out
        plt.switch_backend("Agg")
        stack.enter_context(plt.rc_context())
        yield


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
