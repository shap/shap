""" Unit tests for the Exact explainer.
"""

# pylint: disable=missing-function-docstring
import numpy as np
import pytest
import shap


def test_single_class_independent():
    xgboost = pytest.importorskip('xgboost')

    # get a dataset on income prediction
    X, y = shap.datasets.adult()
    X = X.iloc[:100]
    y = y[:100]

    # train an XGBoost model (but any other model type would also work)
    model = xgboost.XGBClassifier()
    model.fit(X, y)

    # build an Exact explainer and explain the model predictions on the given dataset
    explainer = shap.explainers.Exact(model.predict, X)
    shap_values = explainer(X)

    assert np.max(np.abs(shap_values.base_values + shap_values.values.sum(1) - model.predict(X[:100])) < 1e6)


def test_multi_class_independent():
    xgboost = pytest.importorskip('xgboost')
    # get a dataset on income prediction
    X, y = shap.datasets.adult()
    X = X.iloc[:100]
    y = y[:100]

    # train an XGBoost model (but any other model type would also work)
    model = xgboost.XGBClassifier()
    model.fit(X, y)

    # build an Exact explainer and explain the model predictions on the given dataset
    explainer = shap.explainers.Exact(model.predict_proba, X)
    shap_values = explainer(X)

    assert np.max(np.abs(shap_values.base_values + shap_values.values.sum(1) - model.predict_proba(X[:100])) < 1e6)


def test_single_class_partition():
    xgboost = pytest.importorskip('xgboost')

    # get a dataset on income prediction
    X, y = shap.datasets.adult()
    X = X.iloc[:100]
    y = y[:100]

    # train an XGBoost model (but any other model type would also work)
    model = xgboost.XGBClassifier()
    model.fit(X, y)

    # build an Exact explainer and explain the model predictions on the given dataset
    masker = shap.maskers.Partition(X)
    explainer = shap.explainers.Exact(model.predict, masker)
    shap_values = explainer(X)

    assert np.max(np.abs(shap_values.base_values + shap_values.values.sum(1) - model.predict(X[:100])) < 1e6)

def test_multi_class_partition():
    xgboost = pytest.importorskip('xgboost')

    # get a dataset on income prediction
    X, y = shap.datasets.adult()
    X = X.iloc[:100]
    y = y[:100]

    # train an XGBoost model (but any other model type would also work)
    model = xgboost.XGBClassifier()
    model.fit(X, y)

    # build an Exact explainer and explain the model predictions on the given dataset
    masker = shap.maskers.Partition(X)
    explainer = shap.explainers.Exact(model.predict_proba, masker)
    shap_values = explainer(X)

    assert np.max(np.abs(shap_values.base_values + shap_values.values.sum(1) - model.predict_proba(X[:100])) < 1e6)
