""" Unit tests for the Permutation explainer.
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
    explainer = shap.explainers.Permutation(model.predict, X)
    shap_values = explainer(X)

    assert np.max(np.abs(shap_values.base_values + shap_values.values.sum(1) - model.predict(X[:100])) < 1e6)

def test_single_class_independent_auto_api():
    xgboost = pytest.importorskip('xgboost')

    # get a dataset on income prediction
    X, y = shap.datasets.adult()
    X = X.iloc[:100]
    y = y[:100]

    # train an XGBoost model (but any other model type would also work)
    model = xgboost.XGBClassifier()
    model.fit(X, y)

    # build an Exact explainer and explain the model predictions on the given dataset
    explainer = shap.Explainer(model.predict, X, algorithm="permutation")
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
    explainer = shap.explainers.Permutation(model.predict_proba, X)
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
    explainer = shap.explainers.Permutation(model.predict, masker)
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
    explainer = shap.explainers.Permutation(model.predict_proba, masker)
    shap_values = explainer(X)

    assert np.max(np.abs(shap_values.base_values + shap_values.values.sum(1) - model.predict_proba(X[:100])) < 1e6)


def test_serialization_permutation_dataframe():
    import shap
    import xgboost
    import pickle
    import numpy as np

    # get a dataset on income prediction
    X,y = shap.datasets.adult()

    # train an XGBoost model (but any other model type would also work)
    model = xgboost.XGBClassifier()
    model.fit(X, y)

    explainer_original = shap.Explainer(model.predict_proba, X, algorithm='permutation')
    shap_values_original = explainer_original(X[:1])


    # Serialization 
    out_file = open(r'test_serialization_permutation_dataframe_scratch_file.bin', "wb")
    explainer_original.save(out_file)
    out_file.close()

    # Deserialization
    in_file = open(r'test_serialization_permutation_dataframe_scratch_file.bin', "rb")
    explainer_new = shap.Explainer.load(in_file)
    in_file.close()

    shap_values_new = explainer_new(X[:1])

    for i in range(len(explainer_original.masker.feature_names)):
        assert explainer_original.masker.feature_names[i] == explainer_new.masker.feature_names[i] 

    assert np.array_equal(shap_values_original.base_values,shap_values_new.base_values)
    assert type(explainer_original) == type(explainer_new)
    assert type(explainer_original.masker) == type(explainer_new.masker)


def test_serialization_permutation_numpy():
    import shap
    import xgboost
    import pickle
    import numpy as np

    # get a dataset on income prediction
    X,y = shap.datasets.adult()
    X = X.values

    # train an XGBoost model (but any other model type would also work)
    model = xgboost.XGBClassifier()
    model.fit(X, y)

    explainer_original = shap.Explainer(model.predict_proba, X, algorithm='permutation')
    shap_values_original = explainer_original(X[:1])


    # Serialization 
    out_file = open(r'test_serialization_permutation_dataframe_scratch_file.bin', "wb")
    explainer_original.save(out_file)
    out_file.close()

    # Deserialization
    in_file = open(r'test_serialization_permutation_dataframe_scratch_file.bin', "rb")
    explainer_new = shap.Explainer.load(in_file)
    in_file.close()

    shap_values_new = explainer_new(X[:1])

    assert (getattr(explainer_original.masker, "feature_names", None) == None) and (getattr(explainer_original.masker, "feature_names", None) == None)
    assert np.array_equal(shap_values_original.base_values,shap_values_new.base_values)
    assert type(explainer_original) == type(explainer_new)
    assert type(explainer_original.masker) == type(explainer_new.masker)


def test_serialization_permutation_numpy_custom_save():
    import shap
    import xgboost
    import pickle
    import numpy as np

    # get a dataset on income prediction
    X,y = shap.datasets.adult()
    X = X.values

    # train an XGBoost model (but any other model type would also work)
    model = xgboost.XGBClassifier()
    model.fit(X, y)

    explainer_original = shap.Explainer(model.predict_proba, X, algorithm='permutation')
    shap_values_original = explainer_original(X[:1])


    # Serialization 
    out_file = open(r'test_serialization_permutation_dataframe_scratch_file.bin', "wb")
    explainer_original.model.save = lambda model, out_file: pickle.dump(model, out_file)
    explainer_original.save(out_file)
    out_file.close()

    # Deserialization
    in_file = open(r'test_serialization_permutation_dataframe_scratch_file.bin', "rb")
    custom_explainer_loader = lambda in_file: pickle.load(in_file)
    explainer_new = shap.Explainer.load(in_file, custom_explainer_loader)
    in_file.close()

    shap_values_new = explainer_new(X[:1])

    assert (getattr(explainer_original.masker, "feature_names", None) == None) and (getattr(explainer_original.masker, "feature_names", None) == None)
    assert np.array_equal(shap_values_original.base_values,shap_values_new.base_values)
    assert type(explainer_original) == type(explainer_new)
    assert type(explainer_original.masker) == type(explainer_new.masker)