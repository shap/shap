""" This file contains tests for the Tabular maskers.
"""

import tempfile

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

import shap


def test_serialization_independent_masker_dataframe():
    """ Test the serialization of an Independent masker based on a data frame.
    """

    X, _ = shap.datasets.california(n_points=500)

    # initialize independent masker
    original_independent_masker = shap.maskers.Independent(X)

    with tempfile.TemporaryFile() as temp_serialization_file:

        # serialize independent masker
        original_independent_masker.save(temp_serialization_file)

        temp_serialization_file.seek(0)

        # deserialize masker
        new_independent_masker = shap.maskers.Independent.load(temp_serialization_file)

    mask = np.ones(X.shape[1]).astype(int)
    mask[0] = 0
    mask[4] = 0

    # comparing masked values
    assert np.array_equal(original_independent_masker(mask, X[:1].values[0])[1], new_independent_masker(mask, X[:1].values[0])[1])

def test_serialization_independent_masker_numpy():
    """ Test the serialization of an Independent masker based on a numpy array.
    """


    X, _ = shap.datasets.california(n_points=500)
    X = X.values

    # initialize independent masker
    original_independent_masker = shap.maskers.Independent(X)

    with tempfile.TemporaryFile() as temp_serialization_file:

        # serialize independent masker
        original_independent_masker.save(temp_serialization_file)


        temp_serialization_file.seek(0)

        # deserialize masker
        new_independent_masker = shap.maskers.Independent.load(temp_serialization_file)

    mask = np.ones(X.shape[1]).astype(int)
    mask[0] = 0
    mask[4] = 0

    # comparing masked values
    assert np.array_equal(original_independent_masker(mask, X[0])[0], new_independent_masker(mask, X[0])[0])

def test_serialization_partion_masker_dataframe():
    """ Test the serialization of a Partition masker based on a DataFrame.
    """

    X, _ = shap.datasets.california(n_points=500)

    # initialize partition masker
    original_partition_masker = shap.maskers.Partition(X)

    with tempfile.TemporaryFile() as temp_serialization_file:

        # serialize partition masker
        original_partition_masker.save(temp_serialization_file)

        temp_serialization_file.seek(0)

        # deserialize masker
        new_partition_masker = shap.maskers.Partition.load(temp_serialization_file)

    mask = np.ones(X.shape[1]).astype(int)
    mask[0] = 0
    mask[4] = 0

    # comparing masked values
    assert np.array_equal(original_partition_masker(mask, X[:1].values[0])[1], new_partition_masker(mask, X[:1].values[0])[1])

def test_serialization_partion_masker_numpy():
    """ Test the serialization of a Partition masker based on a numpy array.
    """

    X, _ = shap.datasets.california(n_points=500)
    X = X.values

    # initialize partition masker
    original_partition_masker = shap.maskers.Partition(X)

    with tempfile.TemporaryFile() as temp_serialization_file:

        # serialize partition masker
        original_partition_masker.save(temp_serialization_file)

        temp_serialization_file.seek(0)

        # deserialize masker
        new_partition_masker = shap.maskers.Partition.load(temp_serialization_file)

    mask = np.ones(X.shape[1]).astype(int)
    mask[0] = 0
    mask[4] = 0

    # comparing masked values
    assert np.array_equal(original_partition_masker(mask, X[0])[0], new_partition_masker(mask, X[0])[0])

def test_serialization_impute_masker_dataframe():
    """ Test the serialization of a Partition masker based on a DataFrame.
    """

    X, _ = shap.datasets.california(n_points=500)

    # initialize partition masker
    original_partition_masker = shap.maskers.Impute(X)

    with tempfile.TemporaryFile() as temp_serialization_file:

        # serialize partition masker
        original_partition_masker.save(temp_serialization_file)

        temp_serialization_file.seek(0)

        # deserialize masker
        new_partition_masker = shap.maskers.Impute.load(temp_serialization_file)

    mask = np.ones(X.shape[1]).astype(int)
    mask[0] = 0
    mask[4] = 0

    # comparing masked values
    assert np.array_equal(original_partition_masker(mask, X[:1].values[0])[1], new_partition_masker(mask, X[:1].values[0])[1])

def test_serialization_impute_masker_numpy():
    """ Test the serialization of a Partition masker based on a numpy array.
    """

    X, _ = shap.datasets.california(n_points=500)
    X = X.values

    # initialize partition masker
    original_partition_masker = shap.maskers.Impute(X)

    with tempfile.TemporaryFile() as temp_serialization_file:

        # serialize partition masker
        original_partition_masker.save(temp_serialization_file)

        temp_serialization_file.seek(0)

        # deserialize masker
        new_partition_masker = shap.maskers.Impute.load(temp_serialization_file)

    mask = np.ones(X.shape[1]).astype(int)
    mask[0] = 0
    mask[4] = 0

    # comparing masked values
    assert np.array_equal(original_partition_masker(mask, X[0])[0], new_partition_masker(mask, X[0])[0])

def test_imputation():
    # toy data
    x = np.full((5, 5), np.arange(1,6)).T

    methods = ["linear", "mean", "median", "most_frequent", "knn"]
    # toy background data
    bckg = np.full((5, 5), np.arange(1,6)).T
    for method in methods:
        # toy sample to impute
        x = np.arange(1, 6)
        masker = shap.maskers.Impute(np.full((1,5), 1), method=method)
        # only mask the second value
        mask = np.ones_like(bckg[0])
        mask[1] = 0
        # masker should impute the original value (toy data is predictable)
        imputed = masker(mask.astype(bool), x)
        assert np.all(x == imputed)

def test_imputation_workflow():
    # toy data
    X, y = make_regression(n_samples=100)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        train_size = 0.75)

    # train toy model
    model = MLPRegressor()
    model.fit(X_train, y_train)
    model.score(X_test, y_test)

    background = shap.maskers.Impute(X_train)
    # TypeError here prior to PR #3379
    explainer = shap.Explainer(model.predict, masker=background)

    shap_values = explainer(X_test)
    shap.Explanation(shap_values.values,
                           shap_values.base_values,
                           shap_values.data)
