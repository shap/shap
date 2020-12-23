""" This file contains tests for the Tabular maskers.
"""

import numpy as np
import shap


def test_serialization_independent_masker_dataframe():
    """ Test the serialization of an Independent masker based on a data frame.
    """

    X, _ = shap.datasets.boston()

    # initialize independent masker
    original_independent_masker = shap.maskers.Independent(X)

    # serialize independent masker
    out_file = open(r'test_serialization_independent_masker.bin', "wb")
    original_independent_masker.save(out_file)
    out_file.close()

    # deserialize masker
    in_file = open(r'test_serialization_independent_masker.bin', "rb")
    new_independent_masker = shap.maskers.Independent.load(in_file)
    in_file.close()

    mask = np.ones(X.shape[1]).astype(np.int)
    mask[0] = 0
    mask[4] = 0

    # comparing masked values
    assert np.array_equal(original_independent_masker(mask, X[:1].values[0])[1], new_independent_masker(mask, X[:1].values[0])[1])

def test_serialization_independent_masker_numpy():
    """ Test the serialization of an Independent masker based on a numpy array.
    """

    X, _ = shap.datasets.boston()
    X = X.values

    # initialize independent masker
    original_independent_masker = shap.maskers.Independent(X)

    # serialize independent masker
    out_file = open(r'test_serialization_independent_masker.bin', "wb")
    original_independent_masker.save(out_file)
    out_file.close()

    # deserialize masker
    in_file = open(r'test_serialization_independent_masker.bin', "rb")
    new_independent_masker = shap.maskers.Masker.load(in_file)
    in_file.close()

    mask = np.ones(X.shape[1]).astype(np.int)
    mask[0] = 0
    mask[4] = 0

    # comparing masked values
    assert np.array_equal(original_independent_masker(mask, X[0])[0], new_independent_masker(mask, X[0])[0])

def test_serialization_partion_masker_dataframe():
    """ Test the serialization of a Partition masker based on a DataFrame.
    """

    X, _ = shap.datasets.boston()

    # initialize partition masker
    original_partition_masker = shap.maskers.Partition(X)

    # serialize partition masker
    out_file = open(r'test_serialization_partition_masker.bin', "wb")
    original_partition_masker.save(out_file)
    out_file.close()


    # deserialize masker
    in_file = open(r'test_serialization_partition_masker.bin', "rb")
    new_partition_masker = shap.maskers.Partition.load(in_file)
    in_file.close()

    mask = np.ones(X.shape[1]).astype(np.int)
    mask[0] = 0
    mask[4] = 0

    # comparing masked values
    assert np.array_equal(original_partition_masker(mask, X[:1].values[0])[1], new_partition_masker(mask, X[:1].values[0])[1])

def test_serialization_partion_masker_numpy():
    """ Test the serialization of a Partition masker based on a numpy array.
    """

    X, _ = shap.datasets.boston()
    X = X.values

    # initialize partition masker
    original_partition_masker = shap.maskers.Partition(X)

    # serialize partition masker
    out_file = open(r'test_serialization_partition_masker.bin', "wb")
    original_partition_masker.save(out_file)
    out_file.close()


    # deserialize masker
    in_file = open(r'test_serialization_partition_masker.bin', "rb")
    new_partition_masker = shap.maskers.Masker.load(in_file)
    in_file.close()

    mask = np.ones(X.shape[1]).astype(np.int)
    mask[0] = 0
    mask[4] = 0

    # comparing masked values
    assert np.array_equal(original_partition_masker(mask, X[0])[0], new_partition_masker(mask, X[0])[0])
