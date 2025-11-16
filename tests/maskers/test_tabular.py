"""This file contains tests for the Tabular maskers."""

import tempfile

import numpy as np

import shap


def test_serialization_independent_masker_dataframe():
    """Test the serialization of an Independent masker based on a data frame."""
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
    assert np.array_equal(
        original_independent_masker(mask, X[:1].values[0])[1], new_independent_masker(mask, X[:1].values[0])[1]
    )


def test_serialization_independent_masker_numpy():
    """Test the serialization of an Independent masker based on a numpy array."""
    X, _ = shap.datasets.california(n_points=500)
    X = X.values

    # initialize independent masker
    original_independent_masker = shap.maskers.Independent(X)

    with tempfile.TemporaryFile() as temp_serialization_file:
        # serialize independent masker
        original_independent_masker.save(temp_serialization_file)

        temp_serialization_file.seek(0)

        # deserialize masker
        new_independent_masker = shap.maskers.Masker.load(temp_serialization_file)

    mask = np.ones(X.shape[1]).astype(int)
    mask[0] = 0
    mask[4] = 0

    # comparing masked values
    assert np.array_equal(original_independent_masker(mask, X[0])[0], new_independent_masker(mask, X[0])[0])


def test_serialization_partion_masker_dataframe():
    """Test the serialization of a Partition masker based on a DataFrame."""
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
    assert np.array_equal(
        original_partition_masker(mask, X[:1].values[0])[1], new_partition_masker(mask, X[:1].values[0])[1]
    )


def test_serialization_partion_masker_numpy():
    """Test the serialization of a Partition masker based on a numpy array."""
    X, _ = shap.datasets.california(n_points=500)
    X = X.values

    # initialize partition masker
    original_partition_masker = shap.maskers.Partition(X)

    with tempfile.TemporaryFile() as temp_serialization_file:
        # serialize partition masker
        original_partition_masker.save(temp_serialization_file)

        temp_serialization_file.seek(0)

        # deserialize masker
        new_partition_masker = shap.maskers.Masker.load(temp_serialization_file)

    mask = np.ones(X.shape[1]).astype(int)
    mask[0] = 0
    mask[4] = 0

    # comparing masked values
    assert np.array_equal(original_partition_masker(mask, X[0])[0], new_partition_masker(mask, X[0])[0])


def test_independent_masker_with_dataframe_output():
    """Test that Independent masker returns DataFrame when initialized with DataFrame."""
    import pandas as pd

    df = pd.DataFrame(
        {"a": [0.0, 1.0, 2.0, 3.0, 4.0], "b": [5.0, 6.0, 7.0, 8.0, 9.0], "c": [10.0, 11.0, 12.0, 13.0, 14.0]}
    )

    masker = shap.maskers.Independent(df)

    # Test masking - should return DataFrame
    result = masker(np.array([True, False, True]), np.array([20.0, 21.0, 22.0]))
    assert isinstance(result[0], pd.DataFrame)
    assert list(result[0].columns) == ["a", "b", "c"]


def test_independent_masker_with_dict_mean_cov():
    """Test Independent masker with dictionary containing mean and cov."""
    data_dict = {"mean": np.array([1.5, 2.5, 3.5]), "cov": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])}

    masker = shap.maskers.Independent(data_dict)

    # Should have mean and cov attributes
    assert hasattr(masker, "mean")
    assert hasattr(masker, "cov")
    assert np.allclose(masker.mean, data_dict["mean"])
    assert np.allclose(masker.cov, data_dict["cov"])

    # Should work for masking
    result = masker(True, np.array([5, 6, 7]))
    assert isinstance(result, tuple)


def test_independent_masker_with_sampling():
    """Test Independent masker with large dataset that triggers sampling."""
    # Create data larger than max_samples
    large_data = np.random.randn(150, 5)

    masker = shap.maskers.Independent(large_data, max_samples=50)

    # Should have sampled down to max_samples
    assert masker.data.shape[0] == 50
    assert masker.data.shape[1] == 5


def test_independent_masker_with_clustering_array():
    """Test Independent masker with clustering as numpy array."""
    data = np.random.randn(10, 3)

    # Provide clustering as array - simplified hierarchical clustering
    # Format: [sample1, sample2, distance, new_cluster_size]
    clustering = np.array([[0, 1, 0.5, 2], [2, 3, 1.0, 2]])

    masker = shap.maskers.Independent(data, clustering=clustering)

    # Should use provided clustering
    assert masker.clustering is not None
    assert np.array_equal(masker.clustering, clustering)


def test_independent_masker_dimension_error():
    """Test that dimension mismatch raises appropriate error."""
    import pytest

    data = np.array([[0, 0, 0], [1, 1, 1]])

    masker = shap.maskers.Independent(data)

    # Try to mask with wrong dimension
    with pytest.raises(shap.utils._exceptions.DimensionError):
        masker(np.array([True, False]), np.array([1, 2]))  # Only 2 features instead of 3


def test_independent_masker_invariants_dimension_error():
    """Test invariants method with wrong input shape."""
    import pytest

    data = np.array([[0, 0, 0], [1, 1, 1]])

    masker = shap.maskers.Independent(data)

    # Call invariants with wrong shape
    with pytest.raises(shap.utils._exceptions.DimensionError):
        masker.invariants(np.array([1, 2]))  # Only 2 features instead of 3


def test_independent_masker_invariants():
    """Test invariants method for detecting unchanging features."""
    data = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])

    masker = shap.maskers.Independent(data)

    # Test with matching data
    x = np.array([0, 1, 2])
    invariants = masker.invariants(x)

    # All features should be invariant (match all background samples)
    assert invariants.shape == (3, 3)
    assert np.all(invariants)


def test_partition_masker_with_dataframe_output():
    """Test Partition masker returns DataFrame when initialized with DataFrame."""
    import pandas as pd

    df = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [3.0, 4.0, 5.0], "z": [6.0, 7.0, 8.0]})

    # Use clustering=None to avoid clustering issues with small DataFrame
    masker = shap.maskers.Partition(df, clustering=None)

    # Test masking - should return DataFrame
    result = masker(True, np.array([10.0, 11.0, 12.0]))
    assert isinstance(result[0], pd.DataFrame)


def test_independent_masker_no_clustering_or_partition():
    """Test Independent masker without clustering or partition."""
    data = np.random.randn(10, 3)

    masker = shap.maskers.Independent(data)

    # Both should be None
    assert masker.clustering is None
    assert masker.partition is None


def test_independent_masker_with_small_data():
    """Test Independent masker with data smaller than max_samples."""
    # Create small data that doesn't trigger sampling
    small_data = np.random.randn(5, 3)

    masker = shap.maskers.Independent(small_data, max_samples=100)

    # Should keep all data
    assert masker.data.shape[0] == 5
    assert masker.data.shape[1] == 3
