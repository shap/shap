"""This file contains tests for the Tabular maskers."""

import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import shap
from shap.maskers._tabular import Tabular
from shap.utils import MaskedModel
from shap.utils._exceptions import InvalidClusteringError


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


def test_independent_masker_with_dataframe_init():
    """Test that Independent masker can be initialized with DataFrame."""
    import pandas as pd

    df = pd.DataFrame(
        {"a": [0.0, 1.0, 2.0, 3.0, 4.0], "b": [5.0, 6.0, 7.0, 8.0, 9.0], "c": [10.0, 11.0, 12.0, 13.0, 14.0]}
    )

    masker = shap.maskers.Independent(df)

    # Should preserve feature names
    assert hasattr(masker, "feature_names")
    assert list(masker.feature_names) == ["a", "b", "c"]
    assert masker.shape == (5, 3)


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


def test_independent_masker_with_no_clustering():
    """Test Independent masker without clustering."""
    data = np.random.randn(10, 3)

    # Independent masker doesn't take clustering parameter
    masker = shap.maskers.Independent(data)

    # Should have None clustering since Independent doesn't use it
    assert masker.clustering is None


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


def test_partition_masker_with_dataframe_init():
    """Test Partition masker can be initialized with DataFrame."""
    import pandas as pd

    df = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [3.0, 4.0, 5.0], "z": [6.0, 7.0, 8.0]})

    # Use clustering=None to avoid clustering issues with small DataFrame
    masker = shap.maskers.Partition(df, clustering=None)

    # Should preserve feature names and shape
    assert hasattr(masker, "feature_names")
    assert masker.shape == (3, 3)


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


@pytest.mark.parametrize("bad_clustering", [1.0, [0, 1], {"metric": "correlation"}])
def test_tabular_rejects_non_string_non_ndarray_clustering(bad_clustering):
    """Invalid clustering types raise InvalidClusteringError (targets _tabular.py:78)."""
    data = np.array([[0.0, 1.0], [2.0, 3.0]])
    with pytest.raises(InvalidClusteringError):
        Tabular(data, clustering=bad_clustering)


def test_tabular_rejects_clustering_and_partition_together():
    """Passing both clustering and partition raises ValueError (targets _tabular.py:71)."""
    data = np.array([[0.0, 1.0], [2.0, 3.0]])
    partition = np.array([[0, 1]], dtype=np.int64)
    with pytest.raises(ValueError, match="both 'clustering' and 'partition'"):
        Tabular(data, clustering="correlation", partition=partition)


def test_tabular_partition_only_sets_partition_and_clears_clustering():
    """Explicit partition without clustering stores partition and clears clustering (targets _tabular.py:83-84)."""
    data = np.array([[0.0, 1.0], [2.0, 3.0]])
    partition = np.array([[0, 1]], dtype=np.int64)
    masker = Tabular(data, partition=partition)
    assert masker.partition is partition
    assert masker.clustering is None


def test_independent_dataframe_boolean_mask_returns_dataframe():
    """Boolean masking with DataFrame-initialized masker returns a DataFrame (targets _tabular.py:138)."""
    df = pd.DataFrame({"a": [0.0, 1.0], "b": [2.0, 3.0]})
    masker = shap.maskers.Independent(df)
    x = np.array([10.0, 20.0])
    mask = np.array([True, False])
    out = masker(mask, x)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["a", "b"]
    # True keeps x; False uses background — first background row has b == 2.0
    expected = x * mask + df.iloc[0].values * ~mask
    assert np.allclose(out.iloc[0].values, expected)


def test_independent_dict_init_save_stores_mean_cov_tuple():
    """Tabular.save persists dict-initialized data as (mean, cov) (targets _tabular.py:178)."""
    mean = np.array([1.0, 2.0])
    cov = np.array([[1.0, 0.0], [0.0, 1.0]])
    masker = shap.maskers.Independent({"mean": mean, "cov": cov})
    mock_block = MagicMock()
    with patch("shap.maskers._tabular.Serializer") as serializer_cls:
        serializer_cls.return_value.__enter__.return_value = mock_block
        with tempfile.TemporaryFile() as f:
            masker.save(f)
    data_calls = [c.args for c in mock_block.save.call_args_list if c.args and c.args[0] == "data"]
    assert len(data_calls) == 1
    saved_mean, saved_cov = data_calls[0][1]
    assert np.allclose(saved_mean, mean)
    assert np.allclose(saved_cov, cov)


def test_independent_integer_delta_mask_noop_chained_and_toggle():
    """Integer delta masks: noop, main_effects-style chain, and double-toggle (targets _tabular.py:203-210, 230-265)."""
    data = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    x = np.array([10.0, 11.0, 12.0])
    masker = shap.maskers.Independent(data)
    noop = MaskedModel.delta_mask_noop_value

    out_noop, v_noop = masker(np.array([noop, 0], dtype=np.int64), x)
    assert out_noop[0].shape == (2 * data.shape[0], data.shape[1])
    assert v_noop.shape == (2, data.shape[0])
    assert np.all(v_noop[0, :])

    inds = np.array([0, 1], dtype=np.int64)
    masks_me = np.zeros(2 * len(inds), dtype=np.int64)
    masks_me[0] = noop
    last_ind = -1
    for i in range(len(inds)):
        if i > 0:
            masks_me[2 * i] = -last_ind - 1
        masks_me[2 * i + 1] = inds[i]
        last_ind = inds[i]
    out_me, v_me = masker(masks_me, x)
    assert out_me[0].shape == (3 * data.shape[0], data.shape[1])
    assert v_me.shape == (3, data.shape[0])
    assert np.all(v_me[0, :])

    out_tog, v_tog = masker(np.array([-1, 0], dtype=np.int64), x)
    assert out_tog[0].shape == (data.shape[0], data.shape[1])
    assert v_tog.shape == (1, data.shape[0])
    assert np.allclose(out_tog[0], data)
