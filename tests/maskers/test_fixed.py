"""This file contains tests for the Fixed masker."""

import tempfile

import numpy as np

import shap


def test_fixed_masker_initialization():
    """Test that Fixed masker initializes with correct attributes."""
    masker = shap.maskers.Fixed()

    # Check that shape is set correctly
    assert masker.shape == (None, 0)

    # Check that clustering is a zero array with shape (0, 4)
    assert isinstance(masker.clustering, np.ndarray)
    assert masker.clustering.shape == (0, 4)
    np.testing.assert_array_equal(masker.clustering, np.zeros((0, 4)))


def test_fixed_masker_call_with_scalar():
    """Test that Fixed masker returns input unchanged with scalar."""
    masker = shap.maskers.Fixed()

    # Test with a scalar
    x = 42
    mask = np.array([], dtype=bool)
    result = masker(mask, x)

    # Should return tuple with single element list containing x
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert isinstance(result[0], list)
    assert len(result[0]) == 1
    assert result[0][0] == x


def test_fixed_masker_call_with_array():
    """Test that Fixed masker returns input unchanged with numpy array."""
    masker = shap.maskers.Fixed()

    # Test with a numpy array
    x = np.array([1, 2, 3, 4, 5])
    mask = np.array([], dtype=bool)
    result = masker(mask, x)

    # Should return tuple with single element list containing x
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert isinstance(result[0], list)
    assert len(result[0]) == 1
    np.testing.assert_array_equal(result[0][0], x)


def test_fixed_masker_call_with_2d_array():
    """Test that Fixed masker returns input unchanged with 2D array."""
    masker = shap.maskers.Fixed()

    # Test with a 2D array
    x = np.array([[1, 2], [3, 4], [5, 6]])
    mask = np.array([], dtype=bool)
    result = masker(mask, x)

    # Should return tuple with single element list containing x
    assert isinstance(result, tuple)
    assert len(result) == 1
    np.testing.assert_array_equal(result[0][0], x)


def test_fixed_masker_call_with_string():
    """Test that Fixed masker returns input unchanged with string."""
    masker = shap.maskers.Fixed()

    # Test with a string (like labels)
    x = "label_class_A"
    mask = np.array([], dtype=bool)
    result = masker(mask, x)

    # Should return tuple with single element list containing x
    assert isinstance(result, tuple)
    assert len(result) == 1
    assert result[0][0] == x


def test_fixed_masker_call_mask_values():
    """Test that Fixed masker ignores mask values and returns input unchanged."""
    masker = shap.maskers.Fixed()

    x = np.array([10, 20, 30])

    # Test with different mask values (should all behave the same)
    mask1 = np.array([], dtype=bool)
    mask2 = np.array([True], dtype=bool)  # Doesn't matter what mask is
    mask3 = np.array([False], dtype=bool)

    result1 = masker(mask1, x)
    result2 = masker(mask2, x)
    result3 = masker(mask3, x)

    # All should return the same result (input unchanged)
    np.testing.assert_array_equal(result1[0][0], x)
    np.testing.assert_array_equal(result2[0][0], x)
    np.testing.assert_array_equal(result3[0][0], x)


def test_fixed_masker_mask_shapes():
    """Test that mask_shapes returns correct shape."""
    masker = shap.maskers.Fixed()

    # Test with various inputs
    x1 = 42
    x2 = np.array([1, 2, 3])
    x3 = np.array([[1, 2], [3, 4]])

    # mask_shapes should always return [(0,)] regardless of input
    assert masker.mask_shapes(x1) == [(0,)]
    assert masker.mask_shapes(x2) == [(0,)]
    assert masker.mask_shapes(x3) == [(0,)]


def test_fixed_masker_serialization_basic():
    """Test that Fixed masker can be serialized and deserialized."""
    original_masker = shap.maskers.Fixed()

    with tempfile.TemporaryFile() as temp_file:
        # Serialize the masker
        original_masker.save(temp_file)

        # Reset file pointer
        temp_file.seek(0)

        # Deserialize the masker
        new_masker = shap.maskers.Fixed.load(temp_file)

    # Check that attributes are preserved
    assert new_masker.shape == original_masker.shape
    np.testing.assert_array_equal(new_masker.clustering, original_masker.clustering)


def test_fixed_masker_serialization_functionality():
    """Test that serialized Fixed masker maintains functionality."""
    original_masker = shap.maskers.Fixed()

    with tempfile.TemporaryFile() as temp_file:
        # Serialize the masker
        original_masker.save(temp_file)
        temp_file.seek(0)

        # Deserialize the masker
        new_masker = shap.maskers.Fixed.load(temp_file)

    # Test that both maskers produce the same output
    test_input = np.array([100, 200, 300])
    test_mask = np.array([], dtype=bool)

    original_result = original_masker(test_mask, test_input)
    new_result = new_masker(test_mask, test_input)

    np.testing.assert_array_equal(original_result[0][0], new_result[0][0])


def test_fixed_masker_with_labels():
    """Test Fixed masker with label-like data (primary use case)."""
    masker = shap.maskers.Fixed()

    # Test with integer labels
    labels_int = np.array([0, 1, 1, 0, 1])
    mask = np.array([], dtype=bool)
    result = masker(mask, labels_int)
    np.testing.assert_array_equal(result[0][0], labels_int)

    # Test with string labels
    labels_str = "class_positive"
    result = masker(mask, labels_str)
    assert result[0][0] == labels_str

    # Test with one-hot encoded labels
    labels_onehot = np.array([[1, 0], [0, 1], [0, 1]])
    result = masker(mask, labels_onehot)
    np.testing.assert_array_equal(result[0][0], labels_onehot)


def test_fixed_masker_return_format():
    """Test that Fixed masker returns correct tuple format."""
    masker = shap.maskers.Fixed()

    x = np.array([1, 2, 3])
    mask = np.array([], dtype=bool)
    result = masker(mask, x)

    # Result should be a tuple
    assert isinstance(result, tuple)

    # Tuple should have length 1
    assert len(result) == 1

    # First element should be a list
    assert isinstance(result[0], list)

    # List should contain one element (the input)
    assert len(result[0]) == 1

    # That element should be the input unchanged
    np.testing.assert_array_equal(result[0][0], x)


def test_fixed_masker_multiple_calls():
    """Test that Fixed masker can be called multiple times consistently."""
    masker = shap.maskers.Fixed()

    x = np.array([7, 8, 9])
    mask = np.array([], dtype=bool)

    # Call multiple times
    result1 = masker(mask, x)
    result2 = masker(mask, x)
    result3 = masker(mask, x)

    # All results should be identical
    np.testing.assert_array_equal(result1[0][0], result2[0][0])
    np.testing.assert_array_equal(result2[0][0], result3[0][0])
    np.testing.assert_array_equal(result1[0][0], x)
