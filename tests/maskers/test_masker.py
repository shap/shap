"""This file contains tests for the base Masker class."""

import numpy as np

import shap


def test_masker_standardize_mask_with_true():
    """Test _standardize_mask with True mask."""

    class TestMasker(shap.maskers.Masker):
        def __init__(self):
            self.shape = (10, 5)

        def __call__(self, mask, x):
            return x

    masker = TestMasker()

    # Standardize True should return all ones
    result = masker._standardize_mask(True, None)

    assert isinstance(result, np.ndarray)
    assert result.shape == (5,)
    assert result.dtype == bool
    assert np.all(result == True)


def test_masker_standardize_mask_with_false():
    """Test _standardize_mask with False mask."""

    class TestMasker(shap.maskers.Masker):
        def __init__(self):
            self.shape = (10, 5)

        def __call__(self, mask, x):
            return x

    masker = TestMasker()

    # Standardize False should return all zeros
    result = masker._standardize_mask(False, None)

    assert isinstance(result, np.ndarray)
    assert result.shape == (5,)
    assert result.dtype == bool
    assert np.all(result == False)


def test_masker_standardize_mask_with_callable_shape():
    """Test _standardize_mask with callable shape."""

    class TestMasker(shap.maskers.Masker):
        def __init__(self):
            self.shape = lambda x: (x.shape[0], x.shape[1])

        def __call__(self, mask, x):
            return x

    masker = TestMasker()
    test_data = np.ones((10, 7))

    # Standardize True with callable shape
    result = masker._standardize_mask(True, test_data)

    assert isinstance(result, np.ndarray)
    assert result.shape == (7,)
    assert result.dtype == bool
    assert np.all(result == True)

    # Standardize False with callable shape
    result = masker._standardize_mask(False, test_data)

    assert isinstance(result, np.ndarray)
    assert result.shape == (7,)
    assert result.dtype == bool
    assert np.all(result == False)


def test_masker_standardize_mask_with_array():
    """Test _standardize_mask with array mask (should pass through)."""

    class TestMasker(shap.maskers.Masker):
        def __init__(self):
            self.shape = (10, 5)

        def __call__(self, mask, x):
            return x

    masker = TestMasker()

    # Pass an actual array - should be returned as-is
    input_mask = np.array([True, False, True, False, True])
    result = masker._standardize_mask(input_mask, None)

    assert result is input_mask


def test_masker_standardize_mask_with_various_shapes():
    """Test _standardize_mask with different shape sizes."""

    class TestMasker(shap.maskers.Masker):
        def __init__(self, shape):
            self.shape = shape

        def __call__(self, mask, x):
            return x

    # Test with shape (1, 3)
    masker = TestMasker((1, 3))
    result = masker._standardize_mask(True, None)
    assert result.shape == (3,)
    assert np.all(result == True)

    # Test with shape (100, 20)
    masker = TestMasker((100, 20))
    result = masker._standardize_mask(False, None)
    assert result.shape == (20,)
    assert np.all(result == False)


def test_masker_standardize_mask_callable_shape_with_multiple_args():
    """Test _standardize_mask with callable shape that uses multiple args."""

    class TestMasker(shap.maskers.Masker):
        def __init__(self):
            self.shape = lambda x, y: (x.shape[0], x.shape[1] + y.shape[1])

        def __call__(self, mask, x, y):
            return x

    masker = TestMasker()
    test_data1 = np.ones((10, 3))
    test_data2 = np.ones((10, 4))

    # Shape should be (10, 7) - sum of columns
    result = masker._standardize_mask(True, test_data1, test_data2)

    assert isinstance(result, np.ndarray)
    assert result.shape == (7,)
    assert np.all(result == True)


def test_masker_standardize_mask_preserves_non_boolean():
    """Test that _standardize_mask preserves non-True/False masks."""

    class TestMasker(shap.maskers.Masker):
        def __init__(self):
            self.shape = (10, 5)

        def __call__(self, mask, x):
            return x

    masker = TestMasker()

    # Pass None - should be returned as-is
    result = masker._standardize_mask(None, None)
    assert result is None

    # Pass integer - should be returned as-is
    result = masker._standardize_mask(1, None)
    assert result == 1

    # Pass string - should be returned as-is
    result = masker._standardize_mask("test", None)
    assert result == "test"


def test_masker_standardize_mask_with_zero_columns():
    """Test _standardize_mask with shape having 0 columns."""

    class TestMasker(shap.maskers.Masker):
        def __init__(self):
            self.shape = (10, 0)

        def __call__(self, mask, x):
            return x

    masker = TestMasker()

    # True mask with 0 columns
    result = masker._standardize_mask(True, None)
    assert result.shape == (0,)

    # False mask with 0 columns
    result = masker._standardize_mask(False, None)
    assert result.shape == (0,)
