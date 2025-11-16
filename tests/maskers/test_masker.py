"""This file contains tests for the base Masker class behavior through public API.

These tests verify the True/False mask conversion behavior by creating custom
maskers that use the base class's _standardize_mask internally (as subclasses
should), while test code only calls the public __call__ method.
"""

import numpy as np

import shap


def test_masker_true_mask_standardization():
    """Test that True mask is converted to all-ones array via public API."""

    # This masker properly uses _standardize_mask internally
    class TestMasker(shap.maskers.Masker):
        def __init__(self):
            self.shape = (10, 5)

        def __call__(self, mask, x):
            # Properly use base class's _standardize_mask (as subclasses should)
            standardized_mask = self._standardize_mask(mask, x)
            # Return it so we can verify behavior
            return (standardized_mask,)

    masker = TestMasker()

    # Only call public __call__ method
    result = masker(True, None)

    # Verify the _standardize_mask conversion happened correctly
    assert isinstance(result[0], np.ndarray)
    assert result[0].shape == (5,)
    assert result[0].dtype == bool
    assert np.all(result[0] == True)


def test_masker_false_mask_standardization():
    """Test that False mask is converted to all-zeros array via public API."""

    class TestMasker(shap.maskers.Masker):
        def __init__(self):
            self.shape = (10, 5)

        def __call__(self, mask, x):
            standardized_mask = self._standardize_mask(mask, x)
            return (standardized_mask,)

    masker = TestMasker()
    result = masker(False, None)

    assert isinstance(result[0], np.ndarray)
    assert result[0].shape == (5,)
    assert np.all(result[0] == False)


def test_masker_callable_shape_standardization():
    """Test True/False standardization with callable shape via public API."""

    class TestMasker(shap.maskers.Masker):
        def __init__(self):
            self.shape = lambda x: (x.shape[0], x.shape[1])

        def __call__(self, mask, x):
            standardized_mask = self._standardize_mask(mask, x)
            return (standardized_mask,)

    masker = TestMasker()
    test_data = np.ones((10, 7))

    # Test True
    result = masker(True, test_data)
    assert result[0].shape == (7,)
    assert np.all(result[0] == True)

    # Test False
    result = masker(False, test_data)
    assert result[0].shape == (7,)
    assert np.all(result[0] == False)


def test_masker_array_passthrough():
    """Test that explicit arrays pass through unchanged via public API."""

    class TestMasker(shap.maskers.Masker):
        def __init__(self):
            self.shape = (10, 5)

        def __call__(self, mask, x):
            standardized_mask = self._standardize_mask(mask, x)
            return (standardized_mask,)

    masker = TestMasker()
    input_mask = np.array([True, False, True, False, True])

    result = masker(input_mask, None)
    assert result[0] is input_mask  # Should be same object


def test_masker_multiple_args_callable_shape():
    """Test standardization with callable shape using multiple args."""

    class TestMasker(shap.maskers.Masker):
        def __init__(self):
            self.shape = lambda x, y: (x.shape[0], x.shape[1] + y.shape[1])

        def __call__(self, mask, x, y):
            standardized_mask = self._standardize_mask(mask, x, y)
            return (standardized_mask,)

    masker = TestMasker()
    data1 = np.ones((10, 3))
    data2 = np.ones((10, 4))

    result = masker(True, data1, data2)
    assert result[0].shape == (7,)  # 3 + 4 columns
    assert np.all(result[0] == True)


def test_masker_zero_features():
    """Test standardization with zero features."""

    class TestMasker(shap.maskers.Masker):
        def __init__(self):
            self.shape = (10, 0)

        def __call__(self, mask, x):
            standardized_mask = self._standardize_mask(mask, x)
            return (standardized_mask,)

    masker = TestMasker()

    result = masker(True, None)
    assert result[0].shape == (0,)

    result = masker(False, None)
    assert result[0].shape == (0,)


def test_masker_non_boolean_passthrough():
    """Test that non-True/False values pass through unchanged."""

    class TestMasker(shap.maskers.Masker):
        def __init__(self):
            self.shape = (10, 5)

        def __call__(self, mask, x):
            standardized_mask = self._standardize_mask(mask, x)
            return (standardized_mask,)

    masker = TestMasker()

    # None should pass through
    result = masker(None, None)
    assert result[0] is None

    # Integer should pass through
    result = masker(42, None)
    assert result[0] == 42

    # String should pass through
    result = masker("test", None)
    assert result[0] == "test"


def test_masker_with_existing_public_apis():
    """Test True/False behavior through existing public masker classes."""

    # Test with Fixed masker
    fixed = shap.maskers.Fixed()
    test_input = np.array([1, 2, 3])

    result_true = fixed(True, test_input)
    result_false = fixed(False, test_input)

    # Fixed has 0 features, so True/False both work the same
    assert isinstance(result_true, tuple)
    assert isinstance(result_false, tuple)

    # Test with OutputComposite
    def model(x):
        return np.sum(x)

    output_comp = shap.maskers.OutputComposite(fixed, model)

    result = output_comp(True, test_input)
    assert isinstance(result, tuple)
    assert len(result) == 2

    result = output_comp(False, test_input)
    assert isinstance(result, tuple)
    assert len(result) == 2
