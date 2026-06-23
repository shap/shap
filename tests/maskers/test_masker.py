"""This file contains tests for the base Masker class behavior through public masker APIs.

Tests verify _standardize_mask behavior by using public maskers like Fixed,
Composite, and OutputComposite with True/False masks.
"""

import numpy as np

import shap


class _TupleShapeMasker(shap.maskers.Masker):
    def __init__(self):
        self.shape = (None, 4)
        self.clustering = None


class _CallableShapeMasker(shap.maskers.Masker):
    def __init__(self):
        self.shape = lambda values: (None, len(values))
        self.clustering = None


def test_fixed_masker_with_true_mask():
    """Test Fixed masker with True mask (triggers _standardize_mask)."""
    masker = shap.maskers.Fixed()
    test_input = np.array([1, 2, 3, 4, 5])

    # Fixed masker has 0 features, but True mask should still work
    result = masker(True, test_input)

    assert isinstance(result, tuple)
    assert isinstance(result[0], list)
    assert np.array_equal(result[0][0], test_input)


def test_fixed_masker_with_false_mask():
    """Test Fixed masker with False mask (triggers _standardize_mask)."""
    masker = shap.maskers.Fixed()
    test_input = np.array([1, 2, 3, 4, 5])

    result = masker(False, test_input)

    assert isinstance(result, tuple)
    assert isinstance(result[0], list)
    assert np.array_equal(result[0][0], test_input)


def test_output_composite_with_true_mask():
    """Test OutputComposite with True mask (triggers _standardize_mask in underlying masker)."""
    masker = shap.maskers.Fixed()

    def simple_model(x):
        return np.sum(x)

    output_composite = shap.maskers.OutputComposite(masker, simple_model)
    test_input = np.array([1, 2, 3, 4, 5])

    result = output_composite(True, test_input)

    assert isinstance(result, tuple)
    assert len(result) == 2  # masked input + model output


def test_output_composite_with_false_mask():
    """Test OutputComposite with False mask (triggers _standardize_mask in underlying masker)."""
    masker = shap.maskers.Fixed()

    def simple_model(x):
        return np.sum(x)

    output_composite = shap.maskers.OutputComposite(masker, simple_model)
    test_input = np.array([1, 2, 3, 4, 5])

    result = output_composite(False, test_input)

    assert isinstance(result, tuple)
    assert len(result) == 2


def test_standardize_mask_with_tuple_shape_for_true_and_false():
    """Test _standardize_mask for tuple-defined shape with boolean masks."""
    masker = _TupleShapeMasker()

    true_mask = masker._standardize_mask(True, np.array([1, 2, 3, 4]))
    false_mask = masker._standardize_mask(False, np.array([1, 2, 3, 4]))

    np.testing.assert_array_equal(true_mask, np.array([True, True, True, True]))
    np.testing.assert_array_equal(false_mask, np.array([False, False, False, False]))


def test_standardize_mask_with_callable_shape_for_true_and_false():
    """Test _standardize_mask for callable shape with boolean masks."""
    masker = _CallableShapeMasker()

    values = np.array([10, 20, 30])
    true_mask = masker._standardize_mask(True, values)
    false_mask = masker._standardize_mask(False, values)

    np.testing.assert_array_equal(true_mask, np.array([True, True, True]))
    np.testing.assert_array_equal(false_mask, np.array([False, False, False]))


def test_standardize_mask_passthrough_for_explicit_mask_array():
    """Test _standardize_mask returns explicit mask arrays unchanged."""
    masker = _TupleShapeMasker()
    explicit_mask = np.array([True, False, True, False])

    standardized = masker._standardize_mask(explicit_mask, np.array([1, 2, 3, 4]))

    np.testing.assert_array_equal(standardized, explicit_mask)
