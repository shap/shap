"""This file contains tests for the base Masker class behavior through public masker APIs.

Tests verify _standardize_mask behavior by using public maskers like Fixed,
Composite, and OutputComposite with True/False masks.
"""

import numpy as np

import shap


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
