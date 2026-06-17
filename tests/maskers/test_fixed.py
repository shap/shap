"""This file contains tests for the Fixed masker."""

import tempfile

import numpy as np

import shap


def test_fixed_masker_init():
    """Test Fixed masker initialization."""
    masker = shap.maskers.Fixed()

    assert masker.shape == (None, 0)
    assert isinstance(masker.clustering, np.ndarray)
    assert masker.clustering.shape == (0, 4)


def test_fixed_masker_call():
    """Test that Fixed masker returns input unchanged regardless of mask."""
    masker = shap.maskers.Fixed()
    mask = np.array([], dtype=bool)

    # Test with different input types
    inputs = [
        42,  # scalar
        np.array([1, 2, 3]),  # 1D array
        np.array([[1, 2], [3, 4]]),  # 2D array
        "label_class_A",  # string (label use case)
    ]

    for x in inputs:
        result = masker(mask, x)
        assert isinstance(result, tuple) and len(result) == 1
        if isinstance(x, np.ndarray):
            np.testing.assert_array_equal(result[0][0], x)
        else:
            assert result[0][0] == x


def test_fixed_masker_mask_shapes():
    """Test that mask_shapes always returns [(0,)]."""
    masker = shap.maskers.Fixed()
    assert masker.mask_shapes(42) == [(0,)]
    assert masker.mask_shapes(np.array([1, 2, 3])) == [(0,)]


def test_fixed_masker_mask_shapes_with_various_inputs():
    """Test mask_shapes with different inputs (input should not affect output)."""
    masker = shap.maskers.Fixed()

    # The input to mask_shapes doesn't matter for Fixed masker
    assert masker.mask_shapes(np.array([1, 2, 3])) == [(0,)]
    assert masker.mask_shapes([1, 2, 3]) == [(0,)]
    assert masker.mask_shapes("test") == [(0,)]
    assert masker.mask_shapes(42) == [(0,)]


def test_fixed_masker_serialization():
    """Test that Fixed masker can be serialized and deserialized."""
    original = shap.maskers.Fixed()

    with tempfile.TemporaryFile() as f:
        original.save(f)
        f.seek(0)
        loaded = shap.maskers.Fixed.load(f)

    # Verify behavior is preserved
    test_input = np.array([1, 2, 3])
    mask = np.array([], dtype=bool)
    np.testing.assert_array_equal(
        original(mask, test_input)[0][0],
        loaded(mask, test_input)[0][0],
    )
