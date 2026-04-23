import numpy as np
import pytest

from shap.utils._masked_model import (
    _upcast_array,
    _assert_output_input_match,
    _convert_delta_mask_to_full,
)


def test_upcast_array_float16():
    arr = np.array([1, 2, 3], dtype=np.float16)
    result = _upcast_array(arr)
    assert result.dtype == np.float32


def test_upcast_array_no_change():
    arr = np.array([1, 2, 3], dtype=np.float32)
    result = _upcast_array(arr)
    assert result.dtype == np.float32


def test_assert_output_input_match_pass():
    inputs = (np.array([1, 2, 3]),)
    outputs = np.array([10, 20, 30])
    _assert_output_input_match(inputs, outputs)


def test_assert_output_input_match_fail():
    inputs = (np.array([1, 2, 3]),)
    outputs = np.array([10, 20])  # mismatch

    with pytest.raises(AssertionError):
        _assert_output_input_match(inputs, outputs)


def test_convert_delta_mask_to_full_simple():
    masks = np.array([0, 1])
    full_masks = np.zeros((2, 2), dtype=bool)

    _convert_delta_mask_to_full(masks, full_masks)

    assert full_masks.shape == (2, 2)