"""This file contains tests for the Image masker."""

import tempfile

import numpy as np
import pytest

import shap
from shap.utils import assert_import

try:
    assert_import("cv2")
except ImportError:
    pytestmark = pytest.mark.skip("opencv not installed")


def test_serialization_image_masker_inpaint_telea():
    """Make sure image serialization works with inpaint telea mask."""
    test_image_height = 500
    test_image_width = 500
    test_data = np.ones((test_image_height, test_image_width, 3)) * 50
    test_shape = (test_image_height, test_image_width, 3)
    # initialize image masker
    original_image_masker = shap.maskers.Image("inpaint_telea", test_shape)

    with tempfile.TemporaryFile() as temp_serialization_file:
        # serialize independent masker
        original_image_masker.save(temp_serialization_file)

        temp_serialization_file.seek(0)

        # deserialize masker
        new_image_masker = shap.maskers.Image.load(temp_serialization_file)

    mask = np.ones((test_image_height, test_image_width, 3))
    mask = mask.astype(int)
    mask[0][0] = 0
    mask[4][0] = 0

    # comparing masked values
    assert np.array_equal(original_image_masker(mask, test_data), new_image_masker(mask, test_data))


def test_serialization_image_masker_inpaint_ns():
    """Make sure image serialization works with inpaint ns mask."""
    test_image_height = 500
    test_image_width = 500
    test_data = np.ones((test_image_height, test_image_width, 3)) * 50
    test_shape = (test_image_height, test_image_width, 3)
    # initialize image masker
    original_image_masker = shap.maskers.Image("inpaint_ns", test_shape)

    with tempfile.TemporaryFile() as temp_serialization_file:
        # serialize independent masker
        original_image_masker.save(temp_serialization_file)

        temp_serialization_file.seek(0)

        # deserialize masker
        new_image_masker = shap.maskers.Image.load(temp_serialization_file)

    mask = np.ones((test_image_height, test_image_width, 3))
    mask = mask.astype(int)
    mask[0][0] = 0
    mask[4][0] = 0

    # comparing masked values
    assert np.array_equal(original_image_masker(mask, test_data), new_image_masker(mask, test_data))


def test_serialization_image_masker_blur():
    """Make sure image serialization works with blur mask."""
    test_image_height = 500
    test_image_width = 500
    test_data = np.ones((test_image_height, test_image_width, 3)) * 50
    test_shape = (test_image_height, test_image_width, 3)
    # initialize image masker
    original_image_masker = shap.maskers.Image("blur(10,10)", test_shape)

    with tempfile.TemporaryFile() as temp_serialization_file:
        # serialize independent masker
        original_image_masker.save(temp_serialization_file)

        temp_serialization_file.seek(0)

        # deserialize masker
        new_image_masker = shap.maskers.Image.load(temp_serialization_file)

    mask = np.ones((test_image_height, test_image_width, 3))
    mask = mask.astype(int)
    mask[0][0] = 0
    mask[4][0] = 0

    # comparing masked values
    assert np.array_equal(original_image_masker(mask, test_data), new_image_masker(mask, test_data))


def test_serialization_image_masker_mask():
    """Make sure image serialization works."""
    test_image_height = 500
    test_image_width = 500
    test_data = np.ones((test_image_height, test_image_width, 3)) * 50
    test_shape = (test_image_height, test_image_width, 3)
    test_mask = np.ones((test_image_height, test_image_width, 3))
    # initialize image masker
    original_image_masker = shap.maskers.Image(test_mask, test_shape)

    with tempfile.TemporaryFile() as temp_serialization_file:
        # serialize independent masker
        original_image_masker.save(temp_serialization_file)

        temp_serialization_file.seek(0)

        # deserialize masker
        new_image_masker = shap.maskers.Image.load(temp_serialization_file)

    mask = np.ones((test_image_height, test_image_width, 3))
    mask = mask.astype(int)
    mask[0][0] = 0
    mask[4][0] = 0

    # comparing masked values
    assert np.array_equal(original_image_masker(mask, test_data), new_image_masker(mask, test_data))


def test_init_string_mask_without_shape():
    """Make sure masker raises error when initializing with string mask value without shape"""
    with pytest.raises(TypeError):
        shap.maskers.Image("inpaint_telea")


def test_init_ndarray_mask_without_shape():
    """Make sure that shape is inferred correctly from np.array when no shape is passed"""
    mask_value = np.zeros((5, 5, 3))
    image_masker = shap.maskers.Image(mask_value)  # no shape parameter passed
    assert image_masker.input_shape == (5, 5, 3)
    assert image_masker.shape == (1, 75)  # 5*5*3 = 75, when flattened


def test_init_scalar_mask_with_shape():
    """Make sure mask_value is expanded to a flat array of the scalar when mask_value is an int"""
    image_masker = shap.maskers.Image(5, shape=(5, 5, 3))
    assert image_masker.input_shape == (5, 5, 3)
    assert image_masker.mask_value.shape == (75,)
    assert np.all(image_masker.mask_value == 5)


def test_call_with_torch_tensor():
    """x is converted from torch Tensor to numpy array before masking"""
    torch = pytest.importorskip("torch")
    image_masker = shap.maskers.Image(np.zeros((5, 5, 3)))
    x = torch.zeros(5, 5, 3)
    mask = np.ones(75, dtype=bool)
    result = image_masker(mask, x)
    assert isinstance(result[0], np.ndarray)
