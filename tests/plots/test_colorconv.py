import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from shap.plots.colors._colorconv import (
    _convert,
    _lab2xyz,
    _prepare_colorarray,
    _prepare_lab_array,
    lab2rgb,
    lch2lab,
    xyz2rgb,
)


def test_prepare_colorarray_returns_float64():
    """Input is cast to float64 regardless of original dtype."""
    arr = np.array([[1, 2, 3]], dtype=np.uint8)
    result = _prepare_colorarray(arr)
    assert result.dtype == np.float64


def test_prepare_colorarray_valid_shape():
    """Arrays with 3 channels on the last axis pass without error."""
    arr = np.ones((4, 4, 3))
    result = _prepare_colorarray(arr)
    assert result.shape == (4, 4, 3)


def test_prepare_colorarray_wrong_channel_size_raises():
    """Arrays whose last dimension is not 3 raise ValueError."""
    arr = np.ones((4, 4, 4))
    with pytest.raises(ValueError, match="channel_axis"):
        _prepare_colorarray(arr)


def test_prepare_colorarray_custom_channel_axis():
    """channel_axis parameter selects which axis must have size 3."""
    arr = np.ones((3, 4, 4))
    result = _prepare_colorarray(arr, channel_axis=0)
    assert result.shape == (3, 4, 4)


def test_convert_identity_matrix():
    """Multiplying by the identity matrix returns the original values."""
    arr = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    result = _convert(np.eye(3), arr)
    assert_allclose(result, arr)


def test_convert_preserves_shape():
    """Output shape matches input shape."""
    arr = np.ones((5, 3))
    result = _convert(np.eye(3), arr)
    assert result.shape == arr.shape


def test_xyz2rgb_black():
    """XYZ black point maps to RGB black."""
    xyz = np.array([[0.0, 0.0, 0.0]])
    rgb = xyz2rgb(xyz)
    assert_allclose(rgb, np.zeros((1, 3)), atol=1e-6)


def test_xyz2rgb_output_clipped():
    """Out-of-gamut XYZ values are clipped to [0, 1]."""
    xyz = np.array([[10.0, 10.0, 10.0]])
    rgb = xyz2rgb(xyz)
    assert np.all(rgb >= 0.0)
    assert np.all(rgb <= 1.0)


@pytest.mark.parametrize(
    "xyz",
    [
        np.array([[0.5, 0.5, 0.5]]),  # high values: power-law branch
        np.array([[0.001, 0.001, 0.001]]),  # low values: linear branch
    ],
)
def test_xyz2rgb_output_range(xyz):
    """Both conversion branches produce values in [0, 1]."""
    rgb = xyz2rgb(xyz)
    assert np.all(rgb >= 0.0) and np.all(rgb <= 1.0)


def test_lab2xyz_no_invalid_pixels():
    """Neutral grey LAB produces no invalid Z pixels."""
    lab = np.array([[50.0, 0.0, 0.0]])
    xyz, n_invalid = _lab2xyz(lab)
    assert n_invalid == 0
    assert xyz.shape == (1, 3)


def test_lab2xyz_invalid_z_pixels_counted():
    """Large positive b forces z negative; those pixels are counted in n_invalid."""
    lab = np.array([[10.0, 0.0, 200.0]])
    _, n_invalid = _lab2xyz(lab)
    assert n_invalid > 0


def test_lab2xyz_output_shape():
    """Output shape matches input shape."""
    lab = np.ones((3, 4, 3)) * [50.0, 0.0, 0.0]
    xyz, n_invalid = _lab2xyz(lab)
    assert xyz.shape == (3, 4, 3)
    assert n_invalid == 0


def test_lab2xyz_both_mask_branches():
    """Mix of L values exercises both branches of the 0.2068966 threshold."""
    lab = np.array([[100.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    xyz, _ = _lab2xyz(lab)
    assert xyz.shape == (2, 3)


def test_lab2rgb_white():
    """LAB white [100, 0, 0] converts to sRGB white."""
    lab = np.array([[100.0, 0.0, 0.0]])
    rgb = lab2rgb(lab)
    assert_allclose(rgb, np.ones((1, 3)), atol=1e-3)


def test_lab2rgb_black():
    """LAB black [0, 0, 0] converts to sRGB black."""
    lab = np.array([[0.0, 0.0, 0.0]])
    rgb = lab2rgb(lab)
    assert_allclose(rgb, np.zeros((1, 3)), atol=1e-3)


def test_lab2rgb_output_range():
    """Arbitrary LAB value produces RGB in [0, 1]."""
    lab = np.array([[50.0, 20.0, -30.0]])
    rgb = lab2rgb(lab)
    assert np.all(rgb >= 0.0) and np.all(rgb <= 1.0)


def test_lab2rgb_invalid_z_emits_warning():
    """Invalid Z pixels during conversion trigger a UserWarning."""
    lab = np.array([[10.0, 0.0, 200.0]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lab2rgb(lab)
    assert any("negative Z" in str(w.message) for w in caught)


def test_lab2rgb_batch_input():
    """Batch of LAB values produces matching batch of RGB values."""
    lab = np.array([[50.0, 0.0, 0.0], [75.0, -10.0, 20.0], [25.0, 5.0, -5.0]])
    rgb = lab2rgb(lab)
    assert rgb.shape == (3, 3)


@pytest.mark.parametrize(
    "lch, expected_a, expected_b",
    [
        ([50.0, 30.0, 0.0], 30.0, 0.0),  # h=0: a=C, b=0
        ([50.0, 30.0, np.pi / 2], 0.0, 30.0),  # h=π/2: a=0, b=C
    ],
)
def test_lch2lab_hue_conversion(lch, expected_a, expected_b):
    """LCh hue angle is correctly decomposed into a and b channels."""
    lab = lch2lab(np.array([lch]))
    assert_allclose(lab[..., 1], expected_a, atol=1e-10)
    assert_allclose(lab[..., 2], expected_b, atol=1e-10)


def test_lch2lab_l_channel_preserved():
    """The L channel passes through unchanged."""
    lch = np.array([[75.0, 20.0, 1.0]])
    lab = lch2lab(lch)
    assert_allclose(lab[..., 0], 75.0)


def test_lch2lab_batch_input():
    """Batch input produces matching output shape."""
    lch = np.ones((5, 3)) * [50.0, 20.0, np.pi / 4]
    lab = lch2lab(lch)
    assert lab.shape == (5, 3)


def test_prepare_lab_array_converts_to_float64():
    """Integer arrays are cast to float64."""
    arr = np.array([[1, 2, 3]], dtype=np.int32)
    result = _prepare_lab_array(arr)
    assert result.dtype == np.float64


def test_prepare_lab_array_valid_3_channels():
    """Arrays with exactly 3 channels on the last axis are accepted."""
    arr = np.ones((4, 3))
    result = _prepare_lab_array(arr)
    assert result.shape == (4, 3)


def test_prepare_lab_array_more_than_3_channels_allowed():
    """Arrays with more than 3 channels on the last axis are accepted."""
    arr = np.ones((4, 5))
    result = _prepare_lab_array(arr)
    assert result.shape == (4, 5)


def test_prepare_lab_array_less_than_3_channels_raises():
    """Arrays with fewer than 3 channels raise ValueError."""
    arr = np.ones((4, 2))
    with pytest.raises(ValueError, match="less than 3 channels"):
        _prepare_lab_array(arr)
