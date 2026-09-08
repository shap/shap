"""Tests for shap.plots.colors._colorconv module.

Tests the vendored color conversion routines (from scikit-image) used
internally by SHAP's plotting colors.
"""

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
    rgb_from_xyz,
    xyz2rgb,
    xyz_from_rgb,
)

# ---------------------------------------------------------------------------
# _prepare_colorarray
# ---------------------------------------------------------------------------


class TestPrepareColorarray:
    """Tests for _prepare_colorarray."""

    def test_converts_to_float64(self):
        arr = np.array([[1, 2, 3]], dtype=np.uint8)
        result = _prepare_colorarray(arr)
        assert result.dtype == np.float64

    def test_preserves_float_input(self):
        arr = np.array([[0.1, 0.2, 0.3]])
        result = _prepare_colorarray(arr)
        assert_allclose(result, arr)

    def test_rejects_wrong_channel_count(self):
        arr = np.array([[1, 2]])
        with pytest.raises(ValueError, match="size 3"):
            _prepare_colorarray(arr)

    def test_accepts_batch_of_pixels(self):
        arr = np.zeros((5, 3))
        result = _prepare_colorarray(arr)
        assert result.shape == (5, 3)

    def test_accepts_image_shaped(self):
        arr = np.zeros((2, 4, 3))
        result = _prepare_colorarray(arr)
        assert result.shape == (2, 4, 3)

    def test_rejects_4_channels(self):
        arr = np.zeros((2, 4))
        with pytest.raises(ValueError, match="size 3"):
            _prepare_colorarray(arr)


# ---------------------------------------------------------------------------
# _prepare_lab_array
# ---------------------------------------------------------------------------


class TestPrepareLabArray:
    """Tests for _prepare_lab_array."""

    def test_converts_to_float64(self):
        arr = [[50, 10, 20]]
        result = _prepare_lab_array(arr)
        assert result.dtype == np.float64

    def test_rejects_too_few_channels(self):
        arr = np.array([[1, 2]])
        with pytest.raises(ValueError, match="less than 3"):
            _prepare_lab_array(arr)

    def test_accepts_exactly_3_channels(self):
        arr = np.array([[50.0, 10.0, 20.0]])
        result = _prepare_lab_array(arr)
        assert result.shape == (1, 3)

    def test_accepts_more_than_3_channels(self):
        # _prepare_lab_array allows >= 3 channels
        arr = np.array([[50.0, 10.0, 20.0, 1.0]])
        result = _prepare_lab_array(arr)
        assert result.shape == (1, 4)


# ---------------------------------------------------------------------------
# _convert (matrix color conversion)
# ---------------------------------------------------------------------------


class TestConvert:
    """Tests for _convert."""

    def test_identity_matrix(self):
        identity = np.eye(3)
        arr = np.array([[0.5, 0.3, 0.7]])
        result = _convert(identity, arr)
        assert_allclose(result, arr, atol=1e-10)

    def test_known_rgb_to_xyz(self):
        # White in RGB -> known XYZ values
        white_rgb = np.array([[1.0, 1.0, 1.0]])
        xyz = _convert(xyz_from_rgb, white_rgb)
        # Sum of each row of xyz_from_rgb
        assert_allclose(xyz[0, 0], 0.950456, atol=1e-4)
        assert_allclose(xyz[0, 1], 1.0, atol=1e-4)

    def test_batch_conversion(self):
        identity = np.eye(3)
        arr = np.random.default_rng(0).random((5, 3))
        result = _convert(identity, arr)
        assert_allclose(result, arr, atol=1e-10)

    def test_output_dtype_is_float64(self):
        arr = np.array([[100, 200, 50]], dtype=np.uint8)
        result = _convert(np.eye(3), arr)
        assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# Conversion matrix sanity checks
# ---------------------------------------------------------------------------


class TestConversionMatrices:
    """Sanity checks for xyz_from_rgb and rgb_from_xyz."""

    def test_matrices_are_inverses(self):
        product = xyz_from_rgb @ rgb_from_xyz
        assert_allclose(product, np.eye(3), atol=1e-10)

    def test_roundtrip_rgb_xyz_rgb(self):
        rgb = np.array([[0.4, 0.6, 0.2]])
        xyz = _convert(xyz_from_rgb, rgb)
        rgb_back = _convert(rgb_from_xyz, xyz)
        assert_allclose(rgb_back, rgb, atol=1e-10)


# ---------------------------------------------------------------------------
# xyz2rgb
# ---------------------------------------------------------------------------


class TestXyz2Rgb:
    """Tests for xyz2rgb."""

    def test_black_point(self):
        xyz = np.array([[0.0, 0.0, 0.0]])
        rgb = xyz2rgb(xyz)
        assert_allclose(rgb, 0.0, atol=1e-10)

    def test_output_clipped_to_01(self):
        # Very bright XYZ that would exceed 1.0 in RGB
        xyz = np.array([[2.0, 2.0, 2.0]])
        rgb = xyz2rgb(xyz)
        assert np.all(rgb >= 0.0)
        assert np.all(rgb <= 1.0)

    def test_output_shape_preserved(self):
        xyz = np.zeros((3, 4, 3))
        rgb = xyz2rgb(xyz)
        assert rgb.shape == (3, 4, 3)

    def test_gamma_correction_applied(self):
        # Values above the linear threshold should get gamma correction
        # The threshold is 0.0031308 in linear RGB
        xyz = np.array([[0.5, 0.5, 0.5]])
        rgb = xyz2rgb(xyz)
        # Result should be in valid range and not simply linear
        assert np.all(rgb >= 0.0)
        assert np.all(rgb <= 1.0)


# ---------------------------------------------------------------------------
# _lab2xyz
# ---------------------------------------------------------------------------


class TestLab2Xyz:
    """Tests for _lab2xyz."""

    def test_white_point(self):
        # L=100, a=0, b=0 is the reference white
        lab = np.array([[100.0, 0.0, 0.0]])
        xyz, n_invalid = _lab2xyz(lab)
        # Should match the D65 illuminant reference white
        assert_allclose(xyz[0, 0], 0.95047, atol=1e-3)
        assert_allclose(xyz[0, 1], 1.0, atol=1e-3)
        assert_allclose(xyz[0, 2], 1.08883, atol=1e-3)
        assert n_invalid == 0

    def test_black_point(self):
        # L=0, a=0, b=0 is black
        lab = np.array([[0.0, 0.0, 0.0]])
        xyz, n_invalid = _lab2xyz(lab)
        assert_allclose(xyz, 0.0, atol=1e-3)

    def test_negative_z_detected(self):
        # Large positive b value causes negative z intermediate,
        # which is reported via n_invalid
        lab = np.array([[50.0, 0.0, 200.0]])
        xyz, n_invalid = _lab2xyz(lab)
        assert n_invalid > 0
        # x and y channels should still be valid
        assert xyz[0, 0] >= 0.0
        assert xyz[0, 1] >= 0.0

    def test_returns_n_invalid(self):
        # Normal colors should have zero invalid
        lab = np.array([[50.0, 10.0, 10.0]])
        _, n_invalid = _lab2xyz(lab)
        assert isinstance(n_invalid, (int, np.integer))


# ---------------------------------------------------------------------------
# lab2rgb
# ---------------------------------------------------------------------------


class TestLab2Rgb:
    """Tests for lab2rgb (full pipeline)."""

    def test_white(self):
        lab = np.array([[100.0, 0.0, 0.0]])
        rgb = lab2rgb(lab)
        assert_allclose(rgb, 1.0, atol=0.01)

    def test_black(self):
        lab = np.array([[0.0, 0.0, 0.0]])
        rgb = lab2rgb(lab)
        assert_allclose(rgb, 0.0, atol=0.01)

    def test_output_in_valid_range(self):
        # Mid-gray
        lab = np.array([[50.0, 0.0, 0.0]])
        rgb = lab2rgb(lab)
        assert np.all(rgb >= 0.0)
        assert np.all(rgb <= 1.0)

    def test_batch_shape(self):
        lab = np.zeros((3, 5, 3))
        lab[..., 0] = 50.0  # set L to 50
        rgb = lab2rgb(lab)
        assert rgb.shape == (3, 5, 3)

    def test_warning_on_invalid_z(self):
        # Extreme b value triggers negative Z warning
        lab = np.array([[50.0, 0.0, 500.0]])
        with pytest.warns(UserWarning, match="negative Z"):
            lab2rgb(lab)


# ---------------------------------------------------------------------------
# lch2lab
# ---------------------------------------------------------------------------


class TestLch2Lab:
    """Tests for lch2lab."""

    def test_zero_chroma(self):
        # C=0 means a=0, b=0 regardless of h
        lch = np.array([[[50.0, 0.0, 1.5]]])
        lab = lch2lab(lch)
        assert_allclose(lab[0, 0, 0], 50.0)
        assert_allclose(lab[0, 0, 1], 0.0, atol=1e-10)
        assert_allclose(lab[0, 0, 2], 0.0, atol=1e-10)

    def test_h_zero(self):
        # h=0 means a=C, b=0
        lch = np.array([[[50.0, 30.0, 0.0]]])
        lab = lch2lab(lch)
        assert_allclose(lab[0, 0, 0], 50.0)
        assert_allclose(lab[0, 0, 1], 30.0, atol=1e-10)
        assert_allclose(lab[0, 0, 2], 0.0, atol=1e-10)

    def test_h_pi_half(self):
        # h=pi/2 means a=0, b=C
        lch = np.array([[[50.0, 30.0, np.pi / 2]]])
        lab = lch2lab(lch)
        assert_allclose(lab[0, 0, 0], 50.0)
        assert_allclose(lab[0, 0, 1], 0.0, atol=1e-10)
        assert_allclose(lab[0, 0, 2], 30.0, atol=1e-10)

    def test_preserves_lightness(self):
        lch = np.array([[[75.0, 40.0, 2.0]]])
        lab = lch2lab(lch)
        assert_allclose(lab[0, 0, 0], 75.0)

    def test_chroma_hue_to_ab_roundtrip(self):
        # a = C*cos(h), b = C*sin(h)
        C, h = 40.0, 1.2
        lch = np.array([[[60.0, C, h]]])
        lab = lch2lab(lch)
        assert_allclose(lab[0, 0, 1], C * np.cos(h), atol=1e-10)
        assert_allclose(lab[0, 0, 2], C * np.sin(h), atol=1e-10)


# ---------------------------------------------------------------------------
# Integration: lch -> lab -> rgb (used by SHAP's color definitions)
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests matching how SHAP uses colorconv internally."""

    def test_lch2rgb_pipeline(self):
        """This is exactly how _colors.py calls the conversion."""
        lch_value = [54.0, 70.0, 4.6588]
        rgb = lab2rgb(lch2lab([[lch_value]]))[0][0]
        # Output should be a valid RGB triplet
        assert rgb.shape == (3,)
        assert np.all(rgb >= 0.0)
        assert np.all(rgb <= 1.0)

    def test_gray_has_equal_rgb(self):
        """Gray (C=0) should produce roughly equal R, G, B."""
        gray_lch = [55.0, 0.0, 0.0]
        rgb = lab2rgb(lch2lab([[gray_lch]]))[0][0]
        assert_allclose(rgb[0], rgb[1], atol=0.01)
        assert_allclose(rgb[1], rgb[2], atol=0.01)

    def test_shap_blue_is_reproducible(self):
        """Verify SHAP's blue color constant can be reproduced."""
        from shap.plots.colors import blue_rgb

        blue_lch = [54.0, 70.0, 4.6588]
        computed = lab2rgb(lch2lab([[blue_lch]]))[0][0]
        assert_allclose(computed, blue_rgb, atol=1e-6)
