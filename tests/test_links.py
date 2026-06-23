"""Tests for shap/links.py — identity and logit link functions."""

import numpy as np
import pytest

import shap.links as links

# -- identity


def test_identity_scalar():
    """identity(x) should return x unchanged for a scalar."""
    assert links.identity(3.14) == pytest.approx(3.14)


def test_identity_array():
    """identity applied element-wise should equal the input array."""
    arr = np.array([0.0, 0.5, 1.0, -2.0])
    result = links.identity(arr)
    np.testing.assert_array_almost_equal(result, arr)


def test_identity_zero():
    """identity(0) == 0."""
    assert links.identity(0.0) == 0.0


def test_identity_inverse_roundtrip():
    """identity.inverse(identity(x)) == x."""
    arr = np.array([0.1, 0.5, 0.9])
    np.testing.assert_array_almost_equal(links.identity.inverse(links.identity(arr)), arr)


# --- logit
def test_logit_scaler():
    """logit(0.5) should be 0 (log-odds of 50%)."""
    assert links.logit(0.5) == pytest.approx(0.0, abs=1e-6)


def test_logit_array():
    """logit should match numpy's manual log(p/(1-p))."""
    p = np.array([0.1, 0.25, 0.75, 0.9])
    expected = np.log(p / (1.0 - p))
    np.testing.assert_array_almost_equal(links.logit(p), expected)


def test_logit_inverse_is_sigmoid():
    """logit.inverse should implement the sigmoid / logistic function."""
    x = np.array([-2.0, 0.0, 2.0])
    expected = 1.0 / (1.0 + np.exp(-x))
    np.testing.assert_array_almost_equal(links.logit.inverse(x), expected)


def test_logit_inverse_roundtrip():
    """logit.inverse(logit(p)) ≈ p for valid probabilities."""
    p = np.array([0.1, 0.3, 0.7, 0.9])
    np.testing.assert_array_almost_equal(links.logit.inverse(links.logit(p)), p, decimal=6)


def test_logit_out_of_bounds():
    """logit for invalid probabilities should produce nan."""
    result = links.logit(np.array([-0.1, 1.1]))
    assert np.isnan(result).all()
