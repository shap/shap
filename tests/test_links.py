"""Tests for shap.links module.

Covers the identity and logit link functions and their inverses.
"""

import numpy as np
from numpy.testing import assert_allclose

from shap.links import identity, logit

# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


class TestIdentity:
    """Tests for the identity link function."""

    def test_scalar(self):
        assert identity(5.0) == 5.0

    def test_negative(self):
        assert identity(-3.14) == -3.14

    def test_zero(self):
        assert identity(0.0) == 0.0

    def test_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = identity(arr)
        assert_allclose(result, arr)

    def test_2d_array(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = identity(arr)
        assert_allclose(result, arr)


# ---------------------------------------------------------------------------
# identity.inverse
# ---------------------------------------------------------------------------


class TestIdentityInverse:
    """Tests for the identity inverse function."""

    def test_scalar(self):
        assert identity.inverse(5.0) == 5.0

    def test_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = identity.inverse(arr)
        assert_allclose(result, arr)

    def test_roundtrip(self):
        arr = np.array([-2.0, 0.0, 3.5])
        assert_allclose(identity.inverse(identity(arr)), arr)


# ---------------------------------------------------------------------------
# logit
# ---------------------------------------------------------------------------


class TestLogit:
    """Tests for the logit link function."""

    def test_half(self):
        # logit(0.5) = log(0.5/0.5) = log(1) = 0
        assert_allclose(logit(0.5), 0.0, atol=1e-10)

    def test_known_value(self):
        # logit(0.75) = log(0.75/0.25) = log(3)
        assert_allclose(logit(0.75), np.log(3), atol=1e-10)

    def test_near_one(self):
        # logit near 1 should be large positive
        result = logit(0.999)
        assert result > 0

    def test_near_zero(self):
        # logit near 0 should be large negative
        result = logit(0.001)
        assert result < 0

    def test_symmetry(self):
        # logit(p) = -logit(1-p)
        p = 0.3
        assert_allclose(logit(p), -logit(1 - p), atol=1e-10)

    def test_array(self):
        arr = np.array([0.1, 0.5, 0.9])
        result = logit(arr)
        expected = np.log(arr / (1 - arr))
        assert_allclose(result, expected, atol=1e-10)

    def test_monotonic(self):
        # logit is strictly increasing
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        result = logit(probs)
        assert np.all(np.diff(result) > 0)


# ---------------------------------------------------------------------------
# logit.inverse (sigmoid)
# ---------------------------------------------------------------------------


class TestLogitInverse:
    """Tests for the logit inverse (sigmoid) function."""

    def test_zero(self):
        # sigmoid(0) = 0.5
        assert_allclose(logit.inverse(0.0), 0.5, atol=1e-10)

    def test_large_positive(self):
        # sigmoid of large positive ≈ 1
        assert_allclose(logit.inverse(100.0), 1.0, atol=1e-10)

    def test_large_negative(self):
        # sigmoid of large negative ≈ 0
        assert_allclose(logit.inverse(-100.0), 0.0, atol=1e-10)

    def test_known_value(self):
        # sigmoid(log(3)) = 0.75
        assert_allclose(logit.inverse(np.log(3)), 0.75, atol=1e-10)

    def test_array(self):
        arr = np.array([-2.0, 0.0, 2.0])
        result = logit.inverse(arr)
        expected = 1 / (1 + np.exp(-arr))
        assert_allclose(result, expected, atol=1e-10)

    def test_output_range(self):
        # sigmoid output is always in (0, 1)
        arr = np.array([-50.0, -1.0, 0.0, 1.0, 50.0])
        result = logit.inverse(arr)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_symmetry(self):
        # sigmoid(-x) = 1 - sigmoid(x)
        x = 1.5
        assert_allclose(logit.inverse(-x), 1 - logit.inverse(x), atol=1e-10)


# ---------------------------------------------------------------------------
# Roundtrips
# ---------------------------------------------------------------------------


class TestRoundtrips:
    """Tests for logit → inverse and inverse → logit roundtrips."""

    def test_logit_then_inverse(self):
        probs = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        assert_allclose(logit.inverse(logit(probs)), probs, atol=1e-10)

    def test_inverse_then_logit(self):
        log_odds = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        assert_allclose(logit(logit.inverse(log_odds)), log_odds, atol=1e-10)
