"""Tests for the `shap.links` module."""

import numpy as np
import pytest

from shap.links import identity, logit


class TestIdentity:
    def test_identity_scalar(self):
        assert identity(5.0) == 5.0

    def test_identity_array(self):
        x = np.array([1.0, 2.0, 3.0])
        result = identity(x)
        np.testing.assert_array_equal(result, x)

    def test_identity_inverse_scalar(self):
        assert identity.inverse(5.0) == 5.0

    def test_identity_inverse_array(self):
        x = np.array([1.0, 2.0, 3.0])
        result = identity.inverse(x)
        np.testing.assert_array_equal(result, x)

    def test_identity_roundtrip(self):
        x = np.array([-1.0, 0.0, 1.0, 100.0])
        np.testing.assert_array_equal(identity.inverse(identity(x)), x)


class TestLogit:
    def test_logit_scalar(self):
        result = logit(0.5)
        assert result == pytest.approx(0.0)

    def test_logit_array(self):
        x = np.array([0.1, 0.5, 0.9])
        result = logit(x)
        expected = np.log(x / (1 - x))
        np.testing.assert_allclose(result, expected)

    def test_logit_inverse_scalar(self):
        result = logit.inverse(0.0)
        assert result == pytest.approx(0.5)

    def test_logit_inverse_array(self):
        x = np.array([-2.0, 0.0, 2.0])
        result = logit.inverse(x)
        expected = 1 / (1 + np.exp(-x))
        np.testing.assert_allclose(result, expected)

    def test_logit_roundtrip(self):
        """logit.inverse(logit(x)) should return x for valid probabilities."""
        x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        np.testing.assert_allclose(logit.inverse(logit(x)), x)

    def test_logit_inverse_roundtrip(self):
        """logit(logit.inverse(x)) should return x."""
        x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        np.testing.assert_allclose(logit(logit.inverse(x)), x)
