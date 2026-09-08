import importlib.util
import os

import numpy as np
import pytest

# Import links.py directly to avoid triggering shap/__init__.py which requires compiled extensions
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shap", "links.py"))
spec = importlib.util.spec_from_file_location("shap_links", file_path)
links = importlib.util.module_from_spec(spec)
spec.loader.exec_module(links)


def test_identity_scalar():
    """Tests the identity link function with scalar input."""
    x = 0.5
    assert links.identity(x) == x


def test_identity_array():
    """Tests the identity link function with array input."""
    x = np.array([0.1, 0.5, 0.9])
    np.testing.assert_array_equal(links.identity(x), x)


def test_identity_inverse_scalar():
    """Tests the identity inverse link function with scalar input."""
    x = 0.5
    assert links.identity.inverse(x) == x


def test_identity_inverse_array():
    """Tests the identity inverse link function with array input."""
    x = np.array([0.1, 0.5, 0.9])
    np.testing.assert_array_equal(links.identity.inverse(x), x)


def test_identity_roundtrip():
    """Tests the round-trip consistency of the identity link function."""
    # Array
    x_arr = np.array([0.1, 0.5, 0.9])
    np.testing.assert_array_equal(links.identity.inverse(links.identity(x_arr)), x_arr)
    # Scalar
    x_scalar = 0.72
    assert links.identity.inverse(links.identity(x_scalar)) == x_scalar


@pytest.mark.parametrize("x", [0.1, 0.5, 0.8])
def test_logit_scalar(x):
    """Tests the logit link function with scalar input."""
    expected = np.log(x / (1 - x))
    assert links.logit(x) == pytest.approx(expected)


def test_logit_array():
    """Tests the logit link function with array input."""
    x = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    expected = np.log(x / (1 - x))
    np.testing.assert_allclose(links.logit(x), expected)


@pytest.mark.parametrize("x", [-2.0, 0.0, 2.0])
def test_logit_inverse_scalar(x):
    """Tests the logit inverse link function with scalar input."""
    expected = 1 / (1 + np.exp(-x))
    assert links.logit.inverse(x) == pytest.approx(expected)


def test_logit_inverse_array():
    """Tests the logit inverse link function with array input."""
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    expected = 1 / (1 + np.exp(-x))
    np.testing.assert_allclose(links.logit.inverse(x), expected)


def test_logit_roundtrip():
    """Tests the round-trip consistency of the logit link function."""
    # Probabilities to log-odds and back
    p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    np.testing.assert_allclose(links.logit.inverse(links.logit(p)), p)

    # Log-odds to probabilities and back
    z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    np.testing.assert_allclose(links.logit(links.logit.inverse(z)), z)
