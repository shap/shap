"""Tests for the Impute masker, specifically dict-based initialization.

Regression tests for https://github.com/shap/shap/issues/4372.
"""

from collections import OrderedDict

import numpy as np

from shap.maskers import Impute


def test_impute_dict_initialization():
    """Impute correctly extracts mean/cov from a dict input.

    The previous code used ``data is dict`` (identity check) instead of
    ``isinstance(data, dict)``, so the dict branch was dead code.
    """
    mu = np.zeros(3)
    cov = np.eye(3)
    masker = Impute({"mean": mu, "cov": cov})

    np.testing.assert_array_equal(masker.mean, mu)
    np.testing.assert_array_equal(masker.cov, cov)
    assert masker.data.shape == (1, 3)


def test_impute_dict_subclass():
    """isinstance(data, dict) must match dict subclasses like OrderedDict."""
    mu = np.zeros(3)
    cov = np.eye(3)
    masker = Impute(OrderedDict([("mean", mu), ("cov", cov)]))

    np.testing.assert_array_equal(masker.mean, mu)
    np.testing.assert_array_equal(masker.cov, cov)
    assert masker.data.shape == (1, 3)


def test_impute_dict_mean_only():
    """Dict with 'mean' but no 'cov' should set cov to None."""
    mu = np.array([1.0, 2.0, 3.0])
    masker = Impute({"mean": mu})

    np.testing.assert_array_equal(masker.mean, mu)
    assert masker.cov is None
    assert masker.data.shape == (1, 3)


def test_impute_ndarray_initialization():
    """Impute with a plain ndarray skips the dict branch."""
    data = np.random.randn(100, 4)
    masker = Impute(data)

    np.testing.assert_array_equal(masker.data, data)
    assert not hasattr(masker, "mean")
