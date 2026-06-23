"""Tests for link functions."""

from __future__ import annotations

import numpy as np

from shap import links


def test_identity_returns_inputs_for_scalar_and_array():
    scalar = 0.37
    arr = np.array([0.1, 0.5, 0.9])

    assert np.isclose(links.identity(scalar), scalar)
    np.testing.assert_allclose(links.identity(arr), arr)


def test_identity_inverse_returns_original_values():
    scalar = -2.0
    arr = np.array([-3.0, 0.0, 3.0])

    assert np.isclose(links.identity.inverse(scalar), scalar)
    np.testing.assert_allclose(links.identity.inverse(arr), arr)


def test_logit_and_inverse_round_trip_for_scalar():
    prob = 0.2
    log_odds = links.logit(prob)
    recovered_prob = links.logit.inverse(log_odds)

    assert np.isclose(recovered_prob, prob)


def test_logit_and_inverse_round_trip_for_array():
    probs = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    log_odds = links.logit(probs)
    recovered_probs = links.logit.inverse(log_odds)

    np.testing.assert_allclose(recovered_probs, probs)


def test_logit_inverse_known_point():
    assert np.isclose(links.logit.inverse(0.0), 0.5)


def test_numba_py_func_paths_execute_python_source_lines():
    probs = np.array([0.2, 0.4, 0.6, 0.8])

    np.testing.assert_allclose(links.identity.py_func(probs), probs)
    np.testing.assert_allclose(links.identity.inverse.py_func(probs), probs)

    log_odds = links.logit.py_func(probs)
    recovered_probs = links.logit.inverse.py_func(log_odds)
    np.testing.assert_allclose(recovered_probs, probs)
