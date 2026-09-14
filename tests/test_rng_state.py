"""Regression tests for GitHub issue #4988.

corrgroups60() and independentlinear60() must not corrupt the global
NumPy RNG state.  Before the fix, both functions stored the None return
value of np.random.seed() and later called np.random.seed(None), which
re-seeds from OS entropy instead of restoring the original state.
"""

import numpy as np

import shap


def test_corrgroups60_preserves_rng_state():
    """Calling corrgroups60 must leave the global RNG state unchanged."""
    np.random.seed(12345)
    state_before = np.random.get_state()

    shap.datasets.corrgroups60()

    state_after = np.random.get_state()

    # The state tuple is ('MT19937', array_of_624_uint32, pos, has_gauss, cached_gaussian).
    # Compare every element.
    assert state_before[0] == state_after[0], "RNG algorithm name changed"
    np.testing.assert_array_equal(state_before[1], state_after[1])
    assert state_before[2] == state_after[2], "RNG position changed"
    assert state_before[3] == state_after[3], "has_gauss flag changed"
    assert state_before[4] == state_after[4], "cached_gaussian changed"


def test_independentlinear60_preserves_rng_state():
    """Calling independentlinear60 must leave the global RNG state unchanged."""
    np.random.seed(12345)
    state_before = np.random.get_state()

    shap.datasets.independentlinear60()

    state_after = np.random.get_state()

    assert state_before[0] == state_after[0], "RNG algorithm name changed"
    np.testing.assert_array_equal(state_before[1], state_after[1])
    assert state_before[2] == state_after[2], "RNG position changed"
    assert state_before[3] == state_after[3], "has_gauss flag changed"
    assert state_before[4] == state_after[4], "cached_gaussian changed"


def test_corrgroups60_deterministic_output():
    """The data returned by corrgroups60 must still be reproducible."""
    X1, y1 = shap.datasets.corrgroups60(n_points=100)
    X2, y2 = shap.datasets.corrgroups60(n_points=100)
    np.testing.assert_array_equal(X1.values, X2.values)
    np.testing.assert_array_equal(y1, y2)


def test_independentlinear60_deterministic_output():
    """The data returned by independentlinear60 must still be reproducible."""
    X1, y1 = shap.datasets.independentlinear60(n_points=100)
    X2, y2 = shap.datasets.independentlinear60(n_points=100)
    np.testing.assert_array_equal(X1.values, X2.values)
    np.testing.assert_array_equal(y1, y2)


def test_rng_sequence_continues_after_corrgroups60():
    """Random numbers generated after corrgroups60 must follow the
    pre-call sequence, proving the state was truly restored (not just
    re-seeded to a different fixed value)."""
    np.random.seed(42)
    expected = np.random.randn(5)

    np.random.seed(42)
    shap.datasets.corrgroups60()
    actual = np.random.randn(5)

    np.testing.assert_array_equal(expected, actual)


def test_rng_sequence_continues_after_independentlinear60():
    """Random numbers generated after independentlinear60 must follow the
    pre-call sequence, proving the state was truly restored."""
    np.random.seed(42)
    expected = np.random.randn(5)

    np.random.seed(42)
    shap.datasets.independentlinear60()
    actual = np.random.randn(5)

    np.testing.assert_array_equal(expected, actual)
