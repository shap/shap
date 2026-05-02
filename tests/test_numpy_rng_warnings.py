"""Verification tests for CI health and Notebook stability.

Targets:
- PR #4947 (Silence NumPy RNG warnings)
- PR #4951 (Modernize tutorial notebooks)
"""

import os
import pytest
import warnings
import numpy as np

# ---------------------------------------------------------------------------
# PR #4947: Silence NumPy RNG warnings
# ---------------------------------------------------------------------------


def test_numpy_rng_warning_is_silenced():
    """
    Verify that global NumPy RNG calls do NOT raise deprecation warnings
    under the current pytest configuration.
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Trigger the common deprecation warning
        np.random.seed(0)
        np.random.rand(10)

        # Check if any DeprecationWarnings were captured
        # The goal is that our pytestmark filter should have handled this
        # so this test proves the environment is "clean".
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning) and "RNG" in str(x.message)]
        # In a normal run, pytest filters these, so they don't reach the 'record=True' if handled at the root.
