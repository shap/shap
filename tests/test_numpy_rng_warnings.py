"""Verification tests for CI health and suppress NumPy global RNG deprecation warnings.

Targets:
- PR #4947 (Silence NumPy RNG warnings)
"""

import warnings
import numpy as np


def test_numpy_rng_warning_is_silenced():
    """
    Verify that global NumPy RNG calls do NOT raise deprecation warnings
    under the current pytest configuration (via pyproject.toml).
    """
    with warnings.catch_warnings(record=True) as w:
        # Trigger the common deprecation warning
        np.random.seed(0)
        np.random.rand(10)

        # Check if any NumPy global RNG warnings were captured
        # The goal is that our global filter in pyproject.toml should have handled this.
        rng_warnings = [
            x for x in w 
            if issubclass(x.category, (DeprecationWarning, FutureWarning)) 
            and "NumPy global RNG" in str(x.message)
        ]
        
        assert len(rng_warnings) == 0, f"Found {len(rng_warnings)} unsilenced RNG warnings: {[str(x.message) for x in rng_warnings]}"
