from __future__ import annotations

from typing import Any

import numba
import numpy as np
import numpy.typing as npt


@numba.njit
def identity(x: npt.NDArray[Any] | float) -> npt.NDArray[Any] | float:
    """A no-op link function."""
    return x


@numba.njit
def _identity_inverse(x: npt.NDArray[Any] | float) -> npt.NDArray[Any] | float:
    return x


identity.inverse = _identity_inverse  # type: ignore[attr-defined]


@numba.njit
def logit(x: npt.NDArray[Any] | float, eps: float = 1e-10) -> npt.NDArray[Any] | float:
    """A Numba-jitted logit link function, useful for going from probability units to log-odds units.

    Handles both scalar and array inputs correctly in Numba nopython mode
    by using np.maximum/np.minimum instead of np.clip.

    Parameters
    ----------
    x : np.ndarray or float
        Input probability or array of probabilities.
    eps : float, optional
        Small epsilon value to prevent log(0) or division by zero. Defaults to 1e-10.

    Returns
    -------
    np.ndarray or float
        The log-odds corresponding to x, with values clipped.
        Type matches the input type (scalar or array).
    """
    # Define bounds
    lower_bound = eps
    upper_bound = 1.0 - eps  # Use 1.0 for floating point
    # Clip using np.maximum and np.minimum. Numba handles these element-wise
    # functions correctly for both scalar and array inputs in nopython mode.
    clipped_x = np.maximum(lower_bound, np.minimum(x, upper_bound))
    # Logit transformation. np.log works element-wise for arrays
    # and also works for scalars within numba.
    # Using 1.0 ensures floating-point arithmetic.
    return np.log(clipped_x / (1.0 - clipped_x))


@numba.njit
def _logit_inverse(x: npt.NDArray[Any] | float) -> npt.NDArray[Any] | float:
    return 1 / (1 + np.exp(-x))


logit.inverse = _logit_inverse  # type: ignore[attr-defined]
