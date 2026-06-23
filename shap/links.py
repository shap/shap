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
def logit(x: npt.NDArray[Any] | float) -> npt.NDArray[Any] | float:
    """A logit link function useful for going from probability units to log-odds units.

    This uses ``log(x) - log1p(-x)`` for better numerical stability near 0 and 1.
    Inputs exactly equal to 0 or 1 still map to ``-inf`` and ``inf`` respectively.
    """
    return np.log(x) - np.log1p(-x)


@numba.njit
def _logit_inverse(x: npt.NDArray[Any] | float) -> npt.NDArray[Any] | float:
    return 1 / (1 + np.exp(-x))


logit.inverse = _logit_inverse  # type: ignore[attr-defined]
