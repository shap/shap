from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def identity(x: npt.NDArray[Any] | float) -> npt.NDArray[Any] | float:
    """A no-op link function."""
    return x


def _identity_inverse(x: npt.NDArray[Any] | float) -> npt.NDArray[Any] | float:
    return x


identity.inverse = _identity_inverse  # type: ignore[attr-defined]


def logit(x: npt.NDArray[Any] | float) -> npt.NDArray[Any] | float:
    """A logit link function useful for going from probability units to log-odds units."""
    return np.log(x / (1 - x))


def _logit_inverse(x: npt.NDArray[Any] | float) -> npt.NDArray[Any] | float:
    return 1 / (1 + np.exp(-x))


logit.inverse = _logit_inverse  # type: ignore[attr-defined]
