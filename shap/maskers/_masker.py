from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from .._serializable import Serializable


class Masker(Serializable):
    """This is the superclass of all maskers."""

    # Subclasses should define these attributes
    shape: tuple[int | None, int] | Callable[..., tuple[int | None, int]]
    clustering: npt.NDArray[Any] | Callable[..., Any] | None

    def __call__(self, mask: bool | npt.NDArray[Any], *args: Any) -> Any:
        """Maskers are callable objects that accept the same inputs as the model plus a binary mask."""

    def mask_shapes(self, *args: Any) -> list[tuple[int, ...]]:
        """Return the shape(s) of the mask(s) that this masker produces.

        Subclasses may override this method.
        """
        raise NotImplementedError("Subclasses should implement mask_shapes")

    def _standardize_mask(self, mask: bool | npt.NDArray[Any], *args: Any) -> npt.NDArray[np.bool_]:
        """This allows users to pass True/False as short hand masks."""
        if mask is True or mask is False:
            if callable(self.shape):
                shape = self.shape(*args)
            else:
                shape = self.shape

            if mask is True:
                return np.ones(shape[1], dtype=bool)
            return np.zeros(shape[1], dtype=bool)
        return mask  # type: ignore[return-value]
