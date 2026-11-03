from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from .._serializable import Serializable

if TYPE_CHECKING:
    from collections.abc import Callable


class Masker(Serializable):
    """This is the base class for shap maskers."""

    # Subclasses should define these attributes
    shape: tuple[int | None, int] | Callable[..., tuple[int | None, int]]
    clustering: npt.NDArray[Any] | Callable[..., Any] | None

    def __call__(self, mask: bool | npt.NDArray[Any], *args: Any) -> Any:
        """Masker calls accept the same inputs as the model plus a binary mask.

        Parameters
        ----------
        mask : bool or np.ndarray
            A binary mask indicating which features to keep (True) and
            which to mask (False). The length of the mask should match the number of
            features in the input data, usually ``x`` in ``MyMasker__call__``.
            Masked features will be replaced with a background value upon ``__call__``.
            A common such background value is the mean of the feature across
            the training data.

        x : 2D np.ndarray
            Data to be masked, present as explicit kwarg in most child classes.
            Usually contains a 2D dataframe-like value representing the input data,
            called ``x``, usually ``np.ndarray``.
            The columns of ``x`` correspond to the indices in ``mask``.

        *args : Any
            The remaining arguments are the same as those passed to the model.
        """
        pass

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
