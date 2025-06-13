# Type stubs for shap.maskers._image
from __future__ import annotations

from typing import Any

import numpy as np

from ._masker import Masker

class Image(Masker):
    """A masker for image data that creates superpixel masks."""

    def __init__(
        self,
        mask_value: float | np.ndarray = 0,
        shape: tuple[int, ...] | None = None,
        **kwargs: Any,
    ) -> None: ...
    def __call__(self, mask: np.ndarray, *args: Any) -> Any: ...

    # Instance attributes
    mask_value: float | np.ndarray
    shape: tuple[int, ...]
    image_data: bool
