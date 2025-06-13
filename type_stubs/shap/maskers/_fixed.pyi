# Type stubs for shap.maskers._fixed
from __future__ import annotations

from typing import Any

import numpy as np

from ._masker import Masker

class Fixed(Masker):
    """A masker that uses fixed values for masking."""

    def __init__(
        self,
        data: Any,
        **kwargs: Any,
    ) -> None: ...
    def __call__(self, mask: np.ndarray, *args: Any) -> Any: ...

    # Instance attributes
    data: Any
