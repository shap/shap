# Type stubs for shap.maskers._fixed_composite
from __future__ import annotations

from typing import Any

import numpy as np

from ._masker import Masker

class FixedComposite(Masker):
    """A masker that uses fixed composition for text generation models."""

    def __init__(self, masker: Masker) -> None: ...
    def __call__(self, mask: np.ndarray, *args: Any) -> Any: ...

    # Instance attributes
    masker: Masker
