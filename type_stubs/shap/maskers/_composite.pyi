# Type stubs for shap.maskers._composite
from __future__ import annotations

from typing import Any

import numpy as np

from ._masker import Masker

class Composite(Masker):
    """A masker that combines multiple maskers for different input types."""

    def __init__(self, *maskers: Masker) -> None: ...
    def __call__(self, mask: np.ndarray, *args: Any) -> Any: ...

    # Instance attributes
    maskers: tuple[Masker, ...]
