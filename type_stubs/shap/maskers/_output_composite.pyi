# Type stubs for shap.maskers._output_composite
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ._masker import Masker

class OutputComposite(Masker):
    """A masker that composes with output generation for text models."""

    def __init__(
        self,
        masker: Masker,
        output_generator: Callable[..., Any],
    ) -> None: ...
    def __call__(self, mask: np.ndarray, *args: Any) -> Any: ...

    # Instance attributes
    masker: Masker
    output_generator: Callable[..., Any]
