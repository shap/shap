# Type stubs for shap.maskers._masker
from __future__ import annotations

from typing import Any, Callable

import numpy as np

class Masker:
    """Base class for all maskers."""

    def __init__(self, **kwargs: Any) -> None: ...
    def __call__(self, mask: np.ndarray, *args: Any) -> Any: ...

    # Properties that may be present on subclasses
    shape: tuple[int, ...]
    data: Any
    clustering: Any
    feature_names: list[str] | Callable[..., list[str]] | None
    text_data: bool
    image_data: bool
    default_batch_size: int
