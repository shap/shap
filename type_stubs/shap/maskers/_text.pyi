# Type stubs for shap.maskers._text
from __future__ import annotations

from typing import Any

import numpy as np

from ._masker import Masker

class Text(Masker):
    """A masker for text data that creates token masks."""

    def __init__(
        self,
        tokenizer: Any,
        mask_token: str = "[MASK]",
        collapse_mask_token: bool = False,
        **kwargs: Any,
    ) -> None: ...
    def __call__(self, mask: np.ndarray, *args: Any) -> Any: ...
    def data_transform(self, *args: Any) -> Any: ...

    # Instance attributes
    tokenizer: Any
    mask_token: str
    collapse_mask_token: bool
    text_data: bool
    default_batch_size: int
