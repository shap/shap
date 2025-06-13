# Type stubs for shap.models._text_generation
from __future__ import annotations

from typing import Any

from ._model import Model

class TextGeneration(Model):
    """Model wrapper for text generation models."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        **kwargs: Any,
    ) -> None: ...
    def __call__(self, *args: Any) -> Any: ...

    # Instance attributes
    model: Any
    tokenizer: Any
