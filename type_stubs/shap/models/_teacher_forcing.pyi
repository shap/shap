# Type stubs for shap.models._teacher_forcing
from __future__ import annotations

from typing import Any

from ._model import Model

class TeacherForcing(Model):
    """Model wrapper for teacher forcing text generation."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        **kwargs: Any,
    ) -> None: ...
    def __call__(self, *args: Any) -> Any: ...
    def text_generate(self, *args: Any) -> Any: ...

    # Instance attributes
    model: Any
    tokenizer: Any
