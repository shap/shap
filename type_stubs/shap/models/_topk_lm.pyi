# Type stubs for shap.models._topk_lm
from __future__ import annotations

from typing import Any

from ._model import Model

class TopKLM(Model):
    """Model wrapper for top-k language models."""

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
