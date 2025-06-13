# Type stubs for shap.models._transformers_pipeline
from __future__ import annotations

from typing import Any

from ._model import Model

class TransformersPipeline(Model):
    """Model wrapper for transformers pipeline objects."""

    def __init__(
        self,
        pipeline: Any,
        **kwargs: Any,
    ) -> None: ...
    def __call__(self, *args: Any) -> Any: ...

    # Instance attributes
    pipeline: Any
    inner_model: Any
