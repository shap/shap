# Type stubs for shap.models._model
from __future__ import annotations

from typing import Any, Callable

class Model:
    """Base wrapper class for models."""

    def __init__(self, model: Any) -> None: ...
    def __call__(self, *args: Any) -> Any: ...
    @classmethod
    def load(cls, in_file: Any) -> Model: ...
    def save(self, out_file: Any) -> None: ...

    # Instance attributes
    model: Any
    f: Callable[..., Any]
