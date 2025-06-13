# Type stubs for shap.explainers._explainer
from __future__ import annotations

from typing import Any

from shap._explanation import Explanation

class Explainer:
    """Base class for SHAP explainers."""

    def __init__(
        self,
        model: Any,
        masker: Any = None,
        link: Any = None,
        algorithm: str = "auto",
        output_names: list[str] | None = None,
        feature_names: list[str] | None = None,
        linearize_link: bool = True,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Explanation: ...
    def explain_row(self, *args: Any, **kwargs: Any) -> Explanation: ...
    def supports_model_with_masker(self, model: Any, masker: Any) -> bool: ...

    # Instance attributes
    model: Any
    masker: Any
    feature_names: list[str] | None
    output_names: list[str] | None
