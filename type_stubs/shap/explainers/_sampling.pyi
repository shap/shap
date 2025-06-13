# Type stubs for shap.explainers._sampling
from __future__ import annotations

from typing import Any, Literal

from shap._explanation import Explanation

from ._explainer import Explainer

class SamplingExplainer(Explainer):
    """Uses Sampling SHAP to explain any function's output.

    Sampling SHAP is a model-agnostic method that estimates SHAP values
    by sampling feature coalitions and measuring the change in model output.
    """

    def __init__(
        self,
        model: Any,
        masker: Any,
        link: Any = ...,
        feature_names: list[str] | None = None,
        linearize_link: bool = True,
        **kwargs: Any,
    ) -> None: ...
    def __call__(
        self,
        X: Any,
        max_evals: Literal["auto"] | int = "auto",
        main_effects: bool = False,
        error_bounds: bool = False,
        batch_size: Literal["auto"] | int = "auto",
        outputs: Any = None,
        silent: bool = False,
        **kwargs: Any,
    ) -> Explanation: ...
    def explain_row(
        self,
        *row_args: Any,
        max_evals: Literal["auto"] | int = "auto",
        main_effects: bool = False,
        error_bounds: bool = False,
        batch_size: Literal["auto"] | int = "auto",
        outputs: Any = None,
        silent: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]: ...
