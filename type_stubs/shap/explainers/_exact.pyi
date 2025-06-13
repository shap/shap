# Type stubs for shap.explainers._exact
from __future__ import annotations

from typing import Any, Literal

from shap._explanation import Explanation

from ._explainer import Explainer

class ExactExplainer(Explainer):
    """Computes exact SHAP values by enumerating all possible coalitions.

    This explainer computes exact SHAP values by evaluating the model
    on all possible feature coalitions. It's only practical for small
    numbers of features.
    """

    def __init__(
        self,
        model: Any,
        masker: Any,
        link: Any = ...,
        linearize_link: bool = True,
        feature_names: list[str] | None = None,
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
