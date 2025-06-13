# Type stubs for shap.explainers._partition
from __future__ import annotations

from typing import Any, Literal

from shap._explanation import Explanation

from ._explainer import Explainer

class PartitionExplainer(Explainer):
    """Uses the Partition SHAP method to explain any function's output.

    Partition SHAP computes Shapley values recursively through a hierarchy
    of features, enabling hierarchical explanations and Owen values.
    """

    def __init__(
        self,
        model: Any,
        masker: Any,
        *,
        output_names: list[str] | None = None,
        link: Any = ...,
        linearize_link: bool = True,
        feature_names: list[str] | None = None,
        **call_args: Any,
    ) -> None: ...
    def __call__(
        self,
        *args: Any,
        max_evals: int = 500,
        fixed_context: Any = None,
        main_effects: bool = False,
        error_bounds: bool = False,
        batch_size: Literal["auto"] | int = "auto",
        outputs: Any = None,
        silent: bool = False,
    ) -> Explanation: ...
    def explain_row(
        self,
        *row_args: Any,
        max_evals: int = 500,
        fixed_context: Any = None,
        main_effects: bool = False,
        error_bounds: bool = False,
        batch_size: Literal["auto"] | int = "auto",
        outputs: Any = None,
        silent: bool = False,
    ) -> dict[str, Any]: ...
    def __str__(self) -> str: ...
