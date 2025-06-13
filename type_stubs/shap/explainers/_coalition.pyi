# Type stubs for shap.explainers._coalition
from __future__ import annotations

from typing import Any, Literal

from shap._explanation import Explanation

from ._explainer import Explainer

class CoalitionExplainer(Explainer):
    """Computes Owen values for coalitions of features.

    This explainer computes Owen values, which are a generalization of
    Shapley values for feature coalitions defined by a partition tree.
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
        partition_tree: dict[str, Any] | None = None,
    ) -> None: ...
    def __call__(
        self,
        X: Any,
        **kwargs: Any,
    ) -> Explanation: ...
    def explain_row(
        self,
        *row_args: Any,
        max_evals: int = 100,
        main_effects: bool = False,
        error_bounds: bool = False,
        batch_size: Literal["auto"] | int = "auto",
        outputs: Any = None,
        silent: bool = False,
        fixed_context: Literal["auto"] | Any = "auto",
    ) -> dict[str, Any]: ...
    def __str__(self) -> str: ...

class Node:
    """Node in a partition tree."""

    def __init__(self, name: str) -> None: ...

    name: str
    children: list[Node]
