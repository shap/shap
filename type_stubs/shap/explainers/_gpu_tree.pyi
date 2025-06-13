# Type stubs for shap.explainers._gpu_tree
from __future__ import annotations

from typing import Any, Literal

import numpy as np

from shap._explanation import Explanation

from ._explainer import Explainer

class GPUTreeExplainer(Explainer):
    """GPU-accelerated Tree SHAP explainer for tree ensemble models.

    This explainer provides GPU acceleration for Tree SHAP algorithms,
    offering significant speedup for large datasets and models.
    """

    def __init__(
        self,
        model: Any,
        data: np.ndarray | None = None,
        model_output: Literal["raw", "probability", "log_loss"] | str = "raw",
        feature_perturbation: Literal["auto", "interventional", "tree_path_dependent"] = "auto",
        feature_names: list[str] | None = None,
        **kwargs: Any,
    ) -> None: ...
    def __call__(
        self,
        X: Any,
        y: np.ndarray | None = None,
        interactions: bool = False,
        check_additivity: bool = True,
        **kwargs: Any,
    ) -> Explanation: ...
    def shap_values(
        self,
        X: Any,
        y: np.ndarray | None = None,
        tree_limit: int | None = None,
        check_additivity: bool = True,
    ) -> np.ndarray: ...
    def shap_interaction_values(
        self,
        X: Any,
        y: np.ndarray | None = None,
        tree_limit: int | None = None,
    ) -> np.ndarray: ...
    @staticmethod
    def supports_model_with_masker(model: Any, masker: Any) -> bool: ...

    # Instance attributes
    expected_value: float | np.ndarray | None
    model: Any
    model_output: str
    feature_perturbation: str
