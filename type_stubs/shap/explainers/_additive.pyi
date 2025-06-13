# Type stubs for shap.explainers._additive
from __future__ import annotations

from typing import Any

import numpy as np

from shap._explanation import Explanation

from ._explainer import Explainer

class AdditiveExplainer(Explainer):
    """Uses Additive SHAP to explain the output of additive models.

    This explainer is optimized for additive models and uses the additivity
    property to compute SHAP values efficiently.
    """

    def __init__(
        self,
        model: Any,
        masker: Any,
        link: Any = None,
        feature_names: list[str] | None = None,
        linearize_link: bool = True,
    ) -> None: ...
    def __call__(
        self,
        X: Any,
        **kwargs: Any,
    ) -> Explanation: ...
    def shap_values(
        self,
        X: Any,
        **kwargs: Any,
    ) -> np.ndarray: ...
    @staticmethod
    def supports_model_with_masker(model: Any, masker: Any) -> bool: ...
