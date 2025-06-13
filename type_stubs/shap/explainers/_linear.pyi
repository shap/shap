# Type stubs for shap.explainers._linear
from __future__ import annotations

from typing import Any

import numpy as np

from shap._explanation import Explanation

from ._explainer import Explainer

class LinearExplainer(Explainer):
    """Computes SHAP values for linear models.

    This explainer is optimized for linear models and can compute
    exact SHAP values efficiently using the linear model's coefficients.
    """

    def __init__(
        self,
        model: Any,
        masker: Any,
        link: Any = ...,
        feature_names: list[str] | None = None,
        **kwargs: Any,
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

    # Instance attributes
    expected_value: float | np.ndarray | None
    model: Any
