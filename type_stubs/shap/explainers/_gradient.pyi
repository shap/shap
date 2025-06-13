# Type stubs for shap.explainers._gradient
from __future__ import annotations

from typing import Any

import numpy as np

from shap._explanation import Explanation

from ._explainer import Explainer

class GradientExplainer(Explainer):
    """Computes SHAP values using gradients and integrated gradients.

    This explainer computes approximations to SHAP values using gradients,
    which can be more efficient than other methods for large deep networks.
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        batch_size: int = 50,
        local_smoothing: float = 0,
        **kwargs: Any,
    ) -> None: ...
    def __call__(
        self,
        X: Any,
        ranked_outputs: int | None = None,
        output_rank_order: str = "max",
        **kwargs: Any,
    ) -> Explanation: ...
    def shap_values(
        self,
        X: Any,
        ranked_outputs: int | None = None,
        output_rank_order: str = "max",
        nsamples: int = 200,
        **kwargs: Any,
    ) -> np.ndarray | list[np.ndarray]: ...

    # Instance attributes
    expected_value: float | np.ndarray | None
    model: Any
    data: Any
    batch_size: int
    local_smoothing: float
