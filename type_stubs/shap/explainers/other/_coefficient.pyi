# Type stubs for shap.explainers.other._coefficient
from __future__ import annotations

from typing import Any

import numpy as np

from .._explainer import Explainer

class Coefficient(Explainer):
    """Linear coefficient explainer for comparison with SHAP."""

    def __init__(
        self,
        model: Any,
        data: np.ndarray,
    ) -> None: ...
    def attributions(
        self,
        X: np.ndarray,
    ) -> np.ndarray: ...

    # Instance attributes
    model: Any
    data: np.ndarray
