# Type stubs for shap.explainers._deep
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from shap._explanation import Explanation

from ._explainer import Explainer

class DeepExplainer(Explainer):
    """Computes SHAP values for deep learning models using DeepLIFT or DeepSHAP.

    This explainer is optimized for deep neural networks and uses gradients
    and activations to compute SHAP values efficiently.
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        session: Any = None,
        learning_phase_flags: list[bool] | None = None,
    ) -> None: ...
    def __call__(
        self,
        X: list | np.ndarray | pd.DataFrame | Any,
    ) -> Explanation: ...
    def shap_values(
        self,
        X: Any,
        ranked_outputs: int | None = None,
        output_rank_order: str = "max",
        check_additivity: bool = True,
    ) -> np.ndarray | list[np.ndarray]: ...

    # Instance attributes
    expected_value: float | np.ndarray | None
    explainer: Any
