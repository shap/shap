# Type stubs for shap.explainers.other._lime
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from .._explainer import Explainer

class LimeTabular(Explainer):
    """LIME tabular explainer for comparison with SHAP."""

    def __init__(
        self,
        model: Any,
        data: np.ndarray | pd.DataFrame,
        mode: Literal["classification", "regression"] = "classification",
    ) -> None: ...
    def attributions(
        self,
        X: np.ndarray | pd.DataFrame,
        nsamples: int = 5000,
        num_features: int | None = None,
    ) -> np.ndarray | list[np.ndarray]: ...

    # Instance attributes
    model: Any
    data: np.ndarray
    mode: str
    explainer: Any
    out_dim: int
    flat_out: bool
