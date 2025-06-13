# Type stubs for shap.explainers.other._maple
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._explainer import Explainer

class Maple(Explainer):
    """MAPLE explainer for comparison with SHAP."""

    def __init__(
        self,
        model: Any,
        data: np.ndarray | pd.DataFrame,
    ) -> None: ...
    def attributions(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> np.ndarray | list[np.ndarray]: ...

    # Instance attributes
    model: Any
    data: np.ndarray
    data_mean: np.ndarray
    explainer: Any
    out_dim: int
    flat_out: bool

class TreeMaple(Explainer):
    """Tree MAPLE explainer for comparison with SHAP."""

    def __init__(
        self,
        model: Any,
        data: np.ndarray | pd.DataFrame,
    ) -> None: ...
    def attributions(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> np.ndarray | list[np.ndarray]: ...

    # Instance attributes
    model: Any
    data: np.ndarray
    data_mean: np.ndarray
    explainer: Any
    out_dim: int
    flat_out: bool
