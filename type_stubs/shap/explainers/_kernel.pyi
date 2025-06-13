# Type stubs for shap.explainers._kernel
from __future__ import annotations

from typing import Any, Literal

import numpy as np

from shap._explanation import Explanation

from ._explainer import Explainer

class KernelExplainer(Explainer):
    """Uses Kernel SHAP to explain any function's output.

    Kernel SHAP is a model-agnostic method that can explain any model
    by treating it as a black box and using the Kernel SHAP algorithm.
    """

    def __init__(
        self,
        model: Any,
        data: Any,
        feature_names: list[str] | None = None,
        link: Literal["identity", "logit"] | Any = "identity",
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
        nsamples: Literal["auto"] | int = "auto",
        l1_reg: Literal["auto", "aic", "bic"] | float = "auto",
        silent: bool = False,
        **kwargs: Any,
    ) -> np.ndarray | list[np.ndarray]: ...
    def explain(
        self,
        instance: Any,
        **kwargs: Any,
    ) -> dict[str, Any]: ...

    # Instance attributes
    model: Any
    data: Any
    data_feature_names: list[str] | None
    link: Any
    keep_index: bool
    keep_index_ordered: bool
