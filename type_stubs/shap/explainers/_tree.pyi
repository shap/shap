# Type stubs for shap.explainers._tree
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from shap._explanation import Explanation
from shap.explainers._explainer import Explainer

# Module-level functions
def _safe_check_tree_instance_experimental(tree_instance: Any) -> None:
    """Check if a tree instance has an experimental integration with shap TreeExplainer class."""
    ...

def _check_xgboost_version(v: str) -> None:
    """Check if XGBoost version is compatible with SHAP."""
    ...

def _xgboost_n_iterations(tree_limit: int, num_stacked_models: int) -> int:
    """Convert number of trees to number of iterations for XGBoost models."""
    ...

def _xgboost_cat_unsupported(model: Any) -> None:
    """Check if model has unsupported categorical features."""
    ...

# Module-level constants
output_transform_codes: dict[str, int]
feature_perturbation_codes: dict[str, int]

class TreeExplainer(Explainer):
    """Uses Tree SHAP algorithms to explain the output of ensemble tree models.

    Tree SHAP is a fast and exact method to estimate SHAP values for tree models
    and ensembles of trees, under several different possible assumptions about
    feature dependence. It depends on fast C++ implementations either inside an
    external model package or in the local compiled C extension.
    """

    def __init__(
        self,
        model: Any,
        data: np.ndarray | pd.DataFrame | None = None,
        model_output: Literal["raw", "probability", "log_loss"] | str = "raw",
        feature_perturbation: Literal["auto", "interventional", "tree_path_dependent"] = "auto",
        feature_names: list[str] | None = None,
        approximate: bool = ...,  # Deprecated
        link: Any = None,  # Ignored
        linearize_link: Any = None,  # Ignored
    ) -> None: ...
    def __call__(
        self,
        X: Any,
        y: np.ndarray | pd.Series | None = None,
        interactions: bool = False,
        check_additivity: bool = True,
        approximate: bool = False,
    ) -> Explanation: ...
    def shap_values(
        self,
        X: Any,
        y: np.ndarray | pd.Series | None = None,
        tree_limit: int | None = None,
        approximate: bool = False,
        check_additivity: bool = True,
        from_call: bool = False,
    ) -> np.ndarray: ...
    def shap_interaction_values(
        self,
        X: Any,
        y: np.ndarray | pd.Series | None = None,
        tree_limit: int | None = None,
    ) -> np.ndarray: ...
    def assert_additivity(
        self,
        phi: np.ndarray | list[np.ndarray],
        model_output: np.ndarray,
    ) -> None: ...
    @staticmethod
    def supports_model_with_masker(model: Any, masker: Any) -> bool: ...

    # Instance attributes
    expected_value: float | np.ndarray | list[float] | None
    model: TreeEnsemble
    model_output: str
    data: np.ndarray | None
    data_missing: np.ndarray | None
    feature_perturbation: str
    data_feature_names: list[str] | None

class TreeEnsemble:
    """An ensemble of decision trees.

    This object provides a common interface to many different types of models.
    """

    def __init__(
        self,
        model: Any,
        data: np.ndarray | None = None,
        data_missing: np.ndarray | None = None,
        model_output: str | None = None,
    ) -> None: ...
    def predict(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        output: str | None = None,
        tree_limit: int | None = None,
    ) -> np.ndarray: ...
    def get_transform(self) -> str: ...

    # Instance attributes
    model_type: str
    trees: list[SingleTree] | None
    base_offset: float | np.ndarray
    model_output: str | None
    objective: str | None
    tree_output: str | None
    internal_dtype: type[np.floating[Any]]
    input_dtype: type[np.floating[Any]]
    data: np.ndarray | None
    data_missing: np.ndarray | None
    fully_defined_weighting: bool
    tree_limit: int | None
    num_stacked_models: int
    cat_feature_indices: list[int] | None
    original_model: Any
    num_outputs: int
    values: np.ndarray
    children_left: np.ndarray
    children_right: np.ndarray
    children_default: np.ndarray
    features: np.ndarray
    thresholds: np.ndarray
    node_sample_weight: np.ndarray
    max_depth: int
    num_nodes: np.ndarray

class SingleTree:
    """A single decision tree."""

    def __init__(
        self,
        tree: Any,
        normalize: bool = False,
        scaling: float = 1.0,
        data: np.ndarray | None = None,
        data_missing: np.ndarray | None = None,
    ) -> None: ...

    # Instance attributes
    children_left: np.ndarray
    children_right: np.ndarray
    children_default: np.ndarray
    features: np.ndarray
    thresholds: np.ndarray
    values: np.ndarray
    node_sample_weight: np.ndarray
    max_depth: int

class IsoTree(SingleTree):
    """An isolation tree (for isolation forests)."""

    def __init__(
        self,
        tree: Any,
        feature_names: list[int],
        scaling: float = 1.0,
        data: np.ndarray | None = None,
        data_missing: np.ndarray | None = None,
    ) -> None: ...
