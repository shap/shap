# Type stubs for shap._cext_gpu
"""GPU accelerated C extension for Tree SHAP computations."""

from __future__ import annotations

import numpy as np

def dense_tree_shap(
    children_left: np.ndarray,
    children_right: np.ndarray,
    children_default: np.ndarray,
    features: np.ndarray,
    thresholds: np.ndarray,
    values: np.ndarray,
    node_sample_weight: np.ndarray,
    max_depth: int,
    X: np.ndarray,
    X_missing: np.ndarray,
    y: np.ndarray | None,
    data: np.ndarray | None,
    data_missing: np.ndarray | None,
    tree_limit: int,
    base_offset: np.ndarray,
    phi: np.ndarray,
    feature_dependence: int,
    model_output: int,
    interactions: bool,
) -> None:
    """GPU accelerated implementation of Tree SHAP for dense data.

    This function modifies the `phi` array in-place with the computed SHAP values.

    Parameters
    ----------
    children_left : np.ndarray
        Array of left child indices for each node.
    children_right : np.ndarray
        Array of right child indices for each node.
    children_default : np.ndarray
        Array of default child indices for each node (for missing values).
    features : np.ndarray
        Array of feature indices used for splitting at each node.
    thresholds : np.ndarray
        Array of threshold values for splitting at each node.
    values : np.ndarray
        Array of prediction values at each node.
    node_sample_weight : np.ndarray
        Array of sample weights at each node.
    max_depth : int
        Maximum depth of the trees.
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    X_missing : np.ndarray
        Boolean mask indicating missing values in X.
    y : np.ndarray or None
        Target values for each sample (used for loss explanation).
    data : np.ndarray or None
        Background data for interventional feature perturbation.
    data_missing : np.ndarray or None
        Boolean mask indicating missing values in background data.
    tree_limit : int
        Maximum number of trees to use (-1 for no limit).
    base_offset : np.ndarray
        Base offset values for the model.
    phi : np.ndarray
        Output array for SHAP values (modified in-place).
    feature_dependence : int
        Feature dependence assumption (0=independent, 1=tree_path_dependent).
    model_output : int
        Model output transformation (0=identity, 1=logistic, etc.).
    interactions : bool
        Whether to compute interaction effects.
    """
    ...
