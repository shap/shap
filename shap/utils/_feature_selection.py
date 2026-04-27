"""SHAP-based feature selection utilities.

This module provides functions for ranking and selecting features based on
SHAP values. It leverages the per-feature importance computed by any SHAP
explainer to guide feature selection, which is significantly more efficient
than brute-force search over all possible subsets.

Typical usage:

    >>> import shap
    >>> explainer = shap.TreeExplainer(model)
    >>> shap_values = explainer(X)
    >>> # Rank features by importance
    >>> ranking = shap.utils.rank_features(shap_values)
    >>> # Select optimal feature subset
    >>> result = shap.utils.select_features(
    ...     shap_values, model, X, y, method="backward"
    ... )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

    from .._explanation import Explanation


def _aggregate_shap_values(
    shap_values: Explanation,
    method: str = "mean_abs",
) -> npt.NDArray[np.floating[Any]]:
    """Aggregate SHAP values across samples to get per-feature importance.

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values from any explainer.  Must have a ``.values`` attribute
        with shape ``(n_samples, n_features)`` or
        ``(n_samples, n_features, n_outputs)``.
    method : str
        Aggregation strategy:

        - ``"mean_abs"`` (default): mean of absolute SHAP values.
        - ``"mean"``: mean of raw SHAP values (preserves sign).
        - ``"max_abs"``: maximum absolute SHAP value per feature.

    Returns
    -------
    importance : np.ndarray, shape (n_features,)
        Per-feature importance score.

    Raises
    ------
    ValueError
        If the SHAP values have fewer than 2 dimensions or *method* is
        unrecognised.
    """
    values = np.asarray(shap_values.values)

    # For multi-output models, aggregate across outputs first
    if values.ndim == 3:
        # (n_samples, n_features, n_outputs) -> (n_samples, n_features)
        values = np.abs(values).mean(axis=2)

    if values.ndim < 2:
        raise ValueError(
            f"SHAP values must have at least 2 dimensions (n_samples, n_features), got shape {values.shape}."
        )

    if method == "mean_abs":
        return np.abs(values).mean(axis=0)
    elif method == "mean":
        return values.mean(axis=0)
    elif method == "max_abs":
        return np.abs(values).max(axis=0)
    else:
        raise ValueError(f"Unknown aggregation method '{method}'. Choose from 'mean_abs', 'mean', or 'max_abs'.")


def rank_features(
    shap_values: Explanation,
    method: str = "mean_abs",
    feature_names: list[str] | npt.NDArray[Any] | None = None,
) -> dict[str, Any]:
    """Rank features by their SHAP importance.

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values from any explainer.
    method : str
        Aggregation method for computing importance scores.
        One of ``"mean_abs"`` (default), ``"mean"``, or ``"max_abs"``.
        See :func:`_aggregate_shap_values` for details.
    feature_names : list of str or None
        Optional feature names.  If *None*, the names are taken from
        ``shap_values.feature_names`` or default to ``"Feature 0"``,
        ``"Feature 1"``, etc.

    Returns
    -------
    result : dict
        Dictionary with:

        - ``"ranked_indices"``: ``np.ndarray`` of feature indices sorted by
          importance (most important first).
        - ``"importance_scores"``: ``np.ndarray`` of corresponding scores.
        - ``"feature_names"``: list of feature names in ranked order.

    Examples
    --------
    >>> ranking = shap.utils.rank_features(shap_values)
    >>> print(ranking["feature_names"][:5])  # top 5 features
    """
    importance = _aggregate_shap_values(shap_values, method=method)
    ranked_indices = np.argsort(-np.abs(importance))

    # Resolve feature names
    if feature_names is None:
        fn = shap_values.feature_names
        if fn is not None and hasattr(fn, "__len__") and len(fn) == len(importance):
            feature_names = list(fn)
        else:
            feature_names = [f"Feature {i}" for i in range(len(importance))]

    ranked_names = [feature_names[i] for i in ranked_indices]

    return {
        "ranked_indices": ranked_indices,
        "importance_scores": importance[ranked_indices],
        "feature_names": ranked_names,
    }


def select_features(
    shap_values: Explanation,
    model: Any,
    X: npt.NDArray[Any],
    y: npt.NDArray[Any],
    *,
    method: str = "backward",
    scoring: str | Callable[..., float] = "accuracy",
    cv: int = 5,
    min_features: int = 1,
    aggregation: str = "mean_abs",
    random_state: int = 0,
) -> dict[str, Any]:
    """Select the optimal feature subset using SHAP-guided search.

    Uses SHAP importance to order a greedy forward or backward feature
    selection, evaluating each candidate subset with cross-validation.
    Because SHAP provides a strong prior on feature ordering, this is
    dramatically faster than exhaustive search (O(n) evaluations instead
    of O(2^n)).

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values for the training data.
    model : estimator
        A scikit-learn compatible estimator that implements ``fit`` and
        ``predict``.  The estimator is cloned for each evaluation.
    X : np.ndarray, shape (n_samples, n_features)
        Training data.
    y : np.ndarray, shape (n_samples,)
        Target values.
    method : ``"backward"`` or ``"forward"``
        Selection strategy:

        - ``"backward"``: start with all features, remove the least
          important one at a time.
        - ``"forward"``: start with zero features, add the most important
          one at a time.
    scoring : str or callable
        Scoring metric.  If a string, must be a valid
        :func:`sklearn.model_selection.cross_val_score` ``scoring`` argument
        (e.g. ``"accuracy"``, ``"f1"``, ``"roc_auc"``).
    cv : int
        Number of cross-validation folds.
    min_features : int
        Minimum number of features to retain (only used in backward
        elimination).
    aggregation : str
        Method for aggregating SHAP values.  See :func:`rank_features`.
    random_state : int
        Random seed for cross-validation.

    Returns
    -------
    result : dict
        Dictionary with:

        - ``"selected_indices"``: ``np.ndarray`` of selected feature indices.
        - ``"selected_names"``: list of selected feature names.
        - ``"scores"``: list of ``(n_features_used, mean_cv_score)`` tuples
          for each step of the search.
        - ``"best_score"``: the best mean CV score achieved.

    Raises
    ------
    ValueError
        If *method* is not ``"forward"`` or ``"backward"``, or if
        *min_features* is invalid.

    Examples
    --------
    >>> result = shap.utils.select_features(
    ...     shap_values, model, X, y, method="backward"
    ... )
    >>> print(f"Best score: {result['best_score']:.3f}")
    >>> print(f"Selected features: {result['selected_names']}")
    """
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_score

    if method not in ("forward", "backward"):
        raise ValueError(f"Unknown selection method '{method}'. Choose from 'forward' or 'backward'.")

    n_features = X.shape[1]
    if min_features < 1 or min_features > n_features:
        raise ValueError(f"min_features must be between 1 and {n_features}, got {min_features}.")

    ranking = rank_features(shap_values, method=aggregation)
    ranked_indices = ranking["ranked_indices"]

    scores: list[tuple[int, float]] = []

    if method == "backward":
        # Start with all features, remove least important first
        current_indices = list(ranked_indices)
        best_score = -np.inf
        best_indices = list(current_indices)

        while len(current_indices) >= min_features:
            cv_scores = cross_val_score(
                clone(model),
                X[:, current_indices],
                y,
                cv=cv,
                scoring=scoring if isinstance(scoring, str) else scoring,
            )
            mean_score = float(cv_scores.mean())
            scores.append((len(current_indices), mean_score))

            if mean_score >= best_score:
                best_score = mean_score
                best_indices = list(current_indices)

            if len(current_indices) <= min_features:
                break

            # Remove the least important feature (last in the ranked order)
            current_indices.pop()

    else:  # forward
        current_indices: list[int] = []  # type: ignore[no-redef]
        best_score = -np.inf
        best_indices = []

        for idx in ranked_indices:
            current_indices.append(int(idx))

            cv_scores = cross_val_score(
                clone(model),
                X[:, current_indices],
                y,
                cv=cv,
                scoring=scoring if isinstance(scoring, str) else scoring,
            )
            mean_score = float(cv_scores.mean())
            scores.append((len(current_indices), mean_score))

            if mean_score >= best_score:
                best_score = mean_score
                best_indices = list(current_indices)

    # Resolve selected feature names
    all_feature_names = ranking["feature_names"]
    # Build an index-to-name mapping from the full ranking
    idx_to_name = {int(ranked_indices[i]): all_feature_names[i] for i in range(len(ranked_indices))}
    selected_names = [idx_to_name[i] for i in best_indices]

    return {
        "selected_indices": np.array(best_indices),
        "selected_names": selected_names,
        "scores": scores,
        "best_score": best_score,
    }
