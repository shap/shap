"""
Utilities for ranking feature interactions using SHAP values.
"""

import numpy as np
from ._general import approximate_interactions


def rank_interactions(shap_values, X, feature_names=None, max_pairs=None):
    """
    Rank pairwise feature interactions using SHAP's approximate_interactions.

    Parameters
    ----------
    shap_values : array-like of shape (n_samples, n_features)
        SHAP values.
    X : array-like or DataFrame
        Input feature matrix.
    feature_names : list of str, optional
        Names of features.
    max_pairs : int, optional
        Number of top interaction pairs to return.

    Returns
    -------
    list of tuples
        Sorted list of (feature_i, feature_j, score).
    """

    shap_values = np.asarray(shap_values)
    n_features = shap_values.shape[1]

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]

    interaction_scores = {}

    for i in range(n_features):
        interactions = approximate_interactions(i, shap_values, X)

        for j, score in enumerate(interactions):
            if i == j:
                continue

            pair = tuple(sorted((i, j)))

            interaction_scores[pair] = interaction_scores.get(pair, 0.0) + abs(score)

    ranked_pairs = [
        (feature_names[i], feature_names[j], score)
        for (i, j), score in interaction_scores.items()
    ]

    ranked_pairs.sort(key=lambda x: x[2], reverse=True)

    if max_pairs is not None:
        ranked_pairs = ranked_pairs[:max_pairs]

    return ranked_pairs