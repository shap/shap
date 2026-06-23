from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

from .._explanation import Explanation
from ..utils import rank_interactions
from . import colors


def _prepare_interaction_ranking_inputs(
    shap_values: Explanation,
    max_display: int,
) -> tuple[Explanation, list[tuple[int, int, float]], np.ndarray]:
    """Validate inputs and return reduced explanation and ranking outputs."""
    if not isinstance(shap_values, Explanation):
        raise TypeError("interaction plot requires an `Explanation` object.")

    values = np.asarray(shap_values.values)
    data = shap_values.data
    if data is None:
        raise ValueError("interaction plot requires shap_values.data to be present.")

    data = np.asarray(data)
    if values.ndim != 2:
        raise ValueError("interaction plot expects 2D SHAP values with shape (n_samples, n_features).")
    if data.ndim != 2:
        raise ValueError("interaction plot expects 2D feature data with shape (n_samples, n_features).")
    if values.shape != data.shape:
        raise ValueError("shap_values.values and shap_values.data must have matching shapes.")

    feature_names = shap_values.feature_names
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(values.shape[1])]

    feature_names_arr = np.array(feature_names)
    importance_order = np.argsort(-np.abs(values).mean(0))
    display_inds = importance_order[: max_display if max_display > 0 else values.shape[1]]

    sub_explanation = Explanation(
        values=values[:, display_inds],
        data=data[:, display_inds],
        feature_names=feature_names_arr[display_inds].tolist(),
    )

    pairs, interaction_matrix = rank_interactions(sub_explanation, return_matrix=True)
    return sub_explanation, pairs, interaction_matrix


def interaction_heatmap(
    shap_values: Explanation,
    max_display: int = 10,
    cmap=colors.red_white_blue,
    show: bool = True,
    ax: plt.Axes | None = None,
):
    """Create a heatmap of pairwise interaction strengths.

    Parameters
    ----------
    shap_values : shap.Explanation
        A 2D :class:`.Explanation` object containing SHAP values and the
        corresponding feature data.

    max_display : int
        Maximum number of features to display based on mean absolute SHAP value.

    cmap : matplotlib colormap
        Colormap used for the heatmap.

    show : bool
        Whether to call :external+mpl:func:`matplotlib.pyplot.show()`.

    ax : matplotlib Axes, optional
        Existing Axes object to draw into.

    Returns
    -------
    ax : matplotlib Axes
        Axes containing the plot.
    """
    sub_explanation, _, interaction_matrix = _prepare_interaction_ranking_inputs(shap_values, max_display)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    vmax = np.nanmax(np.abs(interaction_matrix))
    if vmax == 0:
        vmax = 1.0

    im = ax.imshow(interaction_matrix, cmap=cmap, vmin=-vmax, vmax=vmax)
    labels = np.array(sub_explanation.feature_names)
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_title("Pairwise interaction strength")

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Interaction strength")

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def interaction_beeswarm(
    shap_values: Explanation,
    max_display: int = 10,
    max_pairs: int = 20,
    cmap=colors.red_white_blue,
    show: bool = True,
    ax: plt.Axes | None = None,
):
    """Create a beeswarm-style ranking view of pairwise interactions.

    Parameters
    ----------
    shap_values : shap.Explanation
        A 2D :class:`.Explanation` object containing SHAP values and feature data.

    max_display : int
        Maximum number of features considered before ranking interactions.

    max_pairs : int
        Maximum number of ranked interaction pairs shown.

    cmap : matplotlib colormap
        Colormap used for the points based on interaction score.

    show : bool
        Whether to call :external+mpl:func:`matplotlib.pyplot.show()`.

    ax : matplotlib Axes, optional
        Existing Axes object to draw into.

    Returns
    -------
    ax : matplotlib Axes
        Axes containing the plot.
    """
    sub_explanation, pairs, _ = _prepare_interaction_ranking_inputs(shap_values, max_display)

    if max_pairs <= 0:
        raise ValueError("max_pairs must be a positive integer.")

    pairs = pairs[:max_pairs]
    if len(pairs) == 0:
        raise ValueError("No interaction pairs available to plot.")

    labels = np.array(sub_explanation.feature_names)
    pair_labels = [f"{labels[i]} x {labels[j]}" for i, j, _ in pairs]
    scores = np.array([score for _, _, score in pairs], dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(pairs) + 1.5)))

    norm = mcolors.Normalize(vmin=float(np.min(scores)), vmax=float(np.max(scores)) or 1.0)
    y = np.arange(len(pairs))
    x_jitter = np.linspace(-0.08, 0.08, len(pairs)) if len(pairs) > 1 else np.array([0.0])

    ax.hlines(y, xmin=0.0, xmax=scores, color="#999999", linewidth=0.8, alpha=0.5)
    ax.scatter(scores, y + x_jitter, c=scores, cmap=cmap, norm=norm, s=38, alpha=0.95, edgecolors="none")
    ax.set_yticks(y, labels=pair_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Interaction strength")
    ax.set_title("Top pairwise interactions")
    ax.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.7)

    if show:
        plt.tight_layout()
        plt.show()

    return ax
