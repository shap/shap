from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .. import Cohorts, Explanation
from ..utils._exceptions import DimensionError
from ._style import get_style


def _coerce_cohorts(reference: Any, comparison: Any | None) -> tuple[list[str], list[Explanation | np.ndarray]]:
    if comparison is not None:
        return ["reference", "comparison"], [reference, comparison]

    if isinstance(reference, Cohorts):
        cohorts = reference.cohorts
    elif isinstance(reference, dict):
        cohorts = reference
    elif isinstance(reference, (list, tuple)) and len(reference) == 2:
        cohorts = {"reference": reference[0], "comparison": reference[1]}
    else:
        raise TypeError(
            "cohort_difference expects two Explanation-like objects, or a Cohorts object / dictionary with two entries!"
        )

    if len(cohorts) != 2:
        raise ValueError("cohort_difference currently requires exactly two cohorts to compare.")

    names = list(cohorts.keys())
    values = list(cohorts.values())
    return names, values


def _as_2d_values(item: Explanation | np.ndarray) -> tuple[np.ndarray, list[str] | np.ndarray | None]:
    if isinstance(item, Explanation):
        values = np.asarray(item.values)
        feature_names = item.feature_names
    else:
        values = np.asarray(item)
        feature_names = None

    if values.ndim == 1:
        values = values.reshape(1, -1)

    if values.ndim != 2:
        raise ValueError("cohort_difference only supports two-dimensional SHAP value arrays.")

    return values, feature_names


def cohort_difference(
    reference,
    comparison=None,
    feature_names=None,
    max_display: int = 10,
    sort: bool = True,
    n_bootstrap: int = 200,
    confidence: float = 0.95,
    random_state: int | None = None,
    xlabel: str | None = None,
    ax=None,
    show: bool = True,
):
    """Compare the mean SHAP values between two cohorts.

    The plot ranks features by the magnitude of the cohort difference and adds
    bootstrap confidence intervals for the signed mean difference.

    Parameters
    ----------
    reference, comparison : shap.Explanation, numpy.array, shap.Cohorts, dict, or tuple
        Two explanation cohorts to compare. ``reference`` may also be a
        :class:`.Cohorts` object, a dictionary of exactly two explanations, or a
        two-item tuple/list when ``comparison`` is omitted.
    feature_names : list or None
        Names for the plotted features. When omitted, names are inferred from the
        first Explanation object or generated as ``Feature i``.
    max_display : int
        Maximum number of features to show.
    sort : bool
        If ``True``, show the features with the largest absolute cohort shift first.
    n_bootstrap : int
        Number of bootstrap resamples used to estimate confidence intervals.
    confidence : float
        Confidence level for the bootstrap interval.
    random_state : int or None
        Random seed for bootstrap sampling.
    xlabel : str or None
        Label for the x-axis. When omitted, a default cohort comparison label is used.
    ax : matplotlib Axes or None
        Axes object to draw the plot onto, otherwise uses the current Axes.
    show : bool
        Whether :func:`matplotlib.pyplot.show` is called before returning.

    Returns
    -------
    ax : matplotlib Axes
        The Axes object with the plot drawn onto it.
    """
    style = get_style()
    cohort_names, cohort_items = _coerce_cohorts(reference, comparison)
    reference_values, reference_feature_names = _as_2d_values(cohort_items[0])
    comparison_values, comparison_feature_names = _as_2d_values(cohort_items[1])

    if reference_values.shape[1] != comparison_values.shape[1]:
        raise DimensionError("The cohorts to compare must have the same number of feature columns!")

    if feature_names is None:
        feature_names = reference_feature_names if reference_feature_names is not None else comparison_feature_names
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(reference_values.shape[1])]

    feature_names = list(feature_names)
    if len(feature_names) != reference_values.shape[1]:
        raise DimensionError("feature_names must have the same length as the number of feature columns!")

    diff = comparison_values.mean(0) - reference_values.mean(0)

    rng = np.random.default_rng(random_state)
    bootstrap_diffs = []
    lower_q = (1.0 - confidence) / 2.0 * 100.0
    upper_q = (1.0 + confidence) / 2.0 * 100.0

    if n_bootstrap > 0:
        for _ in range(n_bootstrap):
            ref_idx = rng.integers(0, reference_values.shape[0], reference_values.shape[0])
            cmp_idx = rng.integers(0, comparison_values.shape[0], comparison_values.shape[0])
            bootstrap_diffs.append(comparison_values[cmp_idx].mean(0) - reference_values[ref_idx].mean(0))
        bootstrap_diffs = np.asarray(bootstrap_diffs)
        lower = np.percentile(bootstrap_diffs, lower_q, axis=0)
        upper = np.percentile(bootstrap_diffs, upper_q, axis=0)
    else:
        lower = diff
        upper = diff

    reference_std = reference_values.std(0, ddof=1)
    comparison_std = comparison_values.std(0, ddof=1)
    pooled_std = np.sqrt((reference_std**2 + comparison_std**2) / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        effect_size = np.divide(diff, pooled_std, out=np.zeros_like(diff), where=pooled_std > 0)

    if sort:
        order = np.argsort(-np.abs(effect_size if np.any(np.isfinite(effect_size)) else diff))
    else:
        order = np.arange(len(diff))

    order = order[: min(max_display, len(order))]
    y_positions = np.arange(len(order), 0, -1)

    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 0.45 * len(order) + 1.7))

    ax.axvline(0, color="#999999", linewidth=0.8, zorder=0)

    xerr = np.vstack([diff[order] - lower[order], upper[order] - diff[order]])
    bar_colors = [style.primary_color_positive if diff[idx] >= 0 else style.primary_color_negative for idx in order]
    ax.barh(y_positions, diff[order], xerr=xerr, color=bar_colors, capsize=3, height=0.72)

    for row, idx in enumerate(order):
        ax.axhline(y=y_positions[row], color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
        value = diff[idx]
        if value >= 0:
            offset = 0.01 * max(1.0, np.nanmax(np.abs(diff)))
            ax.text(
                value + offset,
                y_positions[row],
                f"{value:+0.02f}",
                ha="left",
                va="center",
                color=style.primary_color_positive,
                fontsize=12,
            )
        else:
            offset = 0.01 * max(1.0, np.nanmax(np.abs(diff)))
            ax.text(
                value - offset,
                y_positions[row],
                f"{value:+0.02f}",
                ha="right",
                va="center",
                color=style.primary_color_negative,
                fontsize=12,
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([feature_names[idx] for idx in order], fontsize=13)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("none")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(labelsize=11)

    if xlabel is None:
        xlabel = f"Mean SHAP value difference ({cohort_names[1]} - {cohort_names[0]})"
    ax.set_xlabel(xlabel, fontsize=13)

    title = f"{cohort_names[1]} vs {cohort_names[0]}"
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlim(None, None)

    if len(order) > 0:
        max_abs = np.nanmax(np.abs(np.concatenate([lower[order], upper[order], diff[order]])))
        if np.isfinite(max_abs) and max_abs > 0:
            ax.set_xlim(-1.15 * max_abs, 1.15 * max_abs)

    if show:
        plt.show()

    return ax
