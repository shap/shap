from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from . import colors

if TYPE_CHECKING:
    from collections.abc import Sequence


def group_difference(
    shap_values: np.ndarray,
    group_mask: np.ndarray,
    feature_names: Sequence[str] | None = None,
    xlabel: str | None = None,
    xmin: float | None = None,
    xmax: float | None = None,
    max_display: int | None = None,
    sort: bool = True,
    show: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """This plots the difference in mean SHAP values between two groups.

    It is useful to decompose many group level metrics about the model output among the
    input features. Quantitative fairness metrics for machine learning models are
    a common example of such group level metrics.

    Parameters
    ----------
    shap_values : numpy.array
        Matrix of SHAP values (# samples x # features) or a vector of model outputs (# samples).

    group_mask : numpy.array
        A boolean mask where True represents the first group of samples and False the second.

    feature_names : list
        A list of feature names.

    show : bool
        Whether :external+mpl:func:`matplotlib.pyplot.show()` is called before returning.
        Setting this to ``False`` allows the plot to be customized further after it has
        been created.

    ax : matplotlib Axes, optional
        Axes object to draw the plot onto, otherwise a new Axes is created.

    Returns
    -------
    ax : matplotlib Axes
        Returns the :external+mpl:class:`~matplotlib.axes.Axes` object with the plot drawn onto it.

    """
    # See if we were passed a single model output vector and not a matrix of SHAP values
    if len(shap_values.shape) == 1:
        shap_values = shap_values.reshape(1, -1).T
        if feature_names is None:
            feature_names = [""]

    # Fill in any missing feature names
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(shap_values.shape[1])]

    # Compute confidence bounds for the group difference value
    vs = []
    gmean = group_mask.mean()
    for _ in range(200):
        r = np.random.rand(shap_values.shape[0]) > gmean
        vs.append(shap_values[r].mean(0) - shap_values[~r].mean(0))
    vs_ = np.array(vs)
    xerr = np.vstack([np.percentile(vs_, 95, axis=0), np.percentile(vs_, 5, axis=0)])

    diff = shap_values[group_mask].mean(0) - shap_values[~group_mask].mean(0)

    if sort is True:
        inds = np.argsort(-np.abs(diff)).astype(int)
    else:
        inds = np.arange(len(diff))

    if max_display is not None:
        inds = inds[:max_display]
    if ax is None:
        # Draw the figure if no ax has been provided
        figsize = (6.4, 0.2 + 0.9 * len(inds))
        _, ax = plt.subplots(figsize=figsize)
    ticks = range(len(inds) - 1, -1, -1)
    diff_values = np.asarray(diff[inds], dtype=float)
    xerr_values = np.asarray(np.abs(xerr[:, inds]), dtype=float)
    ax.axvline(0, color="#999999", linewidth=0.5)
    ax.barh(ticks, diff_values, color=colors.blue_rgb, capsize=3, xerr=xerr_values)

    for i in range(len(inds)):
        ax.axhline(y=i, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("none")
    ax.set_yticks(ticks)
    ax.set_yticklabels([feature_names[i] for i in inds], fontsize=13)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(labelsize=11)
    if xlabel is None:
        xlabel = "Group SHAP value difference"
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_xlim(xmin, xmax)
    if show:
        plt.show()
    return ax
