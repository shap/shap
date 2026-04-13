from __future__ import annotations

import warnings
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .. import Explanation
from . import colors


def group_difference(
    shap_values: Explanation,
    group_mask,
    feature_names: Sequence[str] | None = None,
    xlabel: str | None = None,
    xmin: float | None = None,
    xmax: float | None = None,
    max_display: int | None = None,
    sort: bool = True,
    show: bool = True,
    ax: plt.Axes | None = None,
):
    """Plot the difference in mean SHAP values between two groups.

    It is useful to decompose many group-level metrics about the model output among the
    input features. Quantitative fairness metrics for machine learning models are a
    common example of such group-level metrics.

    Parameters
    ----------
    shap_values : shap.Explanation
        A two-dimensional :class:`.Explanation` object containing SHAP values with shape
        ``(# samples, # features)``, or a one-dimensional Explanation of model outputs
        with shape ``(# samples,)``.

    group_mask : array-like
        A boolean mask where True represents the first group of samples and False the second.

    feature_names : sequence of strings, optional
        Feature names for the y-axis labels.

    ax : matplotlib Axes, optional
        Optionally specify an existing :external+mpl:class:`matplotlib.axes.Axes` object,
        into which the plot will be placed.

    show : bool
        Whether :external+mpl:func:`matplotlib.pyplot.show()` is called before returning.
        Setting this to ``False`` allows the plot to be customized further after it has
        been created.

    Returns
    -------
    ax : matplotlib Axes
        Returns the :external+mpl:class:`~matplotlib.axes.Axes` object with the plot drawn
        onto it. Only returned if ``show=False``.

    """
    if not isinstance(shap_values, Explanation):
        raise TypeError("The shap_values parameter must be a shap.Explanation object!")

    group_mask_arr = np.asarray(group_mask)
    if group_mask_arr.ndim != 1:
        raise ValueError("group_mask must be a one-dimensional boolean mask.")
    if group_mask_arr.dtype != bool:
        unique_vals = set(np.unique(group_mask_arr).tolist())
        if unique_vals.issubset({0, 1}):
            group_mask_arr = group_mask_arr.astype(bool)
        else:
            raise ValueError("group_mask must be a boolean mask (or contain only 0/1 values).")

    # unpack Explanation values
    if len(shap_values.shape) == 1:
        shap_values_arr = np.asarray(shap_values.values).reshape(-1, 1)
        if feature_names is None:
            feature_names = [""]
    elif len(shap_values.shape) == 2:
        shap_values_arr = np.asarray(shap_values.values)
    else:
        raise ValueError("group_difference expects a 1D or 2D Explanation.")

    if shap_values_arr.shape[0] != group_mask_arr.shape[0]:
        raise ValueError(
            "group_mask must have the same length as the number of rows in shap_values. "
            f"Got {group_mask_arr.shape[0]} vs {shap_values_arr.shape[0]}."
        )

    # Fill in any missing feature names
    if feature_names is None:
        if shap_values.feature_names is not None and len(shap_values.shape) == 2:
            feature_names = list(shap_values.feature_names)  # type: ignore[arg-type]
        else:
            feature_names = [f"Feature {i}" for i in range(shap_values_arr.shape[1])]

    # Compute confidence bounds for the group difference value
    vs: list[np.ndarray] = []
    gmean = float(group_mask_arr.mean())
    for _ in range(200):
        r = np.random.rand(shap_values_arr.shape[0]) > gmean
        vs.append(shap_values_arr[r].mean(0) - shap_values_arr[~r].mean(0))
    vs_ = np.array(vs)
    xerr = np.vstack([np.percentile(vs_, 95, axis=0), np.percentile(vs_, 5, axis=0)])

    diff = shap_values_arr[group_mask_arr].mean(0) - shap_values_arr[~group_mask_arr].mean(0)

    if sort is True:
        inds = np.argsort(-np.abs(diff)).astype(int)
    else:
        inds = np.arange(len(diff))

    if max_display is not None:
        inds = inds[:max_display]

    if ax is None:
        ax = plt.gca()
        fig = plt.gcf()
        # Only modify the figure size if ax was not passed in
        fig.set_size_inches(6.4, 0.2 + 0.9 * len(inds))

    ticks = range(len(inds) - 1, -1, -1)
    ax.axvline(0, color="#999999", linewidth=0.5)
    ax.barh(ticks, diff[inds], color=colors.blue_rgb, capsize=3, xerr=np.abs(xerr[:, inds]))

    for i in range(len(inds)):
        ax.axhline(y=i, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("none")
    ax.set_yticks(ticks)
    ax.set_yticklabels([list(feature_names)[i] for i in inds], fontsize=13)
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
        return
    return ax


def group_difference_legacy(
    shap_values,
    group_mask,
    feature_names: Sequence[str] | None = None,
    xlabel: str | None = None,
    xmin: float | None = None,
    xmax: float | None = None,
    max_display: int | None = None,
    sort: bool = True,
    show: bool = True,
    ax: plt.Axes | None = None,
):
    """Legacy numpy-based group_difference plot.

    This function will be removed in a future version. Use :func:`shap.plots.group_difference`.
    """
    warnings.warn(
        "The behaviour of this function will change in a future version to the new plotting API."
        "\nUse `shap.plots.group_difference` to opt-in to the new behaviour and silence this warning."
        "\nFor more information on using the new API, see:\n"
        "https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/migrating-to-new-api.html",
        DeprecationWarning,
    )
    # Preserve legacy behaviour: if an ax is provided, avoid calling plt.show() implicitly.
    if ax is not None:
        show = False

    exp = Explanation(values=np.asarray(shap_values), feature_names=None if feature_names is None else list(feature_names))
    _ = group_difference(
        exp,
        group_mask,
        feature_names=feature_names,
        xlabel=xlabel,
        xmin=xmin,
        xmax=xmax,
        max_display=max_display,
        sort=sort,
        show=show,
        ax=ax,
    )
    return
