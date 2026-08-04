from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from . import colors
from ._labels import labels


def truncate_text(text: str, max_len: int) -> str:
    if len(text) > max_len:
        return text[: int(max_len / 2) - 2] + "..." + text[-int(max_len / 2) + 1 :]
    else:
        return text


def monitoring(
    ind: int,
    shap_values: np.ndarray,
    features: np.ndarray | pd.DataFrame | None = None,
    feature_names: list[str] | None = None,
    show: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Create a SHAP monitoring plot.

    (Note this function is preliminary and subject to change!!)
    A SHAP monitoring plot is meant to display the behavior of a model
    over time. Often the shap_values given to this plot explain the loss
    of a model, so changes in a feature's impact on the model's loss over
    time can help in monitoring the model's performance.

    Parameters
    ----------
    ind : int
        Index of the feature to plot.

    shap_values : numpy.ndarray
        Matrix of SHAP values (# samples x # features). Can also be a
        :class:`shap.Explanation` object.

    features : numpy.ndarray or pandas.DataFrame, optional
        Matrix of feature values (# samples x # features). If ``shap_values`` is an
        Explanation object, this will be extracted automatically.

    feature_names : list of str, optional
        Names of the features (length # features).

    show : bool
        Whether :external+mpl:func:`matplotlib.pyplot.show()` is called before returning.
        Setting this to ``False`` allows the plot to be customized further after
        it has been created. Defaults to ``True``.

    ax : matplotlib Axes, optional
        Optionally specify an existing :external+mpl:class:`matplotlib.axes.Axes`
        object to draw into. When ``None``, uses the current axes.

    Returns
    -------
    ax : matplotlib Axes
        Returns the :external+mpl:class:`~matplotlib.axes.Axes` object with the
        plot drawn onto it.
    """
    # Extract data from Explanation object if provided
    if hasattr(shap_values, "values"):
        if features is None:
            features = getattr(shap_values, "data", None)
        if feature_names is None:
            feature_names = getattr(shap_values, "feature_names", None)
        shap_values = shap_values.values

    if features is None:
        raise ValueError("A features array must be provided!")

    if isinstance(features, pd.DataFrame):
        if feature_names is None:
            feature_names = features.columns.tolist()
        features = features.values

    num_features = shap_values.shape[1]

    if feature_names is None:
        feature_names = [labels["FEATURE"] % str(i) for i in range(num_features)]

    # get (or create) the axes to draw on
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))
    fig = ax.get_figure()

    ys = shap_values[:, ind]
    xs = np.arange(len(ys))  # np.linspace(0, 12*2, len(ys))

    pvals = []
    inc = 50
    for i in range(inc, len(ys) - inc, inc):
        # stat, pval = scipy.stats.mannwhitneyu(v[:i], v[i:], alternative="two-sided")
        _, pval = scipy.stats.ttest_ind(ys[:i], ys[i:])
        pvals.append(pval)

    if len(pvals) > 0:
        min_pval = np.min(pvals)
        min_pval_ind = float(np.argmin(pvals) * inc + inc)

        if min_pval < 0.05 / shap_values.shape[1]:
            ax.axvline(min_pval_ind, linestyle="dashed", color="#666666", alpha=0.2)

    cvals = features[:, ind]
    p = ax.scatter(xs, ys, s=10, c=cvals, cmap=colors.red_blue)

    ax.set_xlabel("Sample index")
    ax.set_ylabel(truncate_text(feature_names[ind], 30) + "\nSHAP value", size=13)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    cb = plt.colorbar(p, ax=ax)
    cb.outline.set_visible(False)  # type: ignore
    bbox = cb.ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    cb.ax.set_aspect((bbox.height - 0.7) * 20)
    cb.set_label(truncate_text(feature_names[ind], 30), size=13)

    if show:
        plt.show()
    return ax
