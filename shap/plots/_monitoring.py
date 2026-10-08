import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from . import colors
from ._labels import labels


def truncate_text(text, max_len):
    if len(text) > max_len:
        return text[: int(max_len / 2) - 2] + "..." + text[-int(max_len / 2) + 1 :]
    else:
        return text


def monitoring(ind, shap_values, features, feature_names=None, show=True, ax=None):
    """Create a SHAP monitoring plot.

    Note: this function is preliminary and subject to change.

    A SHAP monitoring plot displays the behavior of a model over time. Often
    the shap_values given to this plot explain the loss of a model, so changes
    in a feature's impact on the model's loss over time can help in monitoring
    the model's performance.

    Parameters
    ----------
    ind : int
        Index of the feature to plot.

    shap_values : numpy.array
        Matrix of SHAP values (# samples x # features)

    features : numpy.array or pandas.DataFrame
        Matrix of feature values (# samples x # features)

    feature_names : list
        Names of the features (length # features)

    show : bool
        Whether to call ``plt.show()`` after rendering. Set to ``False`` when
        embedding the plot in a larger figure.

    ax : matplotlib.axes.Axes, optional
        Axes object to draw the plot on. If ``None`` (default), a new figure
        with ``figsize=(10, 3)`` is created.

    Returns
    -------
    matplotlib.axes.Axes or None
        The axes object when ``show=False``, otherwise ``None``.
    """
    if isinstance(features, pd.DataFrame):
        if feature_names is None:
            feature_names = features.columns
        features = features.values

    num_features = shap_values.shape[1]

    if feature_names is None:
        feature_names = np.array([labels["FEATURE"] % str(i) for i in range(num_features)])

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))
    else:
        show = False

    ys = shap_values[:, ind]
    xs = np.arange(len(ys))  # np.linspace(0, 12*2, len(ys))

    pvals = []
    inc = 50
    for i in range(inc, len(ys) - inc, inc):
        # stat, pval = scipy.stats.mannwhitneyu(v[:i], v[i:], alternative="two-sided")
        _, pval = scipy.stats.ttest_ind(ys[:i], ys[i:])
        pvals.append(pval)
    min_pval = np.min(pvals)
    min_pval_ind = float(np.argmin(pvals) * inc + inc)

    if min_pval < 0.05 / shap_values.shape[1]:
        ax.axvline(min_pval_ind, linestyle="dashed", color="#666666", alpha=0.2)

    ax.scatter(xs, ys, s=10, c=features[:, ind], cmap=colors.red_blue)

    ax.set_xlabel("Sample index")
    ax.set_ylabel(truncate_text(feature_names[ind], 30) + "\nSHAP value", size=13)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    cb = ax.get_figure().colorbar(ax.collections[0], ax=ax)
    cb.outline.set_visible(False)  # type: ignore
    bbox = cb.ax.get_window_extent().transformed(ax.get_figure().dpi_scale_trans.inverted())
    cb.ax.set_aspect((bbox.height - 0.7) * 20)
    cb.set_label(truncate_text(feature_names[ind], 30), size=13)
    if show:
        plt.show()
    else:
        return ax
