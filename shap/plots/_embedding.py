from __future__ import annotations

import matplotlib.pyplot as plt
import sklearn

from ..utils import convert_name
from . import colors
from ._labels import labels


def embedding(
    ind,
    shap_values,
    feature_names=None,
    method: str = "pca",
    alpha: float = 1.0,
    show: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Use the SHAP values as an embedding which we project to 2D for visualization.

    Parameters
    ----------
    ind : int or string
        If this is an int it is the index of the feature to use to color the embedding.
        If this is a string it is either the name of the feature, or it can have the
        form "rank(int)" to specify the feature with that rank (ordered by mean absolute
        SHAP value over all the samples), or "sum()" to mean the sum of all the SHAP values,
        which is the model's output (minus it's expected value).

    shap_values : numpy.array
        Matrix of SHAP values (# samples x # features).

    feature_names : None or list
        The names of the features in the shap_values array.

    method : "pca" or numpy.array
        How to reduce the dimensions of the shap_values to 2D. If "pca" then the 2D
        PCA projection of shap_values is used. If a numpy array then is should be
        (# samples x 2) and represent the embedding of that values.

    alpha : float
        The transparency of the data points (between 0 and 1). This can be useful to the
        show density of the data points when using a large dataset.

    show : bool
        Whether :external+mpl:func:`matplotlib.pyplot.show()` is called before returning.
        Setting this to ``False`` allows the plot to be customized further after it
        has been created.

    ax : matplotlib.axes.Axes, optional
        Axes object to draw the plot onto. If ``None``, uses the current axes via
        :func:`matplotlib.pyplot.gca`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the plot.

    """
    if ax is None:
        ax = plt.gca()

    if feature_names is None:
        feature_names = [labels["FEATURE"] % str(i) for i in range(shap_values.shape[1])]

    ind = convert_name(ind, shap_values, feature_names)
    if ind == "sum()":
        cvals = shap_values.sum(1)
        fname = "sum(SHAP values)"
    else:
        cvals = shap_values[:, ind]
        fname = feature_names[ind]

    # see if we need to compute the embedding
    if isinstance(method, str) and method == "pca":
        pca = sklearn.decomposition.PCA(2)
        embedding_values = pca.fit_transform(shap_values)
    elif hasattr(method, "shape") and method.shape[1] == 2:
        embedding_values = method
    else:
        print("Unsupported embedding method:", method)

    sc = ax.scatter(
        embedding_values[:, 0],
        embedding_values[:, 1],
        c=cvals,
        cmap=colors.red_blue,
        alpha=alpha,
        linewidth=0,
    )
    ax.axis("off")
    # ax.set_title(feature_names[ind])
    cb = ax.figure.colorbar(sc, ax=ax)
    cb.set_label("SHAP value for\n" + fname, size=13)
    cb.outline.set_visible(False)  # type: ignore

    ax.figure.set_size_inches(7.5, 5)
    bbox = cb.ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    cb.ax.set_aspect((bbox.height - 0.7) * 10)
    cb.set_alpha(1)

    if show:
        plt.show()

    return ax
