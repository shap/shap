from __future__ import annotations

from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import sklearn

from ..utils import convert_name
from . import colors
from ._labels import labels

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.figure import Figure


def embedding(
    ind: int | str,
    shap_values: np.ndarray,
    feature_names: list[str] | None = None,
    method: str | np.ndarray = "pca",
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
        Setting this to ``False`` allows the plot to be customized further after it has
        been created.

    ax : matplotlib Axes, optional
        Axes object to draw the plot onto, otherwise uses the current Axes.

    Returns
    -------
    ax : matplotlib Axes
        Returns the :external+mpl:class:`~matplotlib.axes.Axes` object with the plot drawn onto it.

    """
    if ax is None:
        ax = plt.gca()

    if feature_names is None:
        feature_names = [labels["FEATURE"] % str(i) for i in range(shap_values.shape[1])]

    converted_ind = convert_name(ind, shap_values, feature_names)
    if converted_ind == "sum()":
        cvals = shap_values.sum(1)
        fname = "sum(SHAP values)"
    else:
        feature_ind = cast("int", converted_ind)
        cvals = shap_values[:, feature_ind]
        fname = feature_names[feature_ind]

    # see if we need to compute the embedding
    if isinstance(method, str) and method == "pca":
        pca = sklearn.decomposition.PCA(2)
        embedding_values = pca.fit_transform(shap_values)
    elif hasattr(method, "shape") and method.shape[1] == 2:
        embedding_values = method
    else:
        raise ValueError(f"Unsupported embedding method: {method!r}")

    point_collection = ax.scatter(
        embedding_values[:, 0],
        embedding_values[:, 1],
        c=cvals,
        cmap=colors.red_blue,
        alpha=alpha,
        linewidth=0,
    )
    ax.axis("off")
    # plt.title(feature_names[ind])
    fig = cast("Figure", ax.get_figure())
    cb = fig.colorbar(point_collection, ax=ax)
    cb.set_label("SHAP value for\n" + fname, size=13)
    cb.outline.set_visible(False)  # type: ignore

    fig.set_size_inches(7.5, 5)
    bbox = cb.ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    cb.ax.set_aspect((bbox.height - 0.7) * 10)
    cb.set_alpha(1)
    if show:
        plt.show()
    return ax
