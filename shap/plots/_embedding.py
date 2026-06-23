from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import sklearn

from .. import Explanation
from ..utils import convert_name
from . import colors
from ._labels import labels

if TYPE_CHECKING:
    from collections.abc import Sequence


def embedding(
    ind: int | str,
    shap_values: Explanation,
    feature_names: Sequence[str] | None = None,
    method: Literal["pca"] | np.ndarray | Any = "pca",
    alpha: float = 1.0,
    show: bool = True,
    ax: plt.Axes | None = None,
):
    """Use the SHAP values as an embedding projected to 2D for visualization.

    Parameters
    ----------
    ind : int or string
        If this is an int it is the index of the feature to use to color the embedding.
        If this is a string it is either the name of the feature, or it can have the
        form ``"rank(int)"`` to specify the feature with that rank (ordered by mean
        absolute SHAP value over all the samples), or ``"sum()"`` to mean the sum of
        all the SHAP values for each sample.

    shap_values : shap.Explanation
        A two-dimensional :class:`.Explanation` object containing SHAP values with shape
        ``(# samples, # features)``.

    feature_names : sequence of strings, optional
        Override the feature names used for labeling.

    method : "pca" or numpy.ndarray
        How to reduce the SHAP values to 2D. If ``"pca"`` then a 2D PCA projection of
        the SHAP values is used. If a numpy array, it must have shape ``(# samples, 2)``
        and represent the embedding coordinates.

    alpha : float
        The transparency of the data points (between 0 and 1).

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
        warnings.warn(
            "Passing a numpy array to the embedding plot is deprecated and will be removed in a future version. "
            "Please pass a shap.Explanation object instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        shap_values = Explanation(values=np.asarray(shap_values), feature_names=feature_names)

    if len(shap_values.shape) != 2:
        raise ValueError(
            "The embedding plot expects a 2D Explanation of SHAP values with shape (# samples, # features)."
        )

    shap_values_arr = np.asarray(shap_values.values)

    if feature_names is None:
        feature_names = shap_values.feature_names or [
            labels["FEATURE"] % str(i) for i in range(shap_values_arr.shape[1])
        ]

    feature_names_list = list(feature_names)  # type: ignore
    ind_converted = convert_name(ind, shap_values_arr, feature_names_list)
    if ind_converted == "sum()":
        cvals = shap_values_arr.sum(1)
        fname = "sum(SHAP values)"
    else:
        assert isinstance(ind_converted, int)
        cvals = shap_values_arr[:, ind_converted]
        fname = feature_names_list[ind_converted]

    # compute the embedding
    if isinstance(method, str) and method == "pca":
        pca = sklearn.decomposition.PCA(2)
        embedding_values = pca.fit_transform(shap_values_arr)
    else:
        embedding_values = np.asarray(method)
        if embedding_values.ndim != 2 or embedding_values.shape[1] != 2:
            raise ValueError("Unsupported embedding method. Pass method='pca' or an array of shape (# samples, 2).")
        if embedding_values.shape[0] != shap_values_arr.shape[0]:
            raise ValueError(
                "When passing explicit embedding coordinates, method must have the same number of rows as shap_values."
            )

    if ax is None:
        ax = plt.gca()
        fig = plt.gcf()
        # Only modify the figure size if ax was not passed in
        fig.set_size_inches(7.5, 5)
    else:
        fig = ax.figure

    sc = ax.scatter(
        embedding_values[:, 0],
        embedding_values[:, 1],
        c=cvals,
        cmap=colors.red_blue,
        alpha=alpha,
        linewidth=0,
    )
    ax.axis("off")

    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("SHAP value for\n" + fname, size=13)
    cb.outline.set_visible(False)  # type: ignore

    bbox = cb.ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    cb.ax.set_aspect((bbox.height - 0.7) * 10)
    cb.set_alpha(1)

    if show:
        plt.show()
        return
    return ax
