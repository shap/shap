from __future__ import annotations

import typing
import warnings
from typing import Any, Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.markers import MarkerStyle

from .._explanation import Explanation
from ..utils import approximate_interactions, convert_name
from ..utils._exceptions import DimensionError
from ..utils._general import encode_array_if_needed
from . import colors
from ._labels import labels
from ._utils import AxisLimitSpec, parse_axis_limit


# TODO: Make the color bar a one-sided beeswarm plot so we can see the density along the color axis
def scatter(
    shap_values: Explanation,
    color: str | Explanation | None = "#1E88E5",
    hist: bool = True,
    axis_color="#333333",
    cmap=colors.red_blue,
    dot_size=16,
    x_jitter: float | Literal["auto"] = "auto",
    alpha: float = 1.0,
    title: str | None = None,
    xmin: AxisLimitSpec = None,
    xmax: AxisLimitSpec = None,
    ymin: AxisLimitSpec = None,
    ymax: AxisLimitSpec = None,
    overlay: dict[str, Any] | None = None,
    ax: plt.Axes | None = None,
    ylabel: str = "SHAP value",
    show: bool = True,
):
    """Create a SHAP dependence scatter plot, optionally colored by an interaction feature.

    Plots the value of the feature on the x-axis and the SHAP value of the same feature
    on the y-axis. This shows how the model depends on the given feature, and is like a
    richer extension of classical partial dependence plots. Vertical dispersion of the
    data points represents interaction effects. Grey ticks along the y-axis are data
    points where the feature's value was NaN.

    Note that if you want to change the data being displayed, you can update the
    ``shap_values.display_features`` attribute and it will then be used for plotting instead of
    ``shap_values.data``.

    Parameters
    ----------
    shap_values : shap.Explanation
        Typically a single column of an :class:`.Explanation` object
        (i.e. ``shap_values[:, "Feature A"]``).

        Alternatively, pass multiple columns to create several subplots
        (i.e. ``shap_values[:, ["Feature A", "Feature B"]]``).

    color : string or shap.Explanation, optional
        How to color the scatter plot points. This can be a fixed color string, or an
        :class:`.Explanation` object.

        If it is an :class:`.Explanation` object, then the scatter plot points are
        colored by the feature that seems to have the strongest interaction effect with
        the feature given by the ``shap_values`` argument. This is calculated using
        :func:`shap.utils.approximate_interactions`.

        If only a single column of an :class:`.Explanation` object is passed, then that
        feature column will be used to color the data points.

    hist : bool
        Whether to show a light histogram along the x-axis to show the density of the
        data. Note that the histogram is normalized such that if all the points were in
        a single bin, then that bin would span the full height of the plot. Defaults to
        ``True``.

    x_jitter : 'auto' or float
        Adds random jitter to feature values by specifying a float between 0 to 1. May
        increase plot readability when a feature is discrete. By default, ``x_jitter``
        is chosen based on auto-detection of categorical features.

    title: str, optional
        Plot title.

    alpha : float
        The transparency of the data points (between 0 and 1). This can be useful to
        show the density of the data points when using a large dataset.

    xmin, xmax, ymin, ymax : float, string, aggregated Explanation or None
        Desired axis limits. Can be a float to specify a fixed limit.

        It can be a string of the format ``"percentile(float)"`` to denote that
        percentile of the feature's value.

        It can also be an aggregated column of a single column of an :class:`.Explanation`,
        such as ``explanation[:, "feature_name"].percentile(20)``.

    overlay: dict, optional
        Optional dictionary of up to three additional curves to overlay as line plots.

        The dictionary maps a curve name to a list of (xvalues, yvalues) pairs, where
        there is one pair for each feature to be plotted.

    ax : matplotlib Axes, optional
        Optionally specify an existing :external+mpl:class:`matplotlib.axes.Axes` object, into which
        the plot will be placed.

        Only supported when plotting a single feature.

    show : bool
        Whether :external+mpl:func:`matplotlib.pyplot.show()` is called before returning.

        Setting this to ``False`` allows the plot to be customized further after it
        has been created.

    Returns
    -------
    ax : matplotlib Axes object
        Returns the :external+mpl:class:`~matplotlib.axes.Axes` object with the plot drawn onto it. Only
        returned if ``show=False``.

    Examples
    --------
    See `scatter plot examples <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/scatter.html>`_.

    """
    if not isinstance(shap_values, Explanation):
        raise TypeError("The shap_values parameter must be a shap.Explanation object!")

    # see if we are plotting multiple columns
    if not isinstance(shap_values.feature_names, str) and len(shap_values.feature_names) > 0:
        if ax is not None:
            raise ValueError("The ax parameter is not supported when plotting multiple features")
        # Define order of columns (features) to plot based on average shap value
        inds = np.argsort(np.abs(shap_values.values).mean(0))
        ymin = parse_axis_limit(ymin, shap_values.values, is_shap_axis=True)
        ymax = parse_axis_limit(ymax, shap_values.values, is_shap_axis=True)
        ymin, ymax = _suggest_buffered_limits(ymin, ymax, shap_values.values)
        _ = plt.subplots(1, len(inds), figsize=(min(6 * len(inds), 15), 5))
        for i in inds:
            ax = plt.subplot(1, len(inds), i + 1)
            scatter(shap_values[:, i], color=color, show=False, ax=ax, ymin=ymin, ymax=ymax)
            if overlay is not None:
                line_styles = ["solid", "dotted", "dashed"]
                for j, name in enumerate(overlay):
                    vals = overlay[name]
                    if isinstance(vals[i][0][0], (float, int)):
                        plt.plot(vals[i][0], vals[i][1], color="#000000", linestyle=line_styles[j], label=name)
            if i == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel("")
                ax.set_yticks([])
                ax.spines["left"].set_visible(False)
        if overlay is not None:
            plt.legend()
        if show:
            plt.show()
        return

    if len(shap_values.shape) != 1:
        raise DimensionError(
            "The passed Explanation object has multiple columns. Please pass a single feature column to "
            "shap.plots.scatter like: shap_values[:, column]"
        )

    # this unpacks the explanation object for the code that was written earlier
    feature_names = [shap_values.feature_names]
    ind: int = 0
    shap_values_arr = shap_values.values.reshape(-1, 1)
    features = shap_values.data.reshape(-1, 1)
    if shap_values.display_data is None:
        display_features = features
    else:
        display_features = shap_values.display_data.reshape(-1, 1)
    interaction_index: str | int | None = None

    # wrap np.arrays as Explanations
    if isinstance(color, np.ndarray):
        color = Explanation(values=color, base_values=None, data=color)

    # TODO: This stacking could be avoided if we use the new shap.utils.potential_interactions function
    if isinstance(color, Explanation):
        shap_values2 = color
        if issubclass(type(shap_values2.feature_names), (str, int)):
            feature_names.append(shap_values2.feature_names)
            shap_values_arr = np.hstack([shap_values_arr, shap_values2.values.reshape(-1, len(feature_names) - 1)])
            features = np.hstack([features, shap_values2.data.reshape(-1, len(feature_names) - 1)])
            if shap_values2.display_data is None:
                display_features = np.hstack([display_features, shap_values2.data.reshape(-1, len(feature_names) - 1)])
            else:
                display_features = np.hstack(
                    [display_features, shap_values2.display_data.reshape(-1, len(feature_names) - 1)]
                )
        else:
            feature_names2 = np.array(shap_values2.feature_names)
            mask = ~(feature_names[0] == feature_names2)
            feature_names.extend(feature_names2[mask])
            shap_values_arr = np.hstack([shap_values_arr, shap_values2.values[:, mask]])
            features = np.hstack([features, shap_values2.data[:, mask]])
            if shap_values2.display_data is None:
                display_features = np.hstack([display_features, shap_values2.data[:, mask]])
            else:
                display_features = np.hstack([display_features, shap_values2.display_data[:, mask]])
        color = None
        interaction_index = "auto"

    if isinstance(shap_values_arr, list):
        raise TypeError(
            "The passed shap_values_arr are a list not an array! If you have a list of explanations try "
            "passing shap_values_arr[0] instead to explain the first output class of a multi-output model."
        )

    # convert from DataFrames if we got any
    if isinstance(features, pd.DataFrame):
        if feature_names is None:
            feature_names = features.columns
        features = features.values

    if feature_names is None:
        feature_names = [labels["FEATURE"] % str(i) for i in range(shap_values_arr.shape[1])]

    # allow vectors to be passed
    if len(shap_values_arr.shape) == 1:
        shap_values_arr = np.reshape(shap_values_arr, (len(shap_values_arr), 1))
    if len(features.shape) == 1:
        features = np.reshape(features, (len(features), 1))

    # pick jitter for categorical features
    if x_jitter == "auto":
        x_jitter = _suggest_x_jitter(features[:, ind])

    # guess what other feature as the stongest interaction with the plotted feature
    if interaction_index == "auto":
        interaction_index = approximate_interactions(ind, shap_values_arr, features)[0]
    interaction_index = convert_name(interaction_index, shap_values_arr, feature_names)
    categorical_interaction = False

    # create a matplotlib figure, if `ax` hasn't been specified.
    if ax is None:
        figsize = (7.5, 5) if interaction_index != ind and interaction_index is not None else (6, 5)
        _, ax = plt.subplots(figsize=figsize)

    assert shap_values_arr.shape[0] == features.shape[0], (
        "'shap_values_arr' and 'features' values must have the same number of rows!"
    )
    assert shap_values_arr.shape[1] == features.shape[1], (
        "'shap_values_arr' must have the same number of columns as 'features'!"
    )

    # get both the raw and display feature values
    oinds = np.arange(
        shap_values_arr.shape[0]
    )  # we randomize the ordering so plotting overlaps are not related to data ordering
    np.random.shuffle(oinds)
    xv = encode_array_if_needed(features[oinds, ind])
    xd = display_features[oinds, ind]

    s = shap_values_arr[oinds, ind]
    if isinstance(xd[0], str):
        name_map = {}
        for i in range(len(xv)):
            name_map[xd[i]] = xv[i]
        xnames = list(name_map.keys())

    # allow a single feature name to be passed alone
    if isinstance(feature_names, str):
        feature_names = [feature_names]
    name = feature_names[ind]

    # get both the raw and display color values
    color_norm = None
    if interaction_index is not None:
        interaction_feature_values = encode_array_if_needed(features[:, interaction_index])
        cv = interaction_feature_values
        cd = display_features[:, interaction_index]
        clow = np.nanpercentile(cv.astype(float), 5)
        chigh = np.nanpercentile(cv.astype(float), 95)
        if clow == chigh:
            clow = np.nanmin(cv.astype(float))
            chigh = np.nanmax(cv.astype(float))
        if isinstance(cd[0], str):
            cname_map = {}
            for i in range(len(cv)):
                cname_map[cd[i]] = cv[i]
            cnames = list(cname_map.keys())
            categorical_interaction = True
        elif clow % 1 == 0 and chigh % 1 == 0 and chigh - clow < 10:
            categorical_interaction = True

        # discritize colors for categorical features
        if categorical_interaction and clow != chigh:
            clow = np.nanmin(cv.astype(float))
            chigh = np.nanmax(cv.astype(float))
            bounds = np.linspace(clow, chigh, min(int(chigh - clow + 2), cmap.N - 1))
            color_norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N - 1)

    # optionally add jitter to feature values
    xv_no_jitter = xv.copy()
    if x_jitter > 0:
        if x_jitter > 1:
            x_jitter = 1
        xvals = xv.copy()
        if isinstance(xvals[0], float):
            xvals = xvals.astype(float)
            xvals = xvals[~np.isnan(xvals)]
        xvals = np.unique(xvals)  # returns a sorted array
        if len(xvals) >= 2:
            smallest_diff = np.min(np.diff(xvals))
            jitter_amount = x_jitter * smallest_diff
            xv += (np.random.random_sample(size=len(xv)) * jitter_amount) - (jitter_amount / 2)

    # the actual scatter plot, TODO: adapt the dot_size to the number of data points?
    xv_nan = np.isnan(xv)
    xv_notnan = np.invert(xv_nan)
    if interaction_index is not None:
        # plot the nan values in the interaction feature as grey
        cvals = encode_array_if_needed(features[oinds, interaction_index]).astype(np.float64)
        cvals_imp = cvals.copy()
        cvals_imp[np.isnan(cvals)] = (clow + chigh) / 2.0
        cvals[cvals_imp > chigh] = chigh
        cvals[cvals_imp < clow] = clow
        if color_norm is None:
            vmin = clow
            vmax = chigh
        else:
            vmin = vmax = None
        ax.axhline(0, color="#888888", lw=0.5, dashes=(1, 5), zorder=-1)
        p = ax.scatter(
            xv[xv_notnan],
            s[xv_notnan],
            s=dot_size,
            linewidth=0,
            c=cvals[xv_notnan],
            cmap=cmap,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
            norm=color_norm,
            rasterized=len(xv) > 500,
        )
        p.set_array(cvals[xv_notnan])
    else:
        p = ax.scatter(xv, s, s=dot_size, linewidth=0, color=color, alpha=alpha, rasterized=len(xv) > 500)

    if interaction_index != ind and interaction_index is not None:
        # draw the color bar
        if isinstance(cd[0], str):
            tick_positions = np.array([cname_map[n] for n in cnames])
            tick_positions *= 1 - 1 / len(cnames)
            tick_positions += 0.5 * (chigh - clow) / (chigh - clow + 1)
            cb = plt.colorbar(p, ticks=tick_positions, ax=ax, aspect=80)
            cb.set_ticklabels(cnames)
        else:
            cb = plt.colorbar(p, ax=ax, aspect=80)

        # Type narrowing for mypy
        assert isinstance(interaction_index, (int, np.integer)), f"Unexpected {type(interaction_index)=}"
        cb.set_label(feature_names[interaction_index], size=13)
        cb.ax.tick_params(labelsize=11)
        if categorical_interaction:
            cb.ax.tick_params(length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)  # type: ignore
    #         bbox = cb.ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    #         cb.ax.set_aspect((bbox.height - 0.7) * 20)

    xmin = parse_axis_limit(xmin, xv, is_shap_axis=False)
    xmax = parse_axis_limit(xmax, xv, is_shap_axis=False)
    ymin = parse_axis_limit(ymin, s, is_shap_axis=True)
    ymax = parse_axis_limit(ymax, s, is_shap_axis=True)
    if xmin is not None or xmax is not None:
        ax.set_xlim(*_suggest_buffered_limits(xmin, xmax, xv))
    if ymin is not None or ymax is not None:
        ax.set_ylim(*_suggest_buffered_limits(ymin, ymax, s))

    # plot any nan feature values as tick marks along the y-axis
    xlim = ax.get_xlim()
    if interaction_index is not None:
        p = ax.scatter(
            xlim[0] * np.ones(xv_nan.sum()),
            s[xv_nan],
            marker=MarkerStyle(1),
            linewidth=2,
            c=cvals_imp[xv_nan],
            cmap=cmap,
            alpha=alpha,
            vmin=clow,
            vmax=chigh,
        )
        p.set_array(cvals[xv_nan])
    else:
        ax.scatter(
            xlim[0] * np.ones(xv_nan.sum()), s[xv_nan], marker=MarkerStyle(1), linewidth=2, color=color, alpha=alpha
        )
    ax.set_xlim(xlim)

    # the histogram of the data
    if hist:
        _plot_histogram(ax, xv, xv_no_jitter)

    plt.sca(ax)

    # make the plot more readable
    ax.set_xlabel(name, color=axis_color, fontsize=13)
    ax.set_ylabel(labels["VALUE_FOR"] % name, color=axis_color, fontsize=13)
    if title is not None:
        ax.set_title(title, color=axis_color, fontsize=13)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color, labelsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor(axis_color)
    if isinstance(xd[0], str):
        ax.set_xticks([name_map[n] for n in xnames])
        ax.set_xticklabels(xnames, fontdict=dict(rotation="vertical", fontsize=11))
    if show:
        with warnings.catch_warnings():  # ignore expected matplotlib warnings
            warnings.simplefilter("ignore", RuntimeWarning)
            plt.show()
    else:
        return ax


def _suggest_buffered_limits(ax_min: float | None, ax_max: float | None, values: np.ndarray) -> tuple[float, float]:
    """If either limit is None, suggest suitable value including a buffer either side"""
    nan_max = np.nanmax(values) if ax_max is None else ax_max
    nan_min = np.nanmin(values) if ax_min is None else ax_min
    buffer = (nan_max - nan_min) / 20
    if ax_min is None:
        ax_min = float(nan_min - buffer)
    if ax_max is None:
        ax_max = float(nan_max + buffer)
    return ax_min, ax_max


def _suggest_x_jitter(values: np.ndarray) -> float:
    """Suggest a suitable x_jitter value based on the unique values in the feature"""
    unique_vals = np.sort(np.unique(values))
    if len(unique_vals) < 2:
        # If there is only one unique value, no jitter is needed
        return 0.0
    try:
        # Identify the smallest difference between unique values
        diffs = np.diff(unique_vals)
        min_dist = np.min(diffs[diffs > 1e-8])
    except (TypeError, ValueError):
        # If unique_vals contains non-numeric values or all differences are to small, set arbitrarily at 1
        min_dist = 1

    num_points_per_value = len(values) / len(unique_vals)
    if num_points_per_value < 10:
        # categorical = False
        x_jitter = 0
    elif num_points_per_value < 100:
        # categorical = True
        x_jitter = min_dist * 0.1
    else:
        # categorical = True
        x_jitter = min_dist * 0.2
    return x_jitter


def _plot_histogram(ax: plt.Axes, xv, xv_no_jitter):
    """Add a histogram of the data on a matching secondary axes"""
    ax2 = typing.cast("plt.Axes", ax.twinx())
    xlim = ax.get_xlim()
    xvals = np.unique(xv_no_jitter)

    # Determine suitable bins and limits
    bins: list[float] | int  # Hint for mypy
    if len(xvals) / len(xv_no_jitter) < 0.2 and len(xvals) < 75 and np.max(xvals) < 75 and np.min(xvals) >= 0:
        np.sort(xvals)
        bins = []
        for i in range(int(np.max(xvals) + 1)):
            bins.append(i - 0.5)
        bins.append(int(np.max(xvals)) + 0.5)

        lim = np.floor(np.min(xvals) - 0.5) + 0.5, np.ceil(np.max(xvals) + 0.5) - 0.5
        ax.set_xlim(lim)
    else:
        if len(xv_no_jitter) >= 500:
            bins = 50
        elif len(xv_no_jitter) >= 200:
            bins = 20
        elif len(xv_no_jitter) >= 100:
            bins = 10
        else:
            bins = 5

    # Plot the histogram
    ax2.hist(
        xv[~np.isnan(xv)],
        bins,
        density=False,
        facecolor="#000000",
        alpha=0.1,
        range=(xlim[0], xlim[1]),
        zorder=-1,
    )
    ax2.set_ylim(0, len(xv))
    ax2.xaxis.set_ticks_position("bottom")
    ax2.yaxis.set_ticks_position("left")
    ax2.yaxis.set_ticks([])
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)


def dependence_legacy(
    ind,
    shap_values=None,
    features=None,
    feature_names=None,
    display_features=None,
    interaction_index="auto",
    color="#1E88E5",
    axis_color="#333333",
    cmap=None,
    dot_size=16,
    x_jitter=0,
    alpha=1,
    title=None,
    xmin=None,
    xmax=None,
    ax=None,
    show=True,
    ymin=None,
    ymax=None,
):
    """Create a SHAP dependence plot, colored by an interaction feature.

    Plots the value of the feature on the x-axis and the SHAP value of the same feature
    on the y-axis. This shows how the model depends on the given feature, and is like a
    richer extension of the classical partial dependence plots. Vertical dispersion of the
    data points represents interaction effects. Grey ticks along the y-axis are data
    points where the feature's value was NaN.


    Parameters
    ----------
    ind : int or string
        If this is an int it is the index of the feature to plot. If this is a string it is
        either the name of the feature to plot, or it can have the form "rank(int)" to specify
        the feature with that rank (ordered by mean absolute SHAP value over all the samples).

    shap_values : numpy.array
        Matrix of SHAP values (# samples x # features).

    features : numpy.array or pandas.DataFrame
        Matrix of feature values (# samples x # features).

    feature_names : list
        Names of the features (length # features).

    display_features : numpy.array or pandas.DataFrame
        Matrix of feature values for visual display (such as strings instead of coded values).

    interaction_index : "auto", None, int, or string
        The index of the feature used to color the plot. The name of a feature can also be passed
        as a string. If "auto" then shap.common.approximate_interactions is used to pick what
        seems to be the strongest interaction (note that to find to true stongest interaction you
        need to compute the SHAP interaction values).

    x_jitter : float (0 - 1)
        Adds random jitter to feature values. May increase plot readability when feature
        is discrete.

    alpha : float
        The transparency of the data points (between 0 and 1). This can be useful to the
        show density of the data points when using a large dataset.

    xmin : float or string
        Represents the lower bound of the plot's x-axis. It can be a string of the format
        "percentile(float)" to denote that percentile of the feature's value used on the x-axis.

    xmax : float or string
        Represents the upper bound of the plot's x-axis. It can be a string of the format
        "percentile(float)" to denote that percentile of the feature's value used on the x-axis.

    ax : matplotlib Axes object
         Optionally specify an existing matplotlib Axes object, into which the plot will be placed.
         In this case we do not create a Figure, otherwise we do.

    ymin : float
        Represents the lower bound of the plot's y-axis.

    ymax : float
        Represents the upper bound of the plot's y-axis.

    """
    if cmap is None:
        cmap = colors.red_blue

    if isinstance(shap_values, list):
        raise TypeError(
            "The passed shap_values are a list not an array! If you have a list of explanations try "
            "passing shap_values[0] instead to explain the first output class of a multi-output model."
        )

    # convert from DataFrames if we got any
    if isinstance(features, pd.DataFrame):
        if feature_names is None:
            feature_names = features.columns
        features = features.values
    if isinstance(display_features, pd.DataFrame):
        if feature_names is None:
            feature_names = display_features.columns
        display_features = display_features.values
    elif display_features is None:
        display_features = features

    if feature_names is None:
        feature_names = [labels["FEATURE"] % str(i) for i in range(shap_values.shape[1])]

    # allow vectors to be passed
    if len(shap_values.shape) == 1:
        shap_values = np.reshape(shap_values, (len(shap_values), 1))
    if len(features.shape) == 1:
        features = np.reshape(features, (len(features), 1))

    ind = convert_name(ind, shap_values, feature_names)

    # guess what other feature as the stongest interaction with the plotted feature
    if not hasattr(ind, "__len__"):
        if interaction_index == "auto":
            interaction_index = approximate_interactions(ind, shap_values, features)[0]
        interaction_index = convert_name(interaction_index, shap_values, feature_names)
    categorical_interaction = False

    # create a matplotlib figure, if `ax` hasn't been specified.
    if not ax:
        figsize = (7.5, 5) if interaction_index != ind and interaction_index is not None else (6, 5)
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
    else:
        fig = ax.get_figure()

    # plotting SHAP interaction values
    if len(shap_values.shape) == 3 and hasattr(ind, "__len__") and len(ind) == 2:
        ind1 = convert_name(ind[0], shap_values, feature_names)
        ind2 = convert_name(ind[1], shap_values, feature_names)
        if ind1 == ind2:
            proj_shap_values = shap_values[:, ind2, :]
        else:
            proj_shap_values = shap_values[:, ind2, :] * 2  # off-diag values are split in half

        # there is no interaction coloring for the main effect
        if ind1 == ind2:
            fig.set_size_inches(6, 5, forward=True)

        # TODO: remove recursion; generally the functions should be shorter for more maintainable code
        dependence_legacy(
            ind1,
            proj_shap_values,
            features,
            feature_names=feature_names,
            interaction_index=(None if ind1 == ind2 else ind2),
            display_features=display_features,
            ax=ax,
            show=False,
            xmin=xmin,
            xmax=xmax,
            x_jitter=x_jitter,
            alpha=alpha,
        )
        if ind1 == ind2:
            ax.set_ylabel(labels["MAIN_EFFECT"] % feature_names[ind1])
        else:
            ax.set_ylabel(labels["INTERACTION_EFFECT"] % (feature_names[ind1], feature_names[ind2]))

        if show:
            plt.show()
        return

    assert shap_values.shape[0] == features.shape[0], (
        "'shap_values' and 'features' values must have the same number of rows!"
    )
    assert shap_values.shape[1] == features.shape[1], (
        "'shap_values' must have the same number of columns as 'features'!"
    )

    # get both the raw and display feature values
    oinds = np.arange(
        shap_values.shape[0]
    )  # we randomize the ordering so plotting overlaps are not related to data ordering
    np.random.shuffle(oinds)

    xv = encode_array_if_needed(features[oinds, ind])

    xd = display_features[oinds, ind]
    s = shap_values[oinds, ind]
    if isinstance(xd[0], str):
        name_map = {}
        for i in range(len(xv)):
            name_map[xd[i]] = xv[i]
        xnames = list(name_map.keys())

    # allow a single feature name to be passed alone
    if isinstance(feature_names, str):
        feature_names = [feature_names]
    name = feature_names[ind]

    # get both the raw and display color values
    color_norm = None
    if interaction_index is not None:
        interaction_feature_values = encode_array_if_needed(features[:, interaction_index])
        cv = interaction_feature_values
        cd = display_features[:, interaction_index]
        clow = np.nanpercentile(cv.astype(float), 5)
        chigh = np.nanpercentile(cv.astype(float), 95)
        if clow == chigh:
            clow = np.nanmin(cv.astype(float))
            chigh = np.nanmax(cv.astype(float))
        if isinstance(cd[0], str):
            cname_map = {}
            for i in range(len(cv)):
                cname_map[cd[i]] = cv[i]
            cnames = list(cname_map.keys())
            categorical_interaction = True
        elif clow % 1 == 0 and chigh % 1 == 0 and chigh - clow < 10:
            categorical_interaction = True

        # discritize colors for categorical features
        if categorical_interaction and clow != chigh:
            clow = np.nanmin(cv.astype(float))
            chigh = np.nanmax(cv.astype(float))
            bounds = np.linspace(clow, chigh, min(int(chigh - clow + 2), cmap.N - 1))
            color_norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N - 1)

    # optionally add jitter to feature values
    if x_jitter > 0:
        if x_jitter > 1:
            x_jitter = 1
        xvals = xv.copy()
        if isinstance(xvals[0], float):
            xvals = xvals.astype(float)
            xvals = xvals[~np.isnan(xvals)]
        xvals = np.unique(xvals)  # returns a sorted array
        if len(xvals) >= 2:
            smallest_diff = np.min(np.diff(xvals))
            jitter_amount = x_jitter * smallest_diff
            xv += (np.random.random_sample(size=len(xv)) * jitter_amount) - (jitter_amount / 2)

    # the actual scatter plot, TODO: adapt the dot_size to the number of data points?
    xv_nan = np.isnan(xv)
    xv_notnan = np.invert(xv_nan)
    if interaction_index is not None:
        # plot the nan values in the interaction feature as grey
        cvals = interaction_feature_values[oinds].astype(np.float64)
        cvals_imp = cvals.copy()
        cvals_imp[np.isnan(cvals)] = (clow + chigh) / 2.0
        cvals[cvals_imp > chigh] = chigh
        cvals[cvals_imp < clow] = clow
        p = ax.scatter(
            xv[xv_notnan],
            s[xv_notnan],
            s=dot_size,
            linewidth=0,
            c=cvals[xv_notnan],
            cmap=cmap,
            alpha=alpha,
            norm=color_norm,
            rasterized=len(xv) > 500,
        )
        p.set_array(cvals[xv_notnan])
    else:
        p = ax.scatter(xv, s, s=dot_size, linewidth=0, color=color, alpha=alpha, rasterized=len(xv) > 500)

    if interaction_index != ind and interaction_index is not None:
        # draw the color bar
        if isinstance(cd[0], str):
            tick_positions = [cname_map[n] for n in cnames]
            if len(tick_positions) == 2:
                tick_positions[0] -= 0.25
                tick_positions[1] += 0.25
            cb = plt.colorbar(p, ticks=tick_positions, ax=ax, aspect=80)
            cb.set_ticklabels(cnames)
        else:
            cb = plt.colorbar(p, ax=ax, aspect=80)

        cb.set_label(feature_names[interaction_index], size=13)
        cb.ax.tick_params(labelsize=11)
        if categorical_interaction:
            cb.ax.tick_params(length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)  # type: ignore
    #         bbox = cb.ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    #         cb.ax.set_aspect((bbox.height - 0.7) * 20)

    # handles any setting of xmax and xmin
    # note that we handle None,float, or "percentile(float)" formats
    if xmin is not None or xmax is not None:
        if isinstance(xmin, str) and xmin.startswith("percentile"):
            xmin = np.nanpercentile(xv, float(xmin[11:-1]))
        if isinstance(xmax, str) and xmax.startswith("percentile"):
            xmax = np.nanpercentile(xv, float(xmax[11:-1]))

        if xmin is None or xmin == np.nanmin(xv):
            xmin = np.nanmin(xv) - (xmax - np.nanmin(xv)) / 20
        if xmax is None or xmax == np.nanmax(xv):
            xmax = np.nanmax(xv) + (np.nanmax(xv) - xmin) / 20

        ax.set_xlim(xmin, xmax)

    # plot any nan feature values as tick marks along the y-axis
    xlim = ax.get_xlim()
    if interaction_index is not None:
        p = ax.scatter(
            xlim[0] * np.ones(xv_nan.sum()),
            s[xv_nan],
            marker=1,
            linewidth=2,
            c=cvals_imp[xv_nan],
            cmap=cmap,
            alpha=alpha,
            vmin=clow,
            vmax=chigh,
        )
        p.set_array(cvals[xv_nan])
    else:
        ax.scatter(xlim[0] * np.ones(xv_nan.sum()), s[xv_nan], marker=1, linewidth=2, color=color, alpha=alpha)
    ax.set_xlim(xlim)

    # make the plot more readable
    ax.set_xlabel(name, color=axis_color, fontsize=13)
    ax.set_ylabel(labels["VALUE_FOR"] % name, color=axis_color, fontsize=13)

    if (ymin is not None) or (ymax is not None):
        if ymin is None:
            ymin = -ymax
        if ymax is None:
            ymax = -ymin

        ax.set_ylim(ymin, ymax)

    if title is not None:
        ax.set_title(title, color=axis_color, fontsize=13)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color, labelsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor(axis_color)
    if isinstance(xd[0], str):
        ax.set_xticks([name_map[n] for n in xnames])
        ax.set_xticklabels(xnames, fontdict=dict(rotation="vertical", fontsize=11))
    if show:
        with warnings.catch_warnings():  # ignore expected matplotlib warnings
            warnings.simplefilter("ignore", RuntimeWarning)
            plt.show()
