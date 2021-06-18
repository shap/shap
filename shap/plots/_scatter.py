from __future__ import division

import numpy as np
import warnings
try:
    import matplotlib.pyplot as pl
    import matplotlib
except ImportError:
    warnings.warn("matplotlib could not be loaded!")
    pass
from ._labels import labels
from . import colors
from ..utils import convert_name, approximate_interactions
from ..utils._general import encode_array_if_needed
from .._explanation import Explanation


# TODO: Make the color bar a one-sided beeswarm plot so we can see the density along the color axis
def scatter(shap_values, color="#1E88E5", hist=True, axis_color="#333333", cmap=colors.red_blue,
            dot_size=16, x_jitter="auto", alpha=1, title=None, xmin=None, xmax=None, ymin=None, ymax=None,
            overlay=None, ax=None, ylabel="SHAP value", show=True):
    """ Create a SHAP dependence scatter plot, colored by an interaction feature.

    Plots the value of the feature on the x-axis and the SHAP value of the same feature
    on the y-axis. This shows how the model depends on the given feature, and is like a
    richer extenstion of classical parital dependence plots. Vertical dispersion of the
    data points represents interaction effects. Grey ticks along the y-axis are data
    points where the feature's value was NaN.

    Note that if you want to change the data being displayed you can update the
    shap_values.display_features attribute and it will then be used for plotting instead of
    shap_values.data.


    Parameters
    ----------
    shap_values : shap.Explanation
        A single column of a SHAP Explanation object (i.e. shap_values[:,"Feature A"]).

    color : string or shap.Explanation
        How to color the scatter plot points. This can be a fixed color string, or a Explanation object.
        If it is an explanation object then the scatter plot points are colored by the feature that
        seems to have the strongest interaction effect with the feature given by the shap_values argument.
        This is calculated using shap.utils.approximate_interactions.
        If only a single column of an Explanation object is passed then that feature column will be used
        to color the data points.

    hist : bool
        Whether to show a light histogram along the x-axis to show the density of the data. Note that the
        histogram is normalized that that if all the point were in a single bin then that bin would span
        the full height of the plot.

    x_jitter : 'auto' or float (0 - 1)
        Adds random jitter to feature values. May increase plot readability when a feature
        is discrete. By default x_jitter is chosen base on auto-detection of categorical features

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

    """


    assert str(type(shap_values)).endswith("Explanation'>"), "The shap_values paramemter must be a shap.Explanation object!"

    # see if we are plotting multiple columns
    if not isinstance(shap_values.feature_names, str) and len(shap_values.feature_names) > 0:
        inds = np.argsort(np.abs(shap_values.values).mean(0))
        nan_min = np.nanmin(shap_values.values)
        nan_max = np.nanmax(shap_values.values)
        if ymin is None:
            ymin = nan_min - (nan_max - nan_min)/20
        if ymax is None:
            ymax = nan_max + (nan_max - nan_min)/20
        f = pl.subplots(1, len(inds), figsize=(min(6 * len(inds), 15), 5))
        for i in inds:
            ax = pl.subplot(1,len(inds),i+1)
            scatter(shap_values[:,i], show=False, ax=ax, ymin=ymin, ymax=ymax)
            if overlay is not None:
                line_styles = ["solid", "dotted", "dashed"]
                for j, name in enumerate(overlay):
                    vals = overlay[name]
                    if isinstance(vals[i][0][0], (float, int)):
                        pl.plot(vals[i][0], vals[i][1], color="#000000", linestyle=line_styles[j], label=name)
            if i == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel("")
                ax.set_yticks([])
                ax.spines['left'].set_visible(False)
        if overlay is not None:
            pl.legend()
        if show:
            pl.show()
        return

    if len(shap_values.shape) != 1:
        raise Exception("The passed Explanation object has multiple columns, please pass a single feature column to " + \
                        "shap.plots.dependence like: shap_values[:,column]")

    # this unpacks the explanation object for the code that was written earlier
    feature_names = [shap_values.feature_names]
    ind = 0
    shap_values_arr = shap_values.values.reshape(-1, 1)
    features = shap_values.data.reshape(-1, 1)
    if shap_values.display_data is None:
        display_features = features
    else:
        display_features = shap_values.display_data.reshape(-1, 1)
    interaction_index = None

    # unwrap explanation objects used for bounds
    if issubclass(type(xmin), Explanation):
        xmin = xmin.data
    if issubclass(type(xmax), Explanation):
        xmax = xmax.data
    if issubclass(type(ymin), Explanation):
        ymin = ymin.values
    if issubclass(type(ymax), Explanation):
        ymax = ymax.values

    # wrap np.arrays as Explanations
    if isinstance(color, np.ndarray):
        color = Explanation(values=color, base_values=None, data=color)
    
    # TODO: This stacking could be avoided if we use the new shap.utils.potential_interactions function
    if str(type(color)).endswith("Explanation'>"):
        shap_values2 = color
        if issubclass(type(shap_values2.feature_names), (str, int)):
            feature_names.append(shap_values2.feature_names)
            shap_values_arr = np.hstack([shap_values_arr, shap_values2.values.reshape(-1, len(feature_names)-1)])
            features = np.hstack([features, shap_values2.data.reshape(-1, len(feature_names)-1)])
            if shap_values2.display_data is None:
                display_features = np.hstack([display_features, shap_values2.data.reshape(-1, len(feature_names)-1)])
            else:
                display_features = np.hstack([display_features, shap_values2.display_data.reshape(-1, len(feature_names)-1)])
        else:
            feature_names2 = np.array(shap_values2.feature_names)
            mask = ~(feature_names[0] == feature_names2)
            feature_names.extend(feature_names2[mask])
            shap_values_arr = np.hstack([shap_values_arr, shap_values2.values[:,mask]])
            features = np.hstack([features, shap_values2.data[:,mask]])
            if shap_values2.display_data is None:
                display_features = np.hstack([display_features, shap_values2.data[:,mask]])
            else:
                display_features = np.hstack([display_features, shap_values2.display_data[:,mask]])
        color = None
        interaction_index = "auto"


    if type(shap_values_arr) is list:
        raise TypeError("The passed shap_values_arr are a list not an array! If you have a list of explanations try " \
                        "passing shap_values_arr[0] instead to explain the first output class of a multi-output model.")

    # convert from DataFrames if we got any
    if str(type(features)).endswith("'pandas.core.frame.DataFrame'>"):
        if feature_names is None:
            feature_names = features.columns
        features = features.values

    if feature_names is None:
        feature_names = [labels['FEATURE'] % str(i) for i in range(shap_values_arr.shape[1])]

    # allow vectors to be passed
    if len(shap_values_arr.shape) == 1:
        shap_values_arr = np.reshape(shap_values_arr, len(shap_values_arr), 1)
    if len(features.shape) == 1:
        features = np.reshape(features, len(features), 1)

    ind = convert_name(ind, shap_values_arr, feature_names)

    # pick jitter for categorical features
    vals = np.sort(np.unique(features[:,ind]))
    min_dist = np.inf
    for i in range(1,len(vals)):
        d = vals[i] - vals[i-1]
        if d > 1e-8 and d < min_dist:
            min_dist = d
    num_points_per_value = len(features[:,ind]) / len(vals)
    if num_points_per_value < 10:
        #categorical = False
        if x_jitter == "auto":
            x_jitter = 0
    elif num_points_per_value < 100:
        #categorical = True
        if x_jitter == "auto":
            x_jitter = min_dist * 0.1
    else:
        #categorical = True
        if x_jitter == "auto":
            x_jitter = min_dist * 0.2

    # guess what other feature as the stongest interaction with the plotted feature
    if not hasattr(ind, "__len__"):
        if interaction_index == "auto":
            interaction_index = approximate_interactions(ind, shap_values_arr, features)[0]
        interaction_index = convert_name(interaction_index, shap_values_arr, feature_names)
    categorical_interaction = False

    # create a matplotlib figure, if `ax` hasn't been specified.
    if not ax:
        figsize = (7.5, 5) if interaction_index != ind and interaction_index is not None else (6, 5)
        fig = pl.figure(figsize=figsize)
        ax = fig.gca()
    else:
        fig = ax.get_figure()

    # plotting SHAP interaction values
    if len(shap_values_arr.shape) == 3 and hasattr(ind, "__len__") and len(ind) == 2:
        ind1 = convert_name(ind[0], shap_values_arr, feature_names)
        ind2 = convert_name(ind[1], shap_values_arr, feature_names)
        if ind1 == ind2:
            proj_shap_values_arr = shap_values_arr[:, ind2, :]
        else:
            proj_shap_values_arr = shap_values_arr[:, ind2, :] * 2  # off-diag values are split in half

        # there is no interaction coloring for the main effect
        if ind1 == ind2:
            fig.set_size_inches(6, 5, forward=True)

        # TODO: remove recursion; generally the functions should be shorter for more maintainable code
        dependence_legacy(
            ind1, proj_shap_values_arr, features, feature_names=feature_names,
            interaction_index=(None if ind1 == ind2 else ind2), display_features=display_features, ax=ax, show=False,
            xmin=xmin, xmax=xmax, x_jitter=x_jitter, alpha=alpha
        )
        if ind1 == ind2:
            ax.set_ylabel(labels['MAIN_EFFECT'] % feature_names[ind1])
        else:
            ax.set_ylabel(labels['INTERACTION_EFFECT'] % (feature_names[ind1], feature_names[ind2]))

        if show:
            pl.show()
        return

    assert shap_values_arr.shape[0] == features.shape[0], \
        "'shap_values_arr' and 'features' values must have the same number of rows!"
    assert shap_values_arr.shape[1] == features.shape[1], \
        "'shap_values_arr' must have the same number of columns as 'features'!"

    # get both the raw and display feature values
    oinds = np.arange(shap_values_arr.shape[0]) # we randomize the ordering so plotting overlaps are not related to data ordering
    np.random.shuffle(oinds)
    xv = encode_array_if_needed(features[oinds, ind])
    xd = display_features[oinds, ind]
    
    s = shap_values_arr[oinds, ind]
    if type(xd[0]) == str:
        name_map = {}
        for i in range(len(xv)):
            name_map[xd[i]] = xv[i]
        xnames = list(name_map.keys())
    
    # allow a single feature name to be passed alone
    if type(feature_names) == str:
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
        if type(cd[0]) == str:
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
            bounds = np.linspace(clow, chigh, min(int(chigh - clow + 2), cmap.N-1))
            color_norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N-1)

    # optionally add jitter to feature values
    xv_no_jitter = xv.copy()
    if x_jitter > 0:
        if x_jitter > 1: x_jitter = 1
        xvals = xv.copy()
        if isinstance(xvals[0], float):
            xvals = xvals.astype(float)
            xvals = xvals[~np.isnan(xvals)]
        xvals = np.unique(xvals) # returns a sorted array
        if len(xvals) >= 2:
            smallest_diff = np.min(np.diff(xvals))
            jitter_amount = x_jitter * smallest_diff
            xv += (np.random.random_sample(size = len(xv))*jitter_amount) - (jitter_amount/2)

    
    # the actual scatter plot, TODO: adapt the dot_size to the number of data points?
    xv_nan = np.isnan(xv)
    xv_notnan = np.invert(xv_nan)
    if interaction_index is not None:

        # plot the nan values in the interaction feature as grey
        cvals = features[oinds, interaction_index].astype(np.float64)
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
            xv[xv_notnan], s[xv_notnan], s=dot_size, linewidth=0, c=cvals[xv_notnan],
            cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax,
            norm=color_norm, rasterized=len(xv) > 500
        )
        p.set_array(cvals[xv_notnan])
    else:
        p = ax.scatter(xv, s, s=dot_size, linewidth=0, color=color,
                       alpha=alpha, rasterized=len(xv) > 500)

    if interaction_index != ind and interaction_index is not None:
        # draw the color bar
        if type(cd[0]) == str:
            tick_positions = np.array([cname_map[n] for n in cnames])
            tick_positions *= 1 - 1 / len(cnames)
            tick_positions += 0.5 * (chigh - clow) / (chigh - clow + 1)
            cb = pl.colorbar(p, ticks=tick_positions, ax=ax)
            cb.set_ticklabels(cnames)
        else:
            cb = pl.colorbar(p, ax=ax)

        cb.set_label(feature_names[interaction_index], size=13)
        cb.ax.tick_params(labelsize=11)
        if categorical_interaction:
            cb.ax.tick_params(length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
        bbox = cb.ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        cb.ax.set_aspect((bbox.height - 0.7) * 20)

    # handles any setting of xmax and xmin
    # note that we handle None,float, or "percentile(float)" formats
    if xmin is not None or xmax is not None:
        if type(xmin) == str and xmin.startswith("percentile"):
            xmin = np.nanpercentile(xv, float(xmin[11:-1]))
        if type(xmax) == str and xmax.startswith("percentile"):
            xmax = np.nanpercentile(xv, float(xmax[11:-1]))

        if xmin is None or xmin == np.nanmin(xv):
            xmin = np.nanmin(xv) - (xmax - np.nanmin(xv))/20
        if xmax is None or xmax == np.nanmax(xv):
            xmax = np.nanmax(xv) + (np.nanmax(xv) - xmin)/20

        ax.set_xlim(xmin, xmax)

    if ymin is not None or ymax is not None:
        # if type(ymin) == str and ymin.startswith("percentile"):
        #     ymin = np.nanpercentile(xv, float(ymin[11:-1]))
        # if type(ymax) == str and ymax.startswith("percentile"):
        #     ymax = np.nanpercentile(xv, float(ymax[11:-1]))

        if ymin is None or ymin == np.nanmin(xv):
            ymin = np.nanmin(xv) - (ymax - np.nanmin(xv))/20
        if ymax is None or ymax == np.nanmax(xv):
            ymax = np.nanmax(xv) + (np.nanmax(xv) - ymin)/20

        ax.set_ylim(ymin, ymax)

    # plot any nan feature values as tick marks along the y-axis
    xlim = ax.get_xlim()
    if interaction_index is not None:
        p = ax.scatter(
            xlim[0] * np.ones(xv_nan.sum()), s[xv_nan], marker=1,
            linewidth=2, c=cvals_imp[xv_nan], cmap=cmap, alpha=alpha,
            vmin=clow, vmax=chigh
        )
        p.set_array(cvals[xv_nan])
    else:
        ax.scatter(
            xlim[0] * np.ones(xv_nan.sum()), s[xv_nan], marker=1,
            linewidth=2, color=color, alpha=alpha
        )
    ax.set_xlim(xlim)

    # the histogram of the data
    if hist:
        ax2 = ax.twinx()
        #n, bins, patches = 
        xlim = ax.get_xlim()
        xvals = np.unique(xv_no_jitter)

        if len(xvals) / len(xv_no_jitter) < 0.2 and len(xvals) < 75 and np.max(xvals) < 75 and np.min(xvals) >= 0:
            np.sort(xvals)
            bin_edges = []
            for i in range(int(np.max(xvals)+1)):
                bin_edges.append(i-0.5)

                #bin_edges.append((xvals[i] + xvals[i+1])/2)
            bin_edges.append(int(np.max(xvals))+0.5)

            lim = np.floor(np.min(xvals) - 0.5) + 0.5, np.ceil(np.max(xvals) + 0.5) - 0.5
            ax.set_xlim(lim)
        else:
            if len(xv_no_jitter) >= 500:
                bin_edges = 50
            elif len(xv_no_jitter) >= 200:
                bin_edges = 20
            elif len(xv_no_jitter) >= 100:
                bin_edges = 10
            else:
                bin_edges = 5
        
        ax2.hist(xv[~np.isnan(xv)], bin_edges, density=False, facecolor='#000000', alpha=0.1, range=(xlim[0], xlim[1]), zorder=-1)
        ax2.set_ylim(0,len(xv))

        ax2.xaxis.set_ticks_position('bottom')
        ax2.yaxis.set_ticks_position('left')
        ax2.yaxis.set_ticks([])
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)

    pl.sca(ax)

    # make the plot more readable
    ax.set_xlabel(name, color=axis_color, fontsize=13)
    ax.set_ylabel(labels['VALUE_FOR'] % name, color=axis_color, fontsize=13)
    if title is not None:
        ax.set_title(title, color=axis_color, fontsize=13)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color, labelsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor(axis_color)
    if type(xd[0]) == str:
        ax.set_xticks([name_map[n] for n in xnames])
        ax.set_xticklabels(xnames, dict(rotation='vertical', fontsize=11))
    if show:
        with warnings.catch_warnings(): # ignore expected matplotlib warnings
            warnings.simplefilter("ignore", RuntimeWarning)
            pl.show()




def dependence_legacy(ind, shap_values=None, features=None, feature_names=None, display_features=None,
                      interaction_index="auto",
                      color="#1E88E5", axis_color="#333333", cmap=None,
                      dot_size=16, x_jitter=0, alpha=1, title=None, xmin=None, xmax=None, ax=None, show=True):
    """ Create a SHAP dependence plot, colored by an interaction feature.

    Plots the value of the feature on the x-axis and the SHAP value of the same feature
    on the y-axis. This shows how the model depends on the given feature, and is like a
    richer extenstion of the classical parital dependence plots. Vertical dispersion of the
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

    """

    if cmap is None:
        cmap = colors.red_blue

    if type(shap_values) is list:
        raise TypeError("The passed shap_values are a list not an array! If you have a list of explanations try " \
                        "passing shap_values[0] instead to explain the first output class of a multi-output model.")

    # convert from DataFrames if we got any
    if str(type(features)).endswith("'pandas.core.frame.DataFrame'>"):
        if feature_names is None:
            feature_names = features.columns
        features = features.values
    if str(type(display_features)).endswith("'pandas.core.frame.DataFrame'>"):
        if feature_names is None:
            feature_names = display_features.columns
        display_features = display_features.values
    elif display_features is None:
        display_features = features

    if feature_names is None:
        feature_names = [labels['FEATURE'] % str(i) for i in range(shap_values.shape[1])]

    # allow vectors to be passed
    if len(shap_values.shape) == 1:
        shap_values = np.reshape(shap_values, len(shap_values), 1)
    if len(features.shape) == 1:
        features = np.reshape(features, len(features), 1)

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
        fig = pl.figure(figsize=figsize)
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
            ind1, proj_shap_values, features, feature_names=feature_names,
            interaction_index=(None if ind1 == ind2 else ind2), display_features=display_features, ax=ax, show=False,
            xmin=xmin, xmax=xmax, x_jitter=x_jitter, alpha=alpha
        )
        if ind1 == ind2:
            ax.set_ylabel(labels['MAIN_EFFECT'] % feature_names[ind1])
        else:
            ax.set_ylabel(labels['INTERACTION_EFFECT'] % (feature_names[ind1], feature_names[ind2]))

        if show:
            pl.show()
        return

    assert shap_values.shape[0] == features.shape[0], \
        "'shap_values' and 'features' values must have the same number of rows!"
    assert shap_values.shape[1] == features.shape[1], \
        "'shap_values' must have the same number of columns as 'features'!"

    # get both the raw and display feature values
    oinds = np.arange(shap_values.shape[0]) # we randomize the ordering so plotting overlaps are not related to data ordering
    np.random.shuffle(oinds)
    
    xv = encode_array_if_needed(features[oinds, ind])

    xd = display_features[oinds, ind]
    s = shap_values[oinds, ind]
    if type(xd[0]) == str:
        name_map = {}
        for i in range(len(xv)):
            name_map[xd[i]] = xv[i]
        xnames = list(name_map.keys())

    # allow a single feature name to be passed alone
    if type(feature_names) == str:
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
        if type(cd[0]) == str:
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
            bounds = np.linspace(clow, chigh, min(int(chigh - clow + 2), cmap.N-1))
            color_norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N-1)

    # optionally add jitter to feature values
    if x_jitter > 0:
        if x_jitter > 1: x_jitter = 1
        xvals = xv.copy()
        if isinstance(xvals[0], float):
            xvals = xvals.astype(float)
            xvals = xvals[~np.isnan(xvals)]
        xvals = np.unique(xvals) # returns a sorted array
        if len(xvals) >= 2:
            smallest_diff = np.min(np.diff(xvals))
            jitter_amount = x_jitter * smallest_diff
            xv += (np.random.random_sample(size = len(xv))*jitter_amount) - (jitter_amount/2)

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
            xv[xv_notnan], s[xv_notnan], s=dot_size, linewidth=0, c=cvals[xv_notnan],
            cmap=cmap, alpha=alpha, vmin=clow, vmax=chigh,
            norm=color_norm, rasterized=len(xv) > 500
        )
        p.set_array(cvals[xv_notnan])
    else:
        p = ax.scatter(xv, s, s=dot_size, linewidth=0, color=color,
                       alpha=alpha, rasterized=len(xv) > 500)

    if interaction_index != ind and interaction_index is not None:
        # draw the color bar
        if type(cd[0]) == str:
            tick_positions = [cname_map[n] for n in cnames]
            if len(tick_positions) == 2:
                tick_positions[0] -= 0.25
                tick_positions[1] += 0.25
            cb = pl.colorbar(p, ticks=tick_positions, ax=ax)
            cb.set_ticklabels(cnames)
        else:
            cb = pl.colorbar(p, ax=ax)

        cb.set_label(feature_names[interaction_index], size=13)
        cb.ax.tick_params(labelsize=11)
        if categorical_interaction:
            cb.ax.tick_params(length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
        bbox = cb.ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        cb.ax.set_aspect((bbox.height - 0.7) * 20)

    # handles any setting of xmax and xmin
    # note that we handle None,float, or "percentile(float)" formats
    if xmin is not None or xmax is not None:
        if type(xmin) == str and xmin.startswith("percentile"):
            xmin = np.nanpercentile(xv, float(xmin[11:-1]))
        if type(xmax) == str and xmax.startswith("percentile"):
            xmax = np.nanpercentile(xv, float(xmax[11:-1]))

        if xmin is None or xmin == np.nanmin(xv):
            xmin = np.nanmin(xv) - (xmax - np.nanmin(xv))/20
        if xmax is None or xmax == np.nanmax(xv):
            xmax = np.nanmax(xv) + (np.nanmax(xv) - xmin)/20

        ax.set_xlim(xmin, xmax)

    # plot any nan feature values as tick marks along the y-axis
    xlim = ax.get_xlim()
    if interaction_index is not None:
        p = ax.scatter(
            xlim[0] * np.ones(xv_nan.sum()), s[xv_nan], marker=1,
            linewidth=2, c=cvals_imp[xv_nan], cmap=cmap, alpha=alpha,
            vmin=clow, vmax=chigh
        )
        p.set_array(cvals[xv_nan])
    else:
        ax.scatter(
            xlim[0] * np.ones(xv_nan.sum()), s[xv_nan], marker=1,
            linewidth=2, color=color, alpha=alpha
        )
    ax.set_xlim(xlim)

    # make the plot more readable
    ax.set_xlabel(name, color=axis_color, fontsize=13)
    ax.set_ylabel(labels['VALUE_FOR'] % name, color=axis_color, fontsize=13)
    if title is not None:
        ax.set_title(title, color=axis_color, fontsize=13)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color, labelsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor(axis_color)
    if type(xd[0]) == str:
        ax.set_xticks([name_map[n] for n in xnames])
        ax.set_xticklabels(xnames, dict(rotation='vertical', fontsize=11))
    if show:
        with warnings.catch_warnings(): # ignore expected matplotlib warnings
            warnings.simplefilter("ignore", RuntimeWarning)
            pl.show()
