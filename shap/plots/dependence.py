import numpy as np
try:
    import matplotlib.pyplot as pl
    import matplotlib
except ImportError:
    pass
from . import labels
from . import colors

# TODO: remove color argument / use color argument
def dependence_plot(ind, shap_values, features, feature_names=None, display_features=None,
                    interaction_index="auto", color="#1E88E5", axis_color="#333333",
                    dot_size=16, x_jitter=0, alpha=1, title=None, show=True):
    """
    Create a SHAP dependence plot, colored by an interaction feature.

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

    display_features : numpy.array or pandas.DataFrame
        Matrix of feature values for visual display (such as strings instead of coded values)

    interaction_index : "auto", None, or int
        The index of the feature used to color the plot.
        
    x_jitter : float (0 - 1)
        Adds random jitter to feature values. 
        May increase plot readability when feature is non-continuous.
    """

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

    def convert_name(ind):
        if type(ind) == str:
            nzinds = np.where(feature_names == ind)[0]
            if len(nzinds) == 0:
                print("Could not find feature named: " + ind)
                return None
            else:
                return nzinds[0]
        else:
            return ind

    ind = convert_name(ind)

    # plotting SHAP interaction values
    if len(shap_values.shape) == 3 and len(ind) == 2:
        ind1 = convert_name(ind[0])
        ind2 = convert_name(ind[1])
        if ind1 == ind2:
            proj_shap_values = shap_values[:, ind2, :]
        else:
            proj_shap_values = shap_values[:, ind2, :] * 2  # off-diag values are split in half

        # TODO: remove recursion; generally the functions should be shorter for more maintainable code
        dependence_plot(
            ind1, proj_shap_values, features, feature_names=feature_names,
            interaction_index=ind2, display_features=display_features, show=False
        )
        if ind1 == ind2:
            pl.ylabel(labels['MAIN_EFFECT'] % feature_names[ind1])
        else:
            pl.ylabel(labels['INTERACTION_EFFECT'] % (feature_names[ind1], feature_names[ind2]))

        if show:
            pl.show()
        return

    assert shap_values.shape[0] == features.shape[0], \
        "'shap_values' and 'features' values must have the same number of rows!"
    assert shap_values.shape[1] == features.shape[1], \
        "'shap_values' must have the same number of columns as 'features'!"

    # get both the raw and display feature values
    xv = features[:, ind].astype(np.float64)
    xd = display_features[:, ind]
    s = shap_values[:, ind]
    if type(xd[0]) == str:
        name_map = {}
        for i in range(len(xv)):
            name_map[xd[i]] = xv[i]
        xnames = list(name_map.keys())

    # allow a single feature name to be passed alone
    if type(feature_names) == str:
        feature_names = [feature_names]
    name = feature_names[ind]

    # guess what other feature as the stongest interaction with the plotted feature
    if interaction_index == "auto":
        interaction_index = approx_interactions(ind, shap_values, features)[0]
    interaction_index = convert_name(interaction_index)
    categorical_interaction = False

    # get both the raw and display color values
    if interaction_index is not None:
        cv = features[:, interaction_index]
        cd = display_features[:, interaction_index]
        clow = np.nanpercentile(features[:, interaction_index].astype(np.float), 5)
        chigh = np.nanpercentile(features[:, interaction_index].astype(np.float), 95)
        if type(cd[0]) == str:
            cname_map = {}
            for i in range(len(cv)):
                cname_map[cd[i]] = cv[i]
            cnames = list(cname_map.keys())
            categorical_interaction = True
        elif clow % 1 == 0 and chigh % 1 == 0 and len(set(features[:, interaction_index])) < 50:
            categorical_interaction = True

    # discritize colors for categorical features
    color_norm = None
    if categorical_interaction and clow != chigh:
        bounds = np.linspace(clow, chigh, chigh - clow + 2)
        color_norm = matplotlib.colors.BoundaryNorm(bounds, colors.red_blue.N)
        
    # optionally add jitter to feature values
    if x_jitter > 0:
        if x_jitter > 1: x_jitter = 1
        xvals = xv.copy()
        if isinstance(xvals[0], float):
            xvals = xvals.astype(np.float)
            xvals = xvals[~np.isnan(xvals)]
        xvals = np.unique(xvals)
        if len(xvals) >= 2:
            smallest_diff = np.min(np.diff(np.sort(xvals)))
            jitter_amount = x_jitter * smallest_diff
            xv += (np.random.ranf(size = len(xv))*jitter_amount) - (jitter_amount/2)

    # the actual scatter plot, TODO: adapt the dot_size to the number of data points?
    if interaction_index is not None:

        # plot the nan values in the interaction feature as grey
        cvals = features[:, interaction_index].astype(np.float64)
        cvals_nans = np.isnan(cvals)
        pl.scatter(xv[cvals_nans], s[cvals_nans], s=dot_size, linewidth=0, color="#777777",
                   alpha=alpha, norm=color_norm, rasterized=len(xv) > 500)
        
        
        pl.scatter(xv[np.invert(cvals_nans)], s[np.invert(cvals_nans)], s=dot_size, linewidth=0, c=cvals[np.invert(cvals_nans)], cmap=colors.red_blue,
                   alpha=alpha, vmin=clow, vmax=chigh, norm=color_norm, rasterized=len(xv) > 500)
    else:
        pl.scatter(xv, s, s=dot_size, linewidth=0, color="#1E88E5",
                   alpha=alpha, rasterized=len(xv) > 500)
    
    if interaction_index != ind and interaction_index is not None:
        # draw the color bar
        if type(cd[0]) == str:
            tick_positions = [cname_map[n] for n in cnames]
            if len(tick_positions) == 2:
                tick_positions[0] -= 0.25
                tick_positions[1] += 0.25
            cb = pl.colorbar(ticks=tick_positions)
            cb.set_ticklabels(cnames)
        else:
            cb = pl.colorbar()

        cb.set_label(feature_names[interaction_index], size=13)
        cb.ax.tick_params(labelsize=11)
        if categorical_interaction:
            cb.ax.tick_params(length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
        bbox = cb.ax.get_window_extent().transformed(pl.gcf().dpi_scale_trans.inverted())
        cb.ax.set_aspect((bbox.height - 0.7) * 20)

    # plot any nan feature values as tick marks along the y-axis
    xv_nans = np.isnan(xv)
    xlim = pl.xlim()
    pl.scatter(xlim[0] * np.ones(xv_nans.sum()), s[xv_nans], marker=1, linewidth=2, color="#777777")
    pl.xlim(*xlim)

    # make the plot more readable
    if interaction_index != ind:
        pl.gcf().set_size_inches(7.5, 5)
    else:
        pl.gcf().set_size_inches(6, 5)
    pl.xlabel(name, color=axis_color, fontsize=13)
    pl.ylabel(labels['VALUE_FOR'] % name, color=axis_color, fontsize=13)
    if title is not None:
        pl.title(title, color=axis_color, fontsize=13)
    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('left')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    pl.gca().tick_params(color=axis_color, labelcolor=axis_color, labelsize=11)
    for spine in pl.gca().spines.values():
        spine.set_edgecolor(axis_color)
    if type(xd[0]) == str:
        pl.xticks([name_map[n] for n in xnames], xnames, rotation='vertical', fontsize=11)
    if show:
        pl.show()


def approx_interactions(index, shap_values, X):
    """ Order other features by how much interaction they seem to have with the feature at the given index.

    This just bins the SHAP values for a feature along that feature's value. For true Shapley interaction
    index values for SHAP see the interaction_contribs option implemented in XGBoost.
    """

    if X.shape[0] > 10000:
        a = np.arange(X.shape[0])
        np.random.shuffle(a)
        inds = a[:10000]
    else:
        inds = np.arange(X.shape[0])

    x = X[inds, index]
    srt = np.argsort(x)
    shap_ref = shap_values[inds, index]
    shap_ref = shap_ref[srt]
    inc = max(min(int(len(x) / 10.0), 50), 1)
    interactions = []
    for i in range(X.shape[1]):
        val_other = X[inds, i][srt].astype(np.float)
        v = 0.0
        if not (i == index or np.sum(np.abs(val_other)) < 1e-8):
            for j in range(0, len(x), inc):
                if np.std(val_other[j:j + inc]) > 0 and np.std(shap_ref[j:j + inc]) > 0:
                    v += abs(np.corrcoef(shap_ref[j:j + inc], val_other[j:j + inc])[0, 1])
        interactions.append(v)

    return np.argsort(-np.abs(interactions))
