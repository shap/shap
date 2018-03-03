import warnings
from scipy.stats import gaussian_kde
from iml import Instance, Model, visualize
from iml.explanations import AdditiveExplanation
from iml.links import IdentityLink
from iml.datatypes import DenseData
import iml
import numpy as np

try:
    import matplotlib.pyplot as pl
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import MaxNLocator

    cdict1 = {
          'red': ((0.0, 0.11764705882352941, 0.11764705882352941),
                  (1.0, 0.9607843137254902, 0.9607843137254902)),

        'green': ((0.0, 0.5333333333333333, 0.5333333333333333),
                   (1.0, 0.15294117647058825, 0.15294117647058825)),

         'blue':  ((0.0, 0.8980392156862745, 0.8980392156862745),
                   (1.0, 0.3411764705882353, 0.3411764705882353)),

        'alpha':  ((0.0, 1, 1),
                   (0.5, 0.3, 0.3),
                   (1.0, 1, 1))
    } # #1E88E5 -> #ff0052
    red_blue = LinearSegmentedColormap('RedBlue', cdict1)
except ImportError:
    pass

def dependence_plot(ind, shap_values, features, feature_names=None, display_features=None,
                    interaction_index="auto", color="#1E88E5", axis_color="#333333",
                    dot_size=16, alpha=1, title=None, show=True):
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
    """

    # convert from DataFrames if we got any
    if str(type(features)) == "<class 'pandas.core.frame.DataFrame'>":
        if feature_names is None:
            feature_names = features.columns
        features = features.as_matrix()
    if str(type(display_features)) == "<class 'pandas.core.frame.DataFrame'>":
        if feature_names is None:
            feature_names = display_features.columns
        display_features = display_features.as_matrix()
    elif display_features is None:
        display_features = features

    if feature_names is None:
        feature_names = ["Feature "+str(i) for i in range(shap_values.shape[1]-1)]

    # allow vectors to be passed
    if len(shap_values.shape) == 1:
        shap_values = np.reshape(shap_values, len(shap_values), 1)
    if len(features.shape) == 1:
        features = np.reshape(features, len(features), 1)

    def convert_name(ind):
        if type(ind) == str:
            nzinds = np.where(feature_names == ind)[0]
            if len(nzinds) == 0:
                print("Could not find feature named: "+ind)
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
            proj_shap_values = shap_values[:,ind2,:]
        else:
            proj_shap_values = shap_values[:,ind2,:] * 2 # off-diag values are split in half
        dependence_plot(
            ind1, proj_shap_values, features, feature_names=feature_names,
            interaction_index=ind2, display_features=display_features, show=False
        )
        if ind1 == ind2:
            pl.ylabel("SHAP main effect value for\n"+feature_names[ind1])
        else:
            pl.ylabel("SHAP interaction value for\n"+feature_names[ind1]+" and "+feature_names[ind2])

        if show:
            pl.show()
        return

    # get both the raw and display feature values
    xv = features[:,ind]
    xd = display_features[:,ind]
    s = shap_values[:,ind]
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

    # get both the raw and display color values
    cv = features[:,interaction_index]
    cd = display_features[:,interaction_index]
    categorical_interaction = False
    clow = np.nanpercentile(features[:,interaction_index], 5)
    chigh = np.nanpercentile(features[:,interaction_index], 95)
    if type(cd[0]) == str:
        cname_map = {}
        for i in range(len(cv)):
            cname_map[cd[i]] = cv[i]
        cnames = list(cname_map.keys())
        categorical_interaction = True
    elif clow % 1 == 0 and chigh % 1 == 0 and len(set(features[:,interaction_index])) < 50:
        categorical_interaction = True

    # discritize colors for categorical features
    color_norm = None
    if categorical_interaction and clow != chigh:
        bounds = np.linspace(clow, chigh, chigh-clow+2)
        color_norm = matplotlib.colors.BoundaryNorm(bounds, red_blue.N)

    # the actual scatter plot, TODO: adapt the dot_size to the number of data points?
    pl.scatter(xv, s, s=dot_size, linewidth=0, c=features[:,interaction_index], cmap=red_blue,
               alpha=alpha, vmin=clow, vmax=chigh, norm=color_norm, rasterized=len(xv) > 500)

    if interaction_index != ind:
        # draw the color bar
        norm = None
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
        cb.ax.set_aspect((bbox.height-0.7)*20)

    # make the plot more readable
    if interaction_index != ind:
        pl.gcf().set_size_inches(7.5, 5)
    else:
        pl.gcf().set_size_inches(6, 5)
    pl.xlabel(name, color=axis_color, fontsize=13)
    pl.ylabel("SHAP value for\n"+name, color=axis_color, fontsize=13)
    if title != None:
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

    x = X[inds,index]
    srt = np.argsort(x)
    shap_ref = shap_values[inds,index]
    shap_ref = shap_ref[srt]
    inc = min(int(len(x)/10.0), 50)
    interactions = []
    for i in range(X.shape[1]):
        val_other = X[inds,i][srt]

        if i == index or np.sum(np.abs(val_other)) < 1e-8:
            v = 0
        else:
            v = 0.0
            for i in range(0,len(x),inc):
                if np.std(val_other[i:i+inc]) > 0 and np.std(shap_ref[i:i+inc]) > 0:
                    v += abs(np.corrcoef(shap_ref[i:i+inc],val_other[i:i+inc])[0,1])
        interactions.append(v)

    return np.argsort(-np.abs(interactions))

def summary_plot(shap_values, features=None, feature_names=None, max_display=None, plot_type="dot",
                 color="#ff0052", axis_color="#333333", title=None, alpha=1, show=True, sort=True,
                 color_bar=True, auto_size_plot=True):
    """
    Create a SHAP summary plot, colored by feature values when they are provided.

    Parameters
    ----------
    shap_values : numpy.array
        Matrix of SHAP values (# samples x # features)

    features : numpy.array or pandas.DataFrame or list
        Matrix of feature values (# samples x # features) or a feature_names list as shorthand

    feature_names : list
        Names of the features (length # features)

    max_display : int
        How many top features to include in the plot (default is 20, or 7 for interaction plots)

    plot_type : "dot" (default) or "violin"
        What type of summary plot to produce
    """

    if len(shap_values.shape) == 1:
        assert False, "Summary plots need a matrix of shap_values, not a vector."

    # convert from a DataFrame or other types
    if str(type(features)) == "<class 'pandas.core.frame.DataFrame'>":
        if feature_names is None:
            feature_names = features.columns
        features = features.as_matrix()
    elif str(type(features)) == "<class 'list'>":
        if feature_names is None:
            feature_names = features
        features = None
    elif (features is not None) and len(features.shape) == 1 and feature_names is None:
        feature_names = features
        features = None

    if feature_names is None:
        feature_names = ["Feature "+str(i) for i in range(shap_values.shape[1]-1)]

    # plotting SHAP interaction values
    if len(shap_values.shape) == 3:
        if max_display is None:
            max_display = 7
        else:
            max_display = min(len(feature_names), max_display)

        sort_inds = np.argsort(-np.abs(shap_values[:,:-1,:-1].sum(1)).sum(0))

        # get plotting limits
        delta = 1.0 / (shap_values.shape[1]**2)
        slow = np.nanpercentile(shap_values, delta)
        shigh = np.nanpercentile(shap_values, 100 - delta)
        v = max(abs(slow), abs(shigh))
        slow = -v
        shigh = v

        pl.figure(figsize=(1.5*max_display+1,1*max_display+1))
        pl.subplot(1,max_display,1)
        proj_shap_values = shap_values[:,sort_inds[0],np.hstack((sort_inds, len(sort_inds)))]
        proj_shap_values[:,1:] *= 2 # because off diag effects are split in half
        summary_plot(
            proj_shap_values, features[:,sort_inds],
            feature_names=feature_names[sort_inds],
            sort=False, show=False, color_bar=False,
            auto_size_plot=False,
            max_display=max_display
        )
        pl.xlim((slow,shigh))
        pl.xlabel("")
        title_length_limit = 11
        pl.title(shorten_text(feature_names[sort_inds[0]], title_length_limit))
        for i in range(1,max_display):
            ind = sort_inds[i]
            pl.subplot(1,max_display,i+1)
            proj_shap_values = shap_values[:,ind,np.hstack((sort_inds, len(sort_inds)))]
            proj_shap_values *= 2
            proj_shap_values[:,i] /= 2 # because only off diag effects are split in half
            summary_plot(
                proj_shap_values, features[:,sort_inds],
                sort=False,
                feature_names=["" for i in range(features.shape[1])],
                show=False,
                color_bar=False,
                auto_size_plot=False,
                max_display=max_display
            )
            pl.xlim((slow,shigh))
            pl.xlabel("")
            if i == max_display//2:
                pl.xlabel("SHAP interaction value")
            pl.title(shorten_text(feature_names[ind], title_length_limit))
        pl.tight_layout(pad=0, w_pad=0, h_pad=0.0)
        pl.subplots_adjust(hspace=0, wspace=0.1)
        if show:
            pl.show()
        return

    if max_display is None:
        max_display = 20

    if sort:
        # order features by the sum of their effect magnitudes
        feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0)[:-1])
        feature_order = feature_order[-min(max_display,len(feature_order)):]
    else:
        feature_order = np.flip(np.arange(min(max_display,shap_values.shape[1]-1)),0)

    row_height = 0.4
    if auto_size_plot:
        pl.gcf().set_size_inches(8, len(feature_order)*row_height+1.5)
    pl.axvline(x=0, color="#999999", zorder=-1)

    if plot_type == "dot":
        for pos,i in enumerate(feature_order):
            pl.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1,5), zorder=-1)
            shaps = shap_values[:,i]
            N = len(shaps)
            hspacing = (np.max(shaps) - np.min(shaps))/200
            curr_bin = []
            nbins = 100
            quant = np.round(nbins*(shap_values[:,i] - np.min(shaps))/(np.max(shaps)-np.min(shaps)+1e-8))
            inds = np.argsort(quant+np.random.randn(N)*1e-6)
            layer = 0
            last_bin = -1
            ys = np.zeros(N)
            for ind in inds:
                if quant[ind] != last_bin:
                    layer = 0
                ys[ind] = np.ceil(layer/2) * ((layer%2)*2-1)
                layer += 1
                last_bin = quant[ind]
            ys *= 0.9*(row_height/np.max(ys+1))

            if features is not None:

                # trim the color range, but prevent the color range from collapsing
                vmin = np.nanpercentile(features[:,i], 5)
                vmax = np.nanpercentile(features[:,i], 95)
                if vmin == vmax:
                    vmin = np.nanpercentile(features[:,i], 1)
                    vmax = np.nanpercentile(features[:,i], 99)
                    if vmin == vmax:
                        vmin = np.min(features[:,i])
                        vmax = np.max(features[:,i])

                assert features.shape[0] == len(shaps), "Feature and SHAP matrices must have the same number of rows!"
                pl.scatter(shaps, pos+ys, cmap=red_blue, vmin=vmin, vmax=vmax, s=16,
                           c=np.nan_to_num(features[:,i]), alpha=alpha, linewidth=0,
                           zorder=3, rasterized=len(shaps) > 500)
            else:
                pl.scatter(shaps, pos+ys, s=16, alpha=alpha, linewidth=0, zorder=3,
                           color=color, rasterized=len(shaps) > 500)

    elif plot_type == "violin":
        for pos,i in enumerate(feature_order):
            pl.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1,5), zorder=-1)

        if features is not None:
            global_low = np.nanpercentile(shap_values[:,:len(feature_names)].flatten(), 1)
            global_high = np.nanpercentile(shap_values[:,:len(feature_names)].flatten(), 99)
            for pos,i in enumerate(feature_order):
                shaps = shap_values[:,i]
                shap_min,shap_max = np.min(shaps),np.max(shaps)
                rng = shap_max-shap_min
                xs = np.linspace(np.min(shaps)-rng*0.2, np.max(shaps)+rng*0.2, 100)
                if np.std(shaps) < (global_high-global_low)/100:
                    ds = gaussian_kde(shaps + np.random.randn(len(shaps))*(global_high-global_low)/100)(xs)
                else:
                    ds = gaussian_kde(shaps)(xs)
                ds /= np.max(ds)*3

                values = features[:,i]
                window_size = max(10, len(values)//20)
                smooth_values = np.zeros(len(xs)-1)
                for j in range(len(xs)-1):
                    smooth_values[j] = np.mean(values[max(0,j-window_size):min(len(xs),j+window_size)])

                vmin = np.nanpercentile(values, 5)
                vmax = np.nanpercentile(values, 95)
                # smooth_values -= np.nanpercentile(smooth_values, 5)
                # smooth_values /= np.nanpercentile(smooth_values, 95)
                smooth_values -= vmin
                smooth_values /= vmax-vmin
                for i in range(len(xs)-1):
                    if ds[i] > 0.05 or ds[i+1] > 0.05:
                        pl.fill_between([xs[i],xs[i+1]], [pos+ds[i],pos+ds[i+1]], [pos-ds[i],pos-ds[i+1]], color=red_blue(smooth_values[i]), zorder=2)

                vmin = np.nanpercentile(values, 5)
                vmax = np.nanpercentile(values, 95)
                pl.scatter(shaps, np.ones(shap_values.shape[0])*pos, s=9, cmap=red_blue, vmin=vmin, vmax=vmax, c=values, alpha=alpha, linewidth=0, zorder=3)

        else:
            parts = pl.violinplot(shap_values[:,feature_order], range(len(feature_order)), points=200, vert=False, widths=0.7,
                              showmeans=False, showextrema=False, showmedians=False)

            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_edgecolor('none')
                pc.set_alpha(alpha)

    # draw the color bar
    if color_bar and features is not None:
        cb = pl.colorbar(ticks=[vmin,vmax], aspect=1000)
        cb.set_ticklabels(["Low", "High"])
        cb.set_label("Feature value", size=12, labelpad=0)
        cb.ax.tick_params(labelsize=11, length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
        bbox = cb.ax.get_window_extent().transformed(pl.gcf().dpi_scale_trans.inverted())
        cb.ax.set_aspect((bbox.height-0.9)*20)
        #cb.draw_all()

    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('none')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    pl.gca().spines['left'].set_visible(False)
    pl.gca().tick_params(color=axis_color, labelcolor=axis_color)
    pl.yticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=13)
    pl.gca().tick_params('y', length=20, width=0.5, which='major')
    pl.gca().tick_params('x', labelsize=11)
    pl.ylim(-1, len(feature_order))
    pl.xlabel("SHAP value (impact on model output)", fontsize=13)
    pl.tight_layout()
    if show: pl.show()

def visualize(shap_values, features=None, feature_names=None, out_names=None, data=None,
              link=IdentityLink()):

    warnings.warn("the visualize() function has been renamed to 'force_plot' for consistency")

    # backwards compatability
    if data is not None:
        warnings.warn("the 'data' parameter has been renamed to 'features' for consistency")
        if features is None:
            features = data

    return force_plot(shap_values, features, feature_names, out_names, link)

def force_plot(shap_values, features=None, feature_names=None, out_names=None, link="identity",
               plot_cmap="RdBu"):
    """ Visualize the given SHAP values with an additive force layout. """

    link = iml.links.convert_to_link(link)

    if type(shap_values) == list:
        assert False, "The shap_values arg looks looks multi output, try shap_values[i]."

    if type(shap_values) != np.ndarray:
        return iml.visualize(shap_values)

    # convert from a DataFrame or other types
    if str(type(features)) == "<class 'pandas.core.frame.DataFrame'>":
        if feature_names is None:
            feature_names = list(features.columns)
        features = features.as_matrix()
    elif str(type(features)) == "<class 'pandas.core.series.Series'>":
        if feature_names is None:
            feature_names = list(features.index)
        features = features.as_matrix()
    elif str(type(features)) == "list":
        if feature_names is None:
            feature_names = features
        features = None
    elif features is not None and len(features.shape) == 1 and feature_names is None:
        feature_names = features
        features = None

    if len(shap_values.shape) == 1:
        shap_values = np.reshape(shap_values, (1,len(shap_values)))

    if out_names is None:
        out_names = ["output value"]

    if shap_values.shape[0] == 1:
        if feature_names is None:
            feature_names = ["Feature "+str(i) for i in range(shap_values.shape[1]-1)]
        if features is None:
            features = ["" for i in range(len(feature_names))]
        if type(features) == np.ndarray:
            features = features.flatten()

        instance = Instance(np.zeros((1,len(feature_names))), features)
        e = AdditiveExplanation(
            shap_values[0,-1],
            np.sum(shap_values[0,:]),
            shap_values[0,:-1],
            None,
            instance,
            link,
            Model(None, out_names),
            DenseData(np.zeros((1,len(feature_names))), list(feature_names))
        )
        return iml.visualize(e, plot_cmap)

    else:
        exps = []
        for i in range(shap_values.shape[0]):
            if feature_names is None:
                feature_names = ["Feature "+str(i) for i in range(shap_values.shape[1]-1)]
            if features is None:
                display_features = ["" for i in range(len(feature_names))]
            else:
                display_features = features[i,:]

            instance = Instance(np.ones((1,len(feature_names))), display_features)
            e = AdditiveExplanation(
                shap_values[i,-1],
                np.sum(shap_values[i,:]),
                shap_values[i,:-1],
                None,
                instance,
                link,
                Model(None, out_names),
                DenseData(np.ones((1,len(feature_names))), list(feature_names))
            )
            exps.append(e)
        return iml.visualize(exps, plot_cmap=plot_cmap)

def shorten_text(text, length_limit):
    if len(text) > length_limit:
        return text[:length_limit-3]+"..."
    else:
        return text

def joint_plot(ind, X, shap_value_matrix, feature_names=None, other_ind=None, other_auto_ind=0, alpha=1, axis_color="#000000", show=True):
    warnings.warn("shap.joint_plot is not yet finalized and should be used with caution")

    # convert from a DataFrame if we got one
    if str(type(X)) == "<class 'pandas.core.frame.DataFrame'>":
        if feature_names is None:
            feature_names = X.columns
        X = X.as_matrix()
    if feature_names is None:
        feature_names = ["Feature %d"%i for i in range(X.shape[1])]

    x = X[:,ind]
    xname = feature_names[ind]

    if other_ind is None:
        other_ind = interactions(X, shap_value_matrix, ind)[other_auto_ind]

    y = X[:,other_ind]
    yname = feature_names[other_ind]

    joint_shap_values = shap_value_matrix[:,ind] + shap_value_matrix[:,other_ind]

    if type(x[0]) == str:
        xnames = list(set(x))
        xnames.sort()
        name_map = {n: i for i,n in enumerate(xnames)}
        xv = [name_map[v] for v in x]
    else:
        xv = x

    if type(y[0]) == str:
        ynames = list(set(y))
        ynames.sort()
        name_map = {n: i for i,n in enumerate(ynames)}
        yv = [name_map[v] for v in y]
    else:
        yv = y

    sc = pl.scatter(x, y, s=20, c=joint_shap_values, edgecolor='', alpha=alpha, cmap=red_blue)
    pl.xlabel(xname, color=axis_color)
    pl.ylabel(yname, color=axis_color)
    cb = pl.colorbar(sc, label="Joint SHAP value")
    cb.set_alpha(1)
    cb.draw_all()

    pl.gca().tick_params(color=axis_color, labelcolor=axis_color)
    for spine in pl.gca().spines.values():
        spine.set_edgecolor(axis_color)
    if type(x[0]) == str:
        pl.xticks([name_map[n] for n in xnames], xnames, rotation='vertical')
    if show:
        pl.show()

def interaction_plot(ind, X, shap_value_matrix, feature_names=None, interaction_index=None, color="#ff0052", axis_color="#333333", alpha=1, title=None, dot_size=12, show=True):
    warnings.warn("shap.interaction_plot is deprecated in favor of shap.dependence_plot")

    # convert from a DataFrame if we got one
    if str(type(X)) == "<class 'pandas.core.frame.DataFrame'>":
        if feature_names is None:
            feature_names = X.columns
        X = X.as_matrix()

    x = X[:,ind]
    name = feature_names[ind]
    shap_values = shap_value_matrix[:,ind]
    if type(x[0]) == str:
        xnames = list(set(x))
        xnames.sort()
        name_map = {n: i for i,n in enumerate(xnames)}
        xv = [name_map[v] for v in x]
    else:
        xv = x

    if interaction_index is None:
        interaction_index = approx_interactions(X, shap_value_matrix, ind)[0]
    pl.scatter(xv, shap_values, s=dot_size, linewidth=0, c=X[:,interaction_index], cmap=red_blue, alpha=alpha)
    cb = pl.colorbar(label=feature_names[interaction_index])
    cb.set_alpha(1)
    cb.draw_all()
    # make the plot more readable
    pl.xlabel(name, color=axis_color)
    pl.ylabel("SHAP value for "+name, color=axis_color)
    if title != None:
        pl.title("SHAP plot for "+name, color=axis_color, fontsize=11)
    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('left')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    pl.gca().tick_params(color=axis_color, labelcolor=axis_color)
    for spine in pl.gca().spines.values():
        spine.set_edgecolor(axis_color)
    if type(x[0]) == str:
        pl.xticks([name_map[n] for n in xnames], xnames, rotation='vertical')
    if show:
        pl.show()

def plot(x, shap_values, name, color="#ff0052", axis_color="#333333", alpha=1, title=None, show=True):
    warnings.warn("shap.plot is deprecated in favor of shap.dependence_plot")

    if type(x[0]) == str:
        xnames = list(set(x))
        xnames.sort()
        name_map = {n: i for i,n in enumerate(xnames)}
        xv = [name_map[v] for v in x]
    else:
        xv = x

    pl.plot(xv, shap_values, ".", markersize=5, color=color, alpha=alpha, markeredgewidth=0)

    # make the plot more readable
    pl.xlabel(name, color=axis_color)
    pl.ylabel("SHAP value for "+name, color=axis_color)
    if title != None:
        pl.title("SHAP plot for "+name, color=axis_color, fontsize=11)
    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('left')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    pl.gca().tick_params(color=axis_color, labelcolor=axis_color)
    for spine in pl.gca().spines.values():
        spine.set_edgecolor(axis_color)
    if type(x[0]) == str:
        pl.xticks([name_map[n] for n in xnames], xnames, rotation='vertical')
    if show:
        pl.show()
