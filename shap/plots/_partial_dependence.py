import matplotlib.pyplot as pl
import numpy as np
import pandas as pd

from .. import Explanation
from ..plots.colors import blue_rgb, light_blue_rgb, red_blue_transparent, red_rgb
from ..utils import convert_name


def compute_bounds(xmin, xmax, xv):
    """Handles any setting of xmax and xmin.

    Note that we handle None, float, or "percentile(float)" formats.
    """
    if xmin is not None or xmax is not None:
        if isinstance(xmin, str) and xmin.startswith("percentile"):
            xmin = np.nanpercentile(xv, float(xmin[11:-1]))
        if isinstance(xmax, str) and xmax.startswith("percentile"):
            xmax = np.nanpercentile(xv, float(xmax[11:-1]))

        if xmin is None or xmin == np.nanmin(xv):
            xmin = np.nanmin(xv) - (xmax - np.nanmin(xv))/20
        if xmax is None or xmax == np.nanmax(xv):
            xmax = np.nanmax(xv) + (np.nanmax(xv) - xmin)/20

    return (xmin, xmax)

def partial_dependence(ind, model, data, xmin="percentile(0)", xmax="percentile(100)",
                       npoints=None, feature_names=None, hist=True, model_expected_value=False,
                       feature_expected_value=False, shap_values=None,
                       ylabel=None, ice=True, ace_opacity=1, pd_opacity=1, pd_linewidth=2,
                       ace_linewidth='auto', ax=None, show=True):
    """A basic partial dependence plot function."""
    if isinstance(data, Explanation):
        features = data.data
        shap_values = data
    else:
        features = data

    # convert from DataFrames if we got any
    use_dataframe = False
    if isinstance(features, pd.DataFrame):
        if feature_names is None:
            feature_names = features.columns
        features = features.values
        use_dataframe = True

    if feature_names is None:
        feature_names = ["Feature %d" % i for i in range(features.shape[1])]

    # this is for a 1D partial dependence plot
    if type(ind) is not tuple:
        ind = convert_name(ind, None, feature_names)
        xv = features[:,ind]
        xmin, xmax = compute_bounds(xmin, xmax, xv)
        npoints = 100 if npoints is None else npoints
        xs = np.linspace(xmin, xmax, npoints)


        if ice:
            features_tmp = features.copy()
            ice_vals = np.zeros((npoints, features.shape[0]))
            for i in range(npoints):
                features_tmp[:,ind] = xs[i]
                if use_dataframe:
                    ice_vals[i,:] = model(pd.DataFrame(features_tmp, columns=feature_names))
                else:
                    ice_vals[i,:] = model(features_tmp)
            # if linewidth is None:
            #     linewidth = 1
            # if opacity is None:
            #     opacity = 0.5

        features_tmp = features.copy()
        vals = np.zeros(npoints)
        for i in range(npoints):
            features_tmp[:,ind] = xs[i]
            if use_dataframe:
                vals[i] = model(pd.DataFrame(features_tmp, columns=feature_names)).mean()
            else:
                vals[i] = model(features_tmp).mean()




        if ax is None:
            fig = pl.figure()
            ax1 = pl.gca()
        else:
            fig = pl.gcf()
            ax1 = pl.gca()

        #fig, ax1 = pl.subplots(figsize)
        ax2 = ax1.twinx()

        # the histogram of the data
        if hist:
            #n, bins, patches =
            ax2.hist(xv, 50, density=False, facecolor='black', alpha=0.1, range=(xmin, xmax))



        # ice line plot
        if ice:
            if ace_linewidth == "auto":
                ace_linewidth = min(1, 50/ice_vals.shape[1])
            ax1.plot(xs, ice_vals, color=light_blue_rgb, linewidth=ace_linewidth, alpha=ace_opacity)

        # the line plot
        ax1.plot(xs, vals, color=blue_rgb, linewidth=pd_linewidth, alpha=pd_opacity)

        ax2.set_ylim(0,features.shape[0])#ax2.get_ylim()[0], ax2.get_ylim()[1] * 4)
        ax1.set_xlabel(feature_names[ind], fontsize=13)
        if ylabel is None:
            if not ice:
                ylabel = "E[f(x) | "+ str(feature_names[ind]) + "]"
            else:
                ylabel = "f(x) | "+ str(feature_names[ind])
        ax1.set_ylabel(ylabel, fontsize=13)
        ax1.xaxis.set_ticks_position('bottom')
        ax1.yaxis.set_ticks_position('left')
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.tick_params(labelsize=11)

        ax2.xaxis.set_ticks_position('bottom')
        ax2.yaxis.set_ticks_position('left')
        ax2.yaxis.set_ticks([])
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)


        if feature_expected_value is not False:
            ax3=ax2.twiny()
            ax3.set_xlim(xmin,xmax)
            mval = xv.mean()
            ax3.set_xticks([mval])
            ax3.set_xticklabels(["E["+str(feature_names[ind])+"]"])
            ax3.spines['right'].set_visible(False)
            ax3.spines['top'].set_visible(False)
            ax3.tick_params(length=0, labelsize=11)
            ax1.axvline(mval, color="#999999", zorder=-1, linestyle="--", linewidth=1)

        if model_expected_value is not False or shap_values is not None:
            if model_expected_value is True:
                if use_dataframe:
                    model_expected_value = model(pd.DataFrame(features, columns=feature_names)).mean()
                else:
                    model_expected_value = model(features).mean()
            else:
                model_expected_value = shap_values.base_values
            ymin,ymax = ax1.get_ylim()
            ax4=ax2.twinx()
            ax4.set_ylim(ymin,ymax)
            ax4.set_yticks([model_expected_value])
            ax4.set_yticklabels(["E[f(x)]"])
            ax4.spines['right'].set_visible(False)
            ax4.spines['top'].set_visible(False)
            ax4.tick_params(length=0, labelsize=11)
            ax1.axhline(model_expected_value, color="#999999", zorder=-1, linestyle="--", linewidth=1)

        if shap_values is not None:
            # vals = shap_values.values[:, ind]
            # if shap_value_features is None:
            #     shap_value_features = features
            #     assert shap_values.shape == features.shape
            # #sample_ind = 18
            # vals = shap_values[:, ind]
            # if type(model_expected_value) is bool:
            #     if use_dataframe:
            #         model_expected_value = model(pd.DataFrame(features, columns=feature_names)).mean()
            #     else:
            #         model_expected_value = model(features).mean()
            # if isinstance(shap_value_features, pd.DataFrame):
            #     shap_value_features = shap_value_features.values
            markerline, stemlines, _ = ax1.stem(
                shap_values.data[:,ind], shap_values.base_values + shap_values.values[:, ind],
                bottom=shap_values.base_values,
                markerfmt="o", basefmt=" ",
            )
            stemlines.set_edgecolors([red_rgb if v > 0 else blue_rgb for v in vals])
            pl.setp(stemlines, 'zorder', -1)
            pl.setp(stemlines, 'linewidth', 2)
            pl.setp(markerline, 'color', "black")
            pl.setp(markerline, 'markersize', 4)

        if show:
            pl.show()
        else:
            return fig,ax1


    # this is for a 2D partial dependence plot
    else:
        ind0 = convert_name(ind[0], None, feature_names)
        ind1 = convert_name(ind[1], None, feature_names)
        xv0 = features[:,ind0]
        xv1 = features[:,ind1]

        xmin0 = xmin[0] if type(xmin) is tuple else xmin
        xmin1 = xmin[1] if type(xmin) is tuple else xmin
        xmax0 = xmax[0] if type(xmax) is tuple else xmax
        xmax1 = xmax[1] if type(xmax) is tuple else xmax

        xmin0, xmax0 = compute_bounds(xmin0, xmax0, xv0)
        xmin1, xmax1 = compute_bounds(xmin1, xmax1, xv1)
        npoints = 20 if npoints is None else npoints
        xs0 = np.linspace(xmin0, xmax0, npoints)
        xs1 = np.linspace(xmin1, xmax1, npoints)

        features_tmp = features.copy()
        x0 = np.zeros((npoints, npoints))
        x1 = np.zeros((npoints, npoints))
        vals = np.zeros((npoints, npoints))
        for i in range(npoints):
            for j in range(npoints):
                features_tmp[:,ind0] = xs0[i]
                features_tmp[:,ind1] = xs1[j]
                x0[i, j] = xs0[i]
                x1[i, j] = xs1[j]
                vals[i, j] = model(features_tmp).mean()

        fig = pl.figure()
        ax = fig.add_subplot(111, projection='3d')


#         x = y = np.arange(-3.0, 3.0, 0.05)
#         X, Y = np.meshgrid(x, y)
#         zs = np.array(fun(np.ravel(X), np.ravel(Y)))
#         Z = zs.reshape(X.shape)

        ax.plot_surface(x0, x1, vals, cmap=red_blue_transparent)

        ax.set_xlabel(feature_names[ind0], fontsize=13)
        ax.set_ylabel(feature_names[ind1], fontsize=13)
        ax.set_zlabel("E[f(x) | "+ str(feature_names[ind0]) + ", "+ str(feature_names[ind1]) + "]", fontsize=13)

        if show:
            pl.show()
        else:
            return fig, ax
