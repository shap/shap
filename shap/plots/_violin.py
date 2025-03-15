"""Summary plots of SHAP values (violin plot) across a whole dataset."""

import warnings

import matplotlib
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
from packaging import version
from scipy.stats import gaussian_kde

from ..utils._exceptions import DimensionError
from . import colors
from ._labels import labels

# TODO: simplify this when we drop support for matplotlib 3.9
if version.parse(matplotlib.__version__) >= version.parse("3.10"):
    ORIENTATION_KWARG = dict(orientation="horizontal")
else:
    ORIENTATION_KWARG = dict(vert=False)  # type: ignore[dict-item]


# TODO: remove unused title argument / use title argument
# TODO: Add support for hclustering based explanations where we sort the leaf order by magnitude and then show the dendrogram to the left
def violin(
    shap_values,
    features=None,
    feature_names=None,
    max_display=None,
    plot_type="violin",
    color=None,
    axis_color="#333333",
    title=None,
    alpha=1,
    show=True,
    sort=True,
    color_bar=True,
    plot_size="auto",
    layered_violin_max_num_bins=20,
    class_names=None,
    class_inds=None,
    color_bar_label=labels["FEATURE_VALUE"],
    cmap=colors.red_blue,
    use_log_scale=False,
):
    """Create a SHAP violin plot, colored by feature values when they are provided.

    Parameters
    ----------
    shap_values : Explanation, or numpy.array
        For single output explanations, this is a matrix of SHAP values (# samples x # features).

    features : numpy.array or pandas.DataFrame or list
        Matrix of feature values (# samples x # features) or a ``feature_names`` list as
        shorthand.

    feature_names : list
        Names of the features (length: # features).

    max_display : int
        How many top features to include in the plot (default is 20).

    plot_type : "violin", or "layered_violin".
        What type of summary plot to produce. A "layered_violin" plot shows the
        distribution of the SHAP values of each variable. A "violin" plot is the same,
        except with outliers drawn as scatter points.

    color_bar : bool
        Whether to draw the color bar (legend).

    show : bool
        Whether :external+mpl:func:`matplotlib.pyplot.show()` is called before returning.
        Setting this to ``False`` allows the plot
        to be customized further after it has been created.

    plot_size : "auto" (default), float, (float, float), or None
        What size to make the plot. By default, the size is auto-scaled based on the number of
        features that are being displayed. Passing a single float will cause each row to be that
        many inches high. Passing a pair of floats will scale the plot by that
        number of inches. If ``None`` is passed, then the size of the current figure will be left
        unchanged.

    Examples
    --------
    See `violin plot examples <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/violin.html>`_.

    """
    if title is not None:
        warnings.warn("The `title` argument is unused and will be removed in a future release.", DeprecationWarning)
    # support passing an explanation object
    if str(type(shap_values)).endswith("Explanation'>"):
        shap_exp = shap_values
        shap_values = shap_exp.values
        if features is None:
            features = shap_exp.data
        if feature_names is None:
            feature_names = shap_exp.feature_names
        # if out_names is None: # TODO: waiting for slicer support of this
        #     out_names = shap_exp.output_names

    if isinstance(shap_values, list):
        emsg = "Violin plots don't support multi-output explanations! Use 'shap.plots.bar` instead."
        raise TypeError(emsg)

    if plot_type is None:
        plot_type = "violin"
    if plot_type not in {"violin", "layered_violin"}:
        emsg = f"plot_type: Expected one of ('violin','layered_violin'), received {plot_type} instead."
        raise ValueError(emsg)

    assert len(shap_values.shape) != 1, "Violin summary plots need a matrix of shap_values, not a vector."

    # default color:
    if color is None:
        if plot_type == "layered_violin":
            color = "coolwarm"
        else:
            color = colors.blue_rgb

    # convert from a DataFrame or other types
    if isinstance(features, pd.DataFrame):
        if feature_names is None:
            feature_names = features.columns
        features = features.values
    elif isinstance(features, list):
        if feature_names is None:
            feature_names = features
        features = None
    elif (features is not None) and len(features.shape) == 1 and feature_names is None:
        feature_names = features
        features = None

    num_features = shap_values.shape[1]

    if features is not None:
        shape_msg = "The shape of the shap_values matrix does not match the shape of the provided data matrix."
        if num_features - 1 == features.shape[1]:
            shape_msg += (
                " Perhaps the extra column in the shap_values matrix is the "
                "constant offset? If so, just pass shap_values[:,:-1]."
            )
            raise DimensionError(shape_msg)
        if num_features != features.shape[1]:
            raise DimensionError(shape_msg)

    if feature_names is None:
        feature_names = np.array([labels["FEATURE"] % str(i) for i in range(num_features)])

    if use_log_scale:
        pl.xscale("symlog")

    if max_display is None:
        max_display = 20

    if sort:
        # order features by the sum of their effect magnitudes
        feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
        feature_order = feature_order[-min(max_display, len(feature_order)) :]
    else:
        feature_order = np.flip(np.arange(min(max_display, num_features)), 0)

    row_height = 0.4
    if plot_size == "auto":
        pl.gcf().set_size_inches(8, len(feature_order) * row_height + 1.5)
    elif type(plot_size) in (list, tuple):
        pl.gcf().set_size_inches(plot_size[0], plot_size[1])
    elif plot_size is not None:
        pl.gcf().set_size_inches(8, len(feature_order) * plot_size + 1.5)
    pl.axvline(x=0, color="#999999", zorder=-1)

    if plot_type == "violin":
        for pos in range(len(feature_order)):
            pl.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)

        if features is not None:
            global_low = np.nanpercentile(shap_values[:, : len(feature_names)].flatten(), 1)
            global_high = np.nanpercentile(shap_values[:, : len(feature_names)].flatten(), 99)
            for pos, i in enumerate(feature_order):
                shaps = shap_values[:, i]
                shap_min, shap_max = np.min(shaps), np.max(shaps)
                rng = shap_max - shap_min
                xs = np.linspace(np.min(shaps) - rng * 0.2, np.max(shaps) + rng * 0.2, 100)
                if np.std(shaps) < (global_high - global_low) / 100:
                    ds = gaussian_kde(shaps + np.random.randn(len(shaps)) * (global_high - global_low) / 100)(xs)
                else:
                    ds = gaussian_kde(shaps)(xs)
                ds /= np.max(ds) * 3

                values = features[:, i]
                # window_size = max(10, len(values) // 20)
                smooth_values = np.zeros(len(xs) - 1)
                sort_inds = np.argsort(shaps)
                trailing_pos = 0
                leading_pos = 0
                running_sum = 0
                back_fill = 0
                for j in range(len(xs) - 1):
                    while leading_pos < len(shaps) and xs[j] >= shaps[sort_inds[leading_pos]]:
                        running_sum += values[sort_inds[leading_pos]]
                        leading_pos += 1
                        if leading_pos - trailing_pos > 20:
                            running_sum -= values[sort_inds[trailing_pos]]
                            trailing_pos += 1
                    if leading_pos - trailing_pos > 0:
                        smooth_values[j] = running_sum / (leading_pos - trailing_pos)
                        for k in range(back_fill):
                            smooth_values[j - k - 1] = smooth_values[j]
                    else:
                        back_fill += 1

                # Get nan values:
                nan_mask = np.isnan(values)

                # Trim the value and color range to percentiles
                vmin, vmax, cvals = _trim_crange(values, nan_mask)

                # plot the nan values in the interaction feature as grey
                pl.scatter(
                    shaps[nan_mask],
                    np.ones(shap_values[nan_mask].shape[0]) * pos,
                    color="#777777",
                    s=9,
                    alpha=alpha,
                    linewidth=0,
                    zorder=1,
                )
                # plot the non-nan values colored by the trimmed feature value
                pl.scatter(
                    shaps[np.invert(nan_mask)],
                    np.ones(shap_values[np.invert(nan_mask)].shape[0]) * pos,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    s=9,
                    c=cvals,
                    alpha=alpha,
                    linewidth=0,
                    zorder=1,
                )
                # smooth_values -= nxp.nanpercentile(smooth_values, 5)
                # smooth_values /= np.nanpercentile(smooth_values, 95)
                smooth_values -= vmin
                if vmax - vmin > 0:
                    smooth_values /= vmax - vmin
                for i in range(len(xs) - 1):
                    if ds[i] > 0.05 or ds[i + 1] > 0.05:
                        pl.fill_between(
                            [xs[i], xs[i + 1]],
                            [pos + ds[i], pos + ds[i + 1]],
                            [pos - ds[i], pos - ds[i + 1]],
                            color=colors.red_blue_no_bounds(smooth_values[i]),
                            zorder=2,
                        )

        else:
            parts = pl.violinplot(
                shap_values[:, feature_order],
                range(len(feature_order)),
                points=200,
                **ORIENTATION_KWARG,  # type: ignore[arg-type]
                widths=0.7,
                showmeans=False,
                showextrema=False,
                showmedians=False,
            )

            for pc in parts["bodies"]:  # type: ignore
                pc.set_facecolor(color)
                pc.set_edgecolor("none")
                pc.set_alpha(alpha)

    elif plot_type == "layered_violin":  # courtesy of @kodonnell
        num_x_points = 200
        bins = (
            np.linspace(0, features.shape[0], layered_violin_max_num_bins + 1).round(0).astype("int")
        )  # the indices of the feature data corresponding to each bin
        shap_min, shap_max = np.min(shap_values), np.max(shap_values)
        x_points = np.linspace(shap_min, shap_max, num_x_points)

        # loop through each feature and plot:
        for pos, ind in enumerate(feature_order):
            # decide how to handle: if #unique < layered_violin_max_num_bins then split by unique value, otherwise use bins/percentiles.
            # to keep simpler code, in the case of uniques, we just adjust the bins to align with the unique counts.
            feature = features[:, ind]
            unique, counts = np.unique(feature, return_counts=True)
            if unique.shape[0] <= layered_violin_max_num_bins:
                order = np.argsort(unique)
                thesebins = np.cumsum(counts[order])
                thesebins = np.insert(thesebins, 0, 0)
            else:
                thesebins = bins
            nbins = thesebins.shape[0] - 1
            # order the feature data so we can apply percentiling
            order = np.argsort(feature)
            # x axis is located at y0 = pos, with pos being there for offset
            # y0 = np.ones(num_x_points) * pos
            # calculate kdes:
            ys = np.zeros((nbins, num_x_points))
            for i in range(nbins):
                # get shap values in this bin:
                shaps = shap_values[order[thesebins[i] : thesebins[i + 1]], ind]
                # if there's only one element, then we can't
                if shaps.shape[0] == 1:
                    warnings.warn(
                        f"Not enough data in bin #{i} for feature {feature_names[ind]}, so it'll be ignored."
                        " Try increasing the number of records to plot."
                    )
                    # to ignore it, just set it to the previous y-values (so the area between them will be zero). Not ys is already 0, so there's
                    # nothing to do if i == 0
                    if i > 0:
                        ys[i, :] = ys[i - 1, :]
                    continue
                # save kde of them: note that we add a tiny bit of gaussian noise to avoid singular matrix errors
                ys[i, :] = gaussian_kde(shaps + np.random.normal(loc=0, scale=0.001, size=shaps.shape[0]))(x_points)
                # scale it up so that the 'size' of each y represents the size of the bin. For continuous data this will
                # do nothing, but when we've gone with the unique option, this will matter - e.g. if 99% are male and 1%
                # female, we want the 1% to appear a lot smaller.
                size = thesebins[i + 1] - thesebins[i]
                bin_size_if_even = features.shape[0] / nbins
                relative_bin_size = size / bin_size_if_even
                ys[i, :] *= relative_bin_size
            # now plot 'em. We don't plot the individual strips, as this can leave whitespace between them.
            # instead, we plot the full kde, then remove outer strip and plot over it, etc., to ensure no
            # whitespace
            ys = np.cumsum(ys, axis=0)
            width = 0.8
            scale = ys.max() * 2 / width  # 2 is here as we plot both sides of x axis
            for i in range(nbins - 1, -1, -1):
                y = ys[i, :] / scale
                c = (
                    pl.get_cmap(color)(i / (nbins - 1)) if color in pl.colormaps else color
                )  # if color is a cmap, use it, otherwise use a color
                pl.fill_between(x_points, pos - y, pos + y, facecolor=c, edgecolor="face")
        pl.xlim(shap_min, shap_max)

    # draw the color bar
    if (
        color_bar
        and features is not None
        and plot_type != "bar"
        and (plot_type != "layered_violin" or color in pl.colormaps)
    ):
        import matplotlib.cm as cm

        m = cm.ScalarMappable(cmap=cmap if plot_type != "layered_violin" else pl.get_cmap(color))
        m.set_array([0, 1])
        cb = pl.colorbar(m, ax=pl.gca(), ticks=[0, 1], aspect=80)
        cb.set_ticklabels([labels["FEATURE_VALUE_LOW"], labels["FEATURE_VALUE_HIGH"]])
        cb.set_label(color_bar_label, size=12, labelpad=0)
        cb.ax.tick_params(labelsize=11, length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)  # type: ignore
        # bbox = cb.ax.get_window_extent().transformed(pl.gcf().dpi_scale_trans.inverted())
        # cb.ax.set_aspect((bbox.height - 0.9) * 20)
        # cb.draw_all()

    pl.gca().xaxis.set_ticks_position("bottom")
    pl.gca().yaxis.set_ticks_position("none")
    pl.gca().spines["right"].set_visible(False)
    pl.gca().spines["top"].set_visible(False)
    pl.gca().spines["left"].set_visible(False)
    pl.gca().tick_params(color=axis_color, labelcolor=axis_color)
    pl.yticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=13)
    pl.gca().tick_params("y", length=20, width=0.5, which="major")
    pl.gca().tick_params("x", labelsize=11)
    pl.ylim(-1, len(feature_order))
    pl.xlabel(labels["VALUE"], fontsize=13)

    if show:
        pl.show()


def _trim_crange(values, nan_mask):
    """Trim the color range, but prevent the color range from collapsing."""
    # Get vmin and vmax as 5. and 95. percentiles
    vmin = np.nanpercentile(values, 5)
    vmax = np.nanpercentile(values, 95)
    if vmin == vmax:  # if percentile range is equal, take 1./99. perc.
        vmin = np.nanpercentile(values, 1)
        vmax = np.nanpercentile(values, 99)
        if vmin == vmax:  # if still equal, use min/max
            vmin = np.min(values)
            vmax = np.max(values)

    if vmin > vmax:  # fixes rare numerical precision issues
        vmin = vmax

    # Get color values depending on value range
    cvals = values[np.invert(nan_mask)].astype(np.float64)
    cvals_imp = cvals.copy()
    cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
    cvals[cvals_imp > vmax] = vmax
    cvals[cvals_imp < vmin] = vmin

    return vmin, vmax, cvals


def shorten_text(text, length_limit):
    if len(text) > length_limit:
        return text[: length_limit - 3] + "..."
    else:
        return text
