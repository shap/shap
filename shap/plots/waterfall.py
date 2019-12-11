import numpy as np
import warnings
try:
    import matplotlib.pyplot as pl
except ImportError:
    warnings.warn("matplotlib could not be loaded!")
    pass
from shap.plots import labels
from shap.common import safe_isinstance, format_value
from . import colors


def waterfall_plot(expected_value, shap_values, features=None, feature_names=None, max_display=10, show=True):
    """ Plots an explantion of a single prediction as a waterfall.
    
    Parameters
    ----------
    expected_value : float
        This is the reference value that the feature contributions start from. For SHAP values it should
        be the value of explainer.expected_value.

    shap_values : numpy.array
        Matrix of SHAP values (# features) or (# samples x # features). If this is a 1D array then a single
        force plot will be drawn, if it is a 2D array then a stacked force plot will be drawn.

    features : numpy.array
        Matrix of feature values (# features) or (# samples x # features). This provides the values of all the
        features, and should be the same shape as the shap_values argument.

    feature_names : list
        List of feature names (# features).

    max_display : str
        The maximum number of features to plot.

    show : bool
        Whether matplotlib.pyplot.show() is called before returning. Setting this to False allows the plot
        to be customized further after it has been created.
    """
    
    # unwrap pandas series
    if safe_isinstance(features, "pandas.core.series.Series"):
        if feature_names is None:
            feature_names = list(features.index)
        features = features.values
        
    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(len(shap_values))])
        
    num_features = min(max_display, len(shap_values))
    row_height=0.5
    rng = range(num_features - 1, -1, -1)
    order = np.argsort(-np.abs(shap_values))
    pos_lefts = []
    pos_inds = []
    pos_widths = []
    neg_lefts = []
    neg_inds = []
    neg_widths = []
    loc = expected_value + shap_values.sum()
    yticklabels = ["" for i in range(num_features + 1)]
    
    pl.gcf().set_size_inches(8, num_features * row_height + 1.5)

    if num_features == len(shap_values):
        num_individual = num_features
    else:
        num_individual = num_features - 1
    for i in range(num_individual):
        sval = shap_values[order[i]]
        loc -= sval
        if sval >= 0:
            pos_inds.append(rng[i])
            pos_widths.append(sval)
            pos_lefts.append(loc)
        else:
            neg_inds.append(rng[i])
            neg_widths.append(sval)
            neg_lefts.append(loc)
        if num_individual != num_features or i + 4 < num_individual:
            pl.plot([loc, loc], [rng[i] -1 - 0.4, rng[i] + 0.4], color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
        if features is None:
            yticklabels[rng[i]] = feature_names[order[i]]
        else:
            yticklabels[rng[i]] = feature_names[order[i]] + " = " + format_value(features[order[i]], "%0.03f")
    
    # add a last grouped feature to represent the impact of all the features we didn't show
    if num_features < len(shap_values):
        yticklabels[0] = "%d other features" % (len(shap_values) - num_features + 1)
        remaining_impact = expected_value - loc
        if remaining_impact < 0:
            pos_inds.append(0)
            pos_widths.append(remaining_impact)
            pos_lefts.append(loc)
            c = colors.red_rgb
        else:
            neg_inds.append(0)
            neg_widths.append(-remaining_impact)
            neg_lefts.append(loc + remaining_impact)
            c = colors.blue_rgb
    
    # draw invisible bars just for sizing the axes
    pl.barh(pos_inds, np.array(pos_widths)*1.001, left=pos_lefts, color=colors.red_rgb, alpha=0)
    pl.barh(neg_inds, np.array(neg_widths)*1.001, left=neg_lefts, color=colors.blue_rgb, alpha=0)
    
    
    head_length = 0.08
    bar_width = 0.8
    xlen = pl.xlim()[1] - pl.xlim()[0]
    fig = pl.gcf()
    ax = pl.gca()
    xticks = ax.get_xticks()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    bbox_to_xscale = xlen/width
    hl_scaled = bbox_to_xscale * head_length
    renderer = fig.canvas.get_renderer()
    
    # draw the positive arrows
    for i in range(len(pos_inds)):
        dist = pos_widths[i]
        arrow_obj = pl.arrow(
            pos_lefts[i], pos_inds[i], max(dist-hl_scaled, 0.000001), 0,
            head_length=min(dist, hl_scaled),
            color=colors.red_rgb, width=bar_width,
            head_width=bar_width
        )
        
        txt_obj = pl.text(
            pos_lefts[i] + 0.5*dist, pos_inds[i], format_value(pos_widths[i], '%+0.02f'),
            horizontalalignment='center', verticalalignment='center', color="white",
            fontsize=12
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)
        if text_bbox.width > arrow_bbox.width: 
            txt_obj.remove()
    
    # draw the negative arrows
    for i in range(len(neg_inds)):
        dist = neg_widths[i]
        
        arrow_obj = pl.arrow(
            neg_lefts[i], neg_inds[i], -max(-dist-hl_scaled, 0.000001), 0,
            head_length=min(-dist, hl_scaled),
            color=colors.blue_rgb, width=bar_width,
            head_width=bar_width
        )
        
        txt_obj = pl.text(
            neg_lefts[i] + 0.5*dist, neg_inds[i], format_value(neg_widths[i], '%+0.02f'),
            horizontalalignment='center', verticalalignment='center', color="white",
            fontsize=12
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)
        if text_bbox.width > arrow_bbox.width: 
            txt_obj.remove()

    pl.yticks(range(num_features), yticklabels, fontsize=13)
    
    # put horizontal lines for each feature row
    for i in range(num_features):
        pl.axhline(i, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
    
    # mark the prior expected value and the model prediction
    pl.axvline(expected_value, 0, 1/num_features, color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
    fx = expected_value + shap_values.sum()
    pl.axvline(fx, 0, 1, color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
    
    # clean up the main axis
    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('none')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    pl.gca().spines['left'].set_visible(False)
    pl.xlabel("Model output", fontsize=12)
    
    # remove the ticks that are closest to f(x) and E[f(X)]
    xmin,xmax = ax.get_xlim()
    xticks = ax.get_xticks()
    xticks = list(xticks)
    min_ind = 0
    min_diff = 1e10
    for i in range(len(xticks)):
        v = abs(xticks[i] - expected_value)
        if v < min_diff:
            min_diff = v
            min_ind = i
    xticks.pop(min_ind)
    min_ind = 0
    min_diff = 1e10
    for i in range(len(xticks)):
        v = abs(xticks[i] - fx)
        if v < min_diff:
            min_diff = v
            min_ind = i
    xticks.pop(min_ind)
    ax.set_xticks(xticks)
    ax.tick_params(labelsize=13)
    ax.set_xlim(xmin,xmax)

    # draw the f(x) and E[f(X)] ticks
    ax2=ax.twiny()
    ax2.set_xlim(xmin,xmax)
    ax2.set_xticks([fx, expected_value])
    ax2.set_xticklabels([format_value(fx, "%0.03f"), "$E[f(X)]$"], fontsize=12)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    ax3=ax2.twiny()
    ax3.set_xlim(xmin,xmax)
    ax3.set_xticks([expected_value + shap_values.sum()])
    ax3.set_xticklabels(["$f(x)$"], fontsize=12)
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    
    
    if show:
        pl.show()